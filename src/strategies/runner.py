# Path: src/strategies/runner.py
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from decimal import Decimal
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.data_feed import SpotFeed
from src.backtesting.sim_connector import SimConnector
from src.backtesting.synth import make_synth_1m
from src.broker.interface import OrderRequest, Tick
from src.config import settings
from src.data.broker_source import BrokerDataSource
from src import diagnostics
from src.diagnostics.metrics import metrics, runtime_metrics, record_trade
from src.execution.broker_executor import BrokerOrderExecutor
from src.execution.micro_filters import evaluate_micro
from src.execution.order_executor import OrderManager, OrderReconciler
from src.state import StateStore
from src.features.indicators import atr_pct
from src.logs.journal import Journal
from src.options.instruments_cache import InstrumentsCache
from src.options.resolver import OptionResolver
from src.risk import guards, risk_gates
from src.risk.position_sizing import PositionSizer
from src.risk.cooldown import LossCooldownManager
from src.risk.greeks import (  # noqa: F401
    estimate_greeks_from_mid,
    next_weekly_expiry_ist,
)
from src.risk import limits
from src.risk.limits import Exposure, LimitConfig, RiskEngine
from src.signals.patches import resolve_atr_band
from src.strategies.atr_gate import check_atr
from src.strategies.registry import init_default_registries
from src.strategies.scalping_strategy import compute_score, _log_throttled
from src.strategies.strategy_config import (
    StrategyConfig,
    resolve_config_path,
    try_load,
)
from src.utils import strike_selector
from src.utils.env import env_flag
from src.utils.events import EventWindow, load_calendar
from src.utils.freshness import compute as compute_freshness
from src.utils.indicators import calculate_adx, calculate_bb_width
from src.utils.market_time import (
    is_market_open,
    prev_session_bounds,
    prev_session_last_20m,
)
from src.utils.time_windows import TZ, floor_to_minute
from src.utils.logging_tools import structured_debug_handler

from .warmup import check as warmup_check, required_bars

# Optional broker SDK (graceful if not installed)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Optional live data source (graceful if not present)
try:
    from src.data.source import LiveKiteSource, get_option_quote_safe  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    LiveKiteSource = None  # type: ignore

    def get_option_quote_safe(*args: Any, **kwargs: Any) -> tuple[None, str]:  # type: ignore
        return None, "no_quote"


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if structured_debug_handler not in logger.handlers:
    logger.addHandler(structured_debug_handler)

# =========================== Cadence controller ============================


@dataclass
class CadenceController:
    """Adaptive evaluation cadence based on market activity."""

    min_interval: float = 0.3
    max_interval: float = 1.5
    interval: float = 0.3
    step: float = 0.3

    def __post_init__(self) -> None:
        self.min_interval = float(self.min_interval)
        self.max_interval = float(self.max_interval)
        self.interval = float(self.interval)
        self.step = float(self.step)
        if self.min_interval <= 0:
            raise ValueError("min_interval must be > 0")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        self.interval = max(self.min_interval, min(self.max_interval, self.interval))

    def update(self, atr_pct: float | None, tick_age: float, breaker_open: bool) -> float:
        """Return next evaluation interval."""
        if breaker_open:
            self.interval = self.max_interval
            return self.interval

        atr = float(atr_pct or 0.0)
        # Normalize ATR%% to 0-1 assuming 0-1% range.
        norm = max(0.0, min(1.0, atr))
        self.interval = self.max_interval - norm * (self.max_interval - self.min_interval)
        if tick_age > 1.0:
            self.interval = min(self.max_interval, self.interval + self.step)
        self.interval = max(self.min_interval, min(self.max_interval, self.interval))
        return self.interval


# ============================== Orchestrator ==============================


class Orchestrator:
    """Route ticks to a strategy and execute resulting orders."""

    def __init__(
        self,
        data_source: BrokerDataSource,
        executor: BrokerOrderExecutor,
        on_tick: Callable[[Tick], Iterable[OrderRequest | Mapping[str, Any]] | None],
        *,
        max_ticks: int = 1000,
        min_eval_interval_s: float = 0.0,
        stale_tick_timeout_s: float = 3.0,
        on_stale: Optional[Callable[[], None]] = None,
        risk_config: Optional[guards.RiskConfig] = None,
    ) -> None:
        self.data_source = data_source
        self.executor = executor
        self.on_tick = on_tick
        self.min_eval_interval_s = float(min_eval_interval_s)
        self.stale_tick_timeout_s = float(stale_tick_timeout_s)
        self.on_stale = on_stale
        self._tick_queue: Deque[Tick] = deque(maxlen=max_ticks)
        self._last_eval = 0.0
        self._last_tick = time.time()
        self._paused = False
        self._running = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._risk = guards.RiskGuards(risk_config)
        if self.min_eval_interval_s > 0:
            self._cadence = CadenceController(
                min_interval=settings.cadence_min_interval_s,
                max_interval=settings.cadence_max_interval_s,
                interval=settings.cadence_min_interval_s,
                step=settings.cadence_interval_step_s,
            )
            self.min_eval_interval_s = self._cadence.interval
        else:
            self._cadence = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start processing ticks from the data source."""
        if self._running:
            return
        self._running = True
        self.data_source.set_tick_callback(self._enqueue_tick)
        self.data_source.start()
        self._worker.start()
        self._watchdog.start()

    def stop(self) -> None:
        """Stop processing ticks."""
        self._running = False
        self.data_source.stop()
        self._worker.join(timeout=1)
        self._watchdog.join(timeout=1)

    # ------------------------------------------------------------------
    def _enqueue_tick(self, tick: Tick) -> None:
        now = time.time()
        self._last_tick = now
        from src.utils.health import STATE  # local import to avoid circular

        STATE.last_tick_ts = now
        self._paused = False
        self._tick_queue.append(tick)
        metrics.inc_ticks()
        metrics.set_queue_depth(len(self._tick_queue))

    def _run(self) -> None:
        while self._running:
            try:
                tick = self._tick_queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue
            if self._paused:
                continue
            now = time.time()
            if now - self._last_eval < self.min_eval_interval_s:
                continue
            self._last_eval = now
            self.step(tick)
            if self._cadence is not None:
                self._update_cadence(tick)

    def _watchdog_loop(self) -> None:
        while self._running:
            time.sleep(self.stale_tick_timeout_s)
            if time.time() - self._last_tick > self.stale_tick_timeout_s:
                if not self._paused:
                    self._paused = True
                    if self.on_stale:
                        self.on_stale()

    def _update_cadence(self, tick: Tick) -> None:
        """Adjust evaluation cadence based on market conditions."""
        runner = getattr(self.on_tick, "__self__", None)
        atr = None
        if runner is not None and getattr(runner, "last_plan", None):
            atr = runner.last_plan.get("atr_pct")
        try:
            state = (
                self.executor.api_health().get("orders", {}).get("state", "closed")
            )
            breaker_open = state != "closed"
        except Exception:
            breaker_open = False
        age = max(0.0, time.time() - float(getattr(tick, "ts", time.time())))
        self.min_eval_interval_s = self._cadence.update(atr, age, breaker_open)

    # ------------------------------------------------------------------
    def step(self, tick: Tick, budget_s: float | None = None) -> None:
        """Process a single tick within an optional time budget."""
        # Light upkeep to ensure ATM option tokens stay subscribed
        try:
            ds = getattr(self, "data_source", None)
            if ds:
                if hasattr(ds, "auto_resubscribe_atm"):
                    ds.auto_resubscribe_atm()
                if hasattr(ds, "ensure_atm_tokens"):
                    ds.ensure_atm_tokens()
        except Exception:
            logging.getLogger(__name__).warning("ATM upkeep failed", exc_info=True)

        start = time.time()
        result = self.on_tick(tick)
        if not result:
            return
        metrics.inc_signal()
        elapsed = time.time() - start
        metrics.observe_latency(elapsed * 1000)
        if budget_s is not None and elapsed > budget_s:
            logging.getLogger(self.__class__.__name__).warning(
                "step exceeded budget %.3fs > %.3fs", elapsed, budget_s
            )
        orders = (
            result
            if isinstance(result, Iterable)
            and not isinstance(result, (dict, OrderRequest))
            else [result]
        )
        for order in orders:
            if not isinstance(order, OrderRequest):
                continue
            if not self._risk_ok(order):
                metrics.inc_orders(rejected=1)
                continue
            if BrokerOrderExecutor._kill_switch_engaged():  # type: ignore[attr-defined]
                self._running = False
                break
            self.executor.place_order(order)
            metrics.inc_orders(placed=1)

    def _risk_ok(self, _order: OrderRequest) -> bool:
        return self._risk.ok_to_trade()


# ================================ Models ================================


@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0
    day_realized_pnl: float = 0.0
    # If set, trading is halted until this timestamp after hitting loss streak
    loss_cooldown_until: Optional[datetime] = None


@dataclass
class _RiskCheckState:
    """Internal state for lightweight risk guards."""

    realised_loss: float = 0.0


# Sentinel used when risk gates are intentionally skipped
RISK_GATES_SKIPPED = object()


class _LogGate:
    """Simple per-key emission gate for throttling log noise."""

    def __init__(self, default_interval: float = 0.0) -> None:
        self.default_interval = float(default_interval)
        self._last_emit: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_emit(self, key: str, *, interval: float | None = None) -> bool:
        """Return ``True`` when ``key`` may be emitted again."""

        window = float(self.default_interval if interval is None else interval)
        now = time.time()
        with self._lock:
            last = self._last_emit.get(key)
            if window <= 0.0 or last is None or (now - last) >= window:
                self._last_emit[key] = now
                return True
            return False


# ============================== Runner =================================


class StrategyRunner:
    """
    Pipeline: data → signal → risk gates → sizing → execution.
    TelegramController is provided by main.py; here we only consume it.
    """

    _SINGLETON: ClassVar["StrategyRunner" | None] = None

    # ---------------- init ----------------
    def __init__(
        self,
        kite: Optional[KiteConnect] = None,
        telegram_controller: Any = None,
        strategy_cfg_path: str | None = None,
    ) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.kite = kite

        if telegram_controller is None:
            raise RuntimeError("TelegramController must be provided to StrategyRunner.")
        # keep both names for old controllers
        self.telegram = telegram_controller
        self.telegram_controller = telegram_controller

        StrategyRunner._SINGLETON = self
        self._ohlc_cache: Optional[pd.DataFrame] = None
        self.ready = False
        self._warm = None
        self._fresh = None
        self._score_items: dict[str, float] | None = None
        self._score_total: float | None = None
        self._last_bar_count: int = 0
        self._warmup_log_ts: float = 0.0

        self.settings = settings

        # Event guard configuration
        self.events_path = resolve_config_path(
            "EVENTS_CONFIG_FILE", "config/events.yaml"
        )
        self.event_guard_enabled = env_flag("EVENT_GUARD_ENABLED")
        if os.path.exists(self.events_path):
            self.event_cal = load_calendar(self.events_path)
            self._event_last_mtime = os.path.getmtime(self.events_path)
        else:
            self.event_cal = None
            self._event_last_mtime = None
        self._event_post_widen: float = 0.0

        # strategy configuration
        self.strategy_cfg: StrategyConfig = try_load(strategy_cfg_path, None)
        self.strategy_config_path = self.strategy_cfg.source_path
        self.tz = ZoneInfo(self.strategy_cfg.tz)

        self.under_symbol = str(getattr(settings.instruments, "trade_symbol", "NIFTY"))
        self.lot_size = int(getattr(settings.instruments, "nifty_lot_size", 50))
        self.instruments = InstrumentsCache(self.kite)
        self.option_resolver = OptionResolver(self.instruments, self.kite)
        self._last_option: Optional[dict] = None

        self._position_sizer = PositionSizer()

        self.risk_engine = RiskEngine(
            LimitConfig(
                tz=getattr(self.settings, "TZ", "Asia/Kolkata"),
                max_daily_dd_R=getattr(self.settings, "MAX_DAILY_DD_R", 2.5),
                max_trades_per_session=settings.risk.max_trades_per_day,
                max_lots_per_symbol=settings.risk.max_lots_per_symbol,
                max_notional_rupees=settings.risk.max_notional_rupees,
                exposure_basis=settings.EXPOSURE_BASIS,
                max_gamma_mode_lots=getattr(self.settings, "MAX_GAMMA_MODE_LOTS", 2),
                max_portfolio_delta_units=getattr(
                    self.settings, "MAX_PORTFOLIO_DELTA_UNITS", 100
                ),
                max_portfolio_delta_units_gamma=getattr(
                    self.settings, "MAX_PORTFOLIO_DELTA_UNITS_GAMMA", 60
                ),
                roll10_pause_R=getattr(self.settings, "ROLL10_PAUSE_R", -0.2),
                roll10_pause_minutes=getattr(self.settings, "ROLL10_PAUSE_MIN", 60),
                cooloff_losses=settings.risk.consecutive_loss_limit,
                cooloff_minutes=getattr(self.settings, "COOLOFF_MINUTES", 45),
                skip_next_open_after_two_daily_caps=True,
            )
        )

        self._flatten_time = self._parse_hhmm(self.risk_engine.cfg.eod_flatten_hhmm)

        self._risk_state = _RiskCheckState()
        self._last_trade_time = self.now_ist
        self._auto_relax_active = False

        # Factories that return your existing objects WITHOUT changing their classes
        def _make_strategy():
            from src.strategies.scalping_strategy import EnhancedScalpingStrategy

            return EnhancedScalpingStrategy()

        def _make_data_kite():
            from src.data.source import LiveKiteSource

            return LiveKiteSource(self.kite)

        def _make_connector_kite():
            from src.execution.order_executor import OrderExecutor

            return OrderExecutor(
                kite=self.kite,
                telegram_controller=self.telegram,
                on_trade_closed=self._on_trade_closed,
            )

        def _make_connector_shadow():
            from src.execution.order_executor import OrderExecutor

            shadow_settings = self.settings
            try:
                setattr(shadow_settings, "enable_live_trading", False)
            except Exception:
                pass
            return OrderExecutor(
                kite=None,
                telegram_controller=self.telegram,
                on_trade_closed=self._on_trade_closed,
            )

        self.components = init_default_registries(
            self.settings,
            make_strategy=_make_strategy,
            make_data_kite=_make_data_kite,
            make_connector_kite=_make_connector_kite,
            make_connector_shadow=_make_connector_shadow,
        )

        self.state_store = StateStore(
            getattr(self.settings, "STATE_STORE_PATH", "data/state.json")
        )
        self.strategy = self.components.strategy
        try:
            setattr(self.strategy, "runner", self)
        except Exception:
            pass
        self.data_source = self.components.data_provider
        self.source = self.components.data_provider
        if hasattr(self.data_source, "connect"):
            try:
                self.data_source.connect()
            except Exception as e:
                self.log.debug("Data source connect failed: %s", e)
        self.order_executor = self.components.order_connector
        if hasattr(self.order_executor, "state_store") and not self.order_executor.state_store:
            self.order_executor.state_store = self.state_store
        snap = self.state_store.snapshot()
        live_orders: Dict[str, Any] = {}
        if getattr(self.order_executor, "kite", None):
            try:  # pragma: no cover - network
                for o in self.order_executor.kite.orders() or []:
                    tag = str(o.get("tag") or "")
                    if tag and str(o.get("status", "")).upper() in {"OPEN", "TRIGGER PENDING"}:
                        live_orders[tag] = o
            except Exception:
                self.log.debug("live order fetch failed", exc_info=True)
        for tag, order in live_orders.items():
            if tag not in snap.open_orders:
                try:
                    self.order_executor.kite.cancel_order(
                        variety=order.get("variety", "regular"),
                        order_id=order.get("order_id"),
                    )
                    self.log.info("cancelled orphan order %s", tag)
                except Exception:
                    self.log.debug("cancel orphan failed: %s", tag, exc_info=True)
        for oid, data in snap.open_orders.items():
            if oid in live_orders:
                try:
                    self.order_executor.restore_record(oid, data)
                except Exception:
                    self.log.debug("restore record failed: %s", oid, exc_info=True)
            else:
                self.state_store.remove_order(oid)
        if getattr(self.order_executor, "kite", None):
            try:  # pragma: no cover - network
                positions = self.order_executor.get_positions_kite()
                for sym, info in positions.items():
                    qty = int(info.get("quantity") or 0)
                    if qty:
                        self.state_store.record_position(sym, info)
            except Exception:
                self.log.debug("live positions fetch failed", exc_info=True)
        try:
            pos_snap = self.state_store.snapshot().positions
            recs = getattr(self.order_executor, "get_active_orders", lambda: [])()
            for rec in recs:
                info = pos_snap.get(rec.symbol, {})
                ltp = float(info.get("last_price") or info.get("average_price") or 0.0)
                if ltp <= 0:
                    continue
                atr = rec.r_value / (rec.trailing_mult or 1.0)
                try:
                    self.order_executor.update_trailing_stop(
                        rec.record_id, current_price=ltp, atr=atr
                    )
                except Exception:
                    self.log.debug("trail restore failed: %s", rec.symbol, exc_info=True)
        except Exception:
            self.log.debug("trail resume failed", exc_info=True)
        self.executor = self.order_executor
        self.order_manager = OrderManager(
            self.order_executor.place_order,
            kite=getattr(self.order_executor, "kite", None),
            tick_size=getattr(self.order_executor, "tick_size", 0.05),
        )
        self.reconciler = OrderReconciler(
            getattr(self.order_executor, "kite", None), self.order_executor, self.log
        )
        self.journal = Journal.open(
            getattr(self.settings, "JOURNAL_DB_PATH", "data/journal.sqlite")
        )
        self.order_executor.journal = self.journal

        rehydrated = self.journal.rehydrate_open_legs()
        if rehydrated:
            self.log.info(f"Journal rehydrate: {len(rehydrated)} open legs")
            for leg in rehydrated:
                fsm = self.order_executor.get_or_create_fsm(leg["trade_id"])
                self.order_executor.attach_leg_from_journal(fsm, leg)
            self.reconciler.step(self.now_ist)
        self._active_names = self.components.names
        self.strategy_name = self._active_names.get("strategy", "")
        self.data_provider_name = self._active_names.get("data_provider", "")

        # Trading window
        self._start_time = self._parse_hhmm(settings.risk.trading_window_start)
        self._end_time = self._parse_hhmm(settings.risk.trading_window_end)

        # Data warmup (best effort)
        self._last_fetch_ts: float = 0.0
        if self.kite is not None and self.data_source is not None:
            try:
                token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
                warm_min = int(
                    getattr(settings, "WARMUP_MIN_BARS", getattr(settings, "warmup_min_bars", 0))
                )
                if (
                    warm_min > 0
                    and hasattr(self.data_source, "have_min_bars")
                    and not self.data_source.have_min_bars(warm_min)
                ):
                    self.log.info(
                        "Warmup mode: live bars (broker OHLC unavailable)",
                    )
                if token > 0:
                    now_ist = floor_to_minute(self._now_ist())
                    if is_market_open(now_ist):
                        start = now_ist - timedelta(minutes=20)
                        end = now_ist
                    else:
                        start, end = prev_session_last_20m(now_ist)
                    self.plan_probe_window = (start.isoformat(), end.isoformat())
                    df = self.data_source.fetch_ohlc_df(
                        token=token, start=start, end=end, timeframe="minute"
                    )
                    if df.empty and not is_market_open(now_ist):
                        prev_start, prev_end = prev_session_last_20m(
                            now_ist - timedelta(days=1)
                        )
                        df = self.data_source.fetch_ohlc_df(
                            token=token,
                            start=prev_start,
                            end=prev_end,
                            timeframe="minute",
                        )
                        if not df.empty:
                            self.log.warning(
                                "startup: first probe empty; used holiday fallback window=%s..%s",
                                prev_start,
                                prev_end,
                            )
                            self.plan_probe_window = (
                                prev_start.isoformat(),
                                prev_end.isoformat(),
                            )
                    if df.empty:
                        self.log.warning(
                            "instrument_token %s returned no historical data; falling back",
                            token,
                        )
            except Exception as e:
                self.log.debug("Initial OHLC fetch failed: %s", e)

        # Risk + equity cache
        self.risk = RiskState(trading_day=self._today_ist())
        self._loss_cooldown = LossCooldownManager(settings.risk)
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        loss_cap = getattr(settings.risk, "max_daily_loss_rupees", None)
        if loss_cap is not None:
            self._max_daily_loss_rupees = float(loss_cap)
        else:
            self._max_daily_loss_rupees = self._equity_cached_value * float(
                settings.risk.max_daily_drawdown_pct
            )

        # State + debug
        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_spot: Optional[float] = None
        self._log_signal_changes_only = (
            os.getenv("LOG_SIGNAL_CHANGES_ONLY", "true").lower() != "false"
        )
        self._last_reason_block: Optional[str] = None
        self._last_has_signal: Optional[bool] = None
        self.eval_count: int = 0
        self.last_eval_ts: Optional[str] = None
        self.trace_ticks_remaining: int = 0
        self.hb_enabled: bool = True
        self._gate = _LogGate()
        self._prev_score_bucket: str | None = None
        self._last_regime: Optional[str] = None
        self._last_atr_state: Optional[str] = None
        self._last_confidence_zone: Optional[str] = None

        # Runtime flags
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0
        # ensure off-hours notification is not spammed
        self._offhours_notified: bool = False
        # track last notification to avoid spamming identical messages
        self._last_notification: Tuple[str, float] = ("", 0.0)
        self._last_hb_ts: float = 0.0

        self._last_diag_emit_ts: float = 0.0
        self._last_signal_hash: tuple | None = None

        self.within_window: bool = False

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading,
            settings.risk.use_live_equity,
        )
        # Log initial equity snapshot
        self._refresh_equity_if_due(silent=False)

    # Optional start hook (main calls it if present)
    def start(self) -> None:
        if self.data_source is None:
            return
        try:
            df = self._fetch_spot_ohlc()
            bars = int(len(df)) if isinstance(df, pd.DataFrame) else 0
            self._last_flow_debug["bars"] = bars
        except Exception as e:
            self.log.warning("Initial data fetch failed: %s", e)
            self._last_flow_debug["bars"] = 0

        # Make sure we have enough bars before first eval
        need = 0
        try:
            need = required_bars(self.strategy_cfg)
        except Exception:
            need = getattr(self.strategy_cfg, "min_bars", 15)
        if hasattr(self.data_source, "ensure_history"):
            lookback_min = max(
                getattr(self.strategy_cfg, "lookback_minutes", 15), int(need)
            )
            try:
                self.data_source.ensure_history(minutes=lookback_min)
            except Exception as e:
                self.log.warning(f"ensure_history({lookback_min}m) skipped: {e}")

    def _emit_diag(self, plan: dict, micro: dict | None = None):
        msg = (
            f"diag | within_window={getattr(self, 'within_window', None)} "
            f"regime={plan.get('regime')} score={plan.get('score')} atr%={plan.get('atr_pct')} "
            f"rr={plan.get('rr')} opt={plan.get('option_type')} strike={plan.get('strike')} "
            f"atm_strike={plan.get('atm_strike')} token={plan.get('option_token')} "
            f"reason_block={plan.get('reason_block')} "
            f"reasons={','.join(plan.get('reasons', []))}"
        )
        if micro:
            msg += (
                " micro={spread%:%s cap%%:%s depth_ok:%s req:%s avail:%s side:%s}" % (
                    micro.get("spread_pct"),
                    micro.get("cap_pct"),
                    micro.get("depth_ok"),
                    micro.get("required_qty"),
                    micro.get("depth_available"),
                    micro.get("side"),
                )
            )
        self.log.info(msg)
        if getattr(self.settings, "TELEGRAM__PRETRADE_ALERTS", False):
            try:
                self.telegram.send_message(msg)
            except Exception:
                pass

    def _maybe_emit_minute_diag(self, plan: dict):
        import time

        if not getattr(self.settings, "ENABLE_SIGNAL_DEBUG", False):
            return
        interval = int(getattr(self.settings, "DIAG_INTERVAL_SECONDS", 60))
        now = time.time()
        if now - self._last_diag_emit_ts >= interval:
            self._last_diag_emit_ts = now
            self._emit_diag(plan)

    def _shadow_blockers(self, plan: Dict[str, Any]) -> list[str]:
        """Non-fatal conditions that would block if score passed."""

        shadows: list[str] = []
        m = plan.get("micro") or {}
        if m.get("mode") == "SOFT":
            sp = m.get("spread_pct")
            cap = m.get("cap_pct")
            if sp is not None and cap is not None and sp > cap:
                shadows.append(f"micro_spread {sp:.2f}%>{cap:.2f}%")
            if m.get("depth_ok") is False:
                shadows.append("micro_depth")
        rr_thresh = getattr(self.strategy_cfg, "rr_threshold", None)
        if rr_thresh is None:
            rr_thresh = (self.strategy_cfg.raw or {}).get("rr_threshold")
        rr_thresh = float(rr_thresh or 0.0)
        if rr_thresh and plan.get("rr", 0.0) < rr_thresh:
            shadows.append(f"rr_low {plan.get('rr')}")
        return shadows

    def _window_active(self, name: str) -> bool:
        checker = getattr(self.telegram, "_window_active", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except Exception:
                return False
        return False

    def _log_score_state(self, score: float) -> None:
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        bucket = self._score_bucket_label(score_f)
        prev = self._prev_score_bucket
        diag_active = self._window_active("diag")
        changed = (prev != bucket) or diag_active
        if changed and self._gate.should_emit("runner:score_bucket"):
            self.log.info(
                "runner.score",
                extra={"score": round(score_f, 3), "bucket": bucket},
            )
        self._prev_score_bucket = bucket

    def _log_regime_state(self, plan: Mapping[str, Any]) -> None:
        regime = plan.get("regime")
        if not regime:
            return
        regime_str = str(regime)
        if self._last_regime == regime_str:
            return
        extras: dict[str, Any] = {"regime": regime_str}
        if self._last_regime is not None:
            extras["prev_regime"] = self._last_regime
        self.log.info("state.regime_change", extra=extras)
        self._last_regime = regime_str

    def _log_atr_band_state(self, plan: Mapping[str, Any]) -> None:
        def _coerce(value: Any) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        atr_raw = _coerce(plan.get("atr_pct_raw"))
        atr_value = atr_raw if atr_raw is not None else _coerce(plan.get("atr_pct"))
        if atr_value is None:
            return

        band = plan.get("atr_band")
        min_val = _coerce(plan.get("atr_min"))
        max_val = _coerce(plan.get("atr_max"))
        if isinstance(band, (list, tuple)) and band:
            if min_val is None and len(band) > 0:
                min_val = _coerce(band[0])
            if max_val is None and len(band) > 1:
                max_val = _coerce(band[1])

        state = "in_band"
        if min_val is not None and atr_value < min_val:
            state = "below_band"

        effective_max = max_val
        if effective_max is not None and effective_max <= 0:
            effective_max = None
        if effective_max is not None and atr_value > effective_max:
            state = "above_band"

        if state == self._last_atr_state:
            return

        extras: dict[str, Any] = {
            "state": state,
            "atr_pct": round(atr_value, 4),
            "min": min_val,
            "max": effective_max,
        }
        if self._last_atr_state is not None:
            extras["prev_state"] = self._last_atr_state
        self.log.info("state.atr_band", extra=extras)
        self._last_atr_state = state

    def _log_confidence_state(self, plan: Mapping[str, Any]) -> None:
        def _coerce(value: Any) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        confidence = _coerce(plan.get("confidence"))
        if confidence is None:
            score_val = _coerce(plan.get("score"))
            confidence = score_val / 10.0 if score_val is not None else None
        if confidence is None:
            return

        strict_raw = _coerce(getattr(self.strategy_cfg, "confidence_threshold", None))
        if strict_raw is None:
            strict_raw = _coerce(getattr(settings.strategy, "confidence_threshold", None))
        relaxed_raw = _coerce(
            getattr(self.strategy_cfg, "confidence_threshold_relaxed", None)
        )
        if relaxed_raw is None:
            relaxed_raw = _coerce(
                getattr(settings.strategy, "confidence_threshold_relaxed", None)
            )

        if strict_raw is None and relaxed_raw is None:
            return

        base_raw = strict_raw if strict_raw is not None else relaxed_raw
        if base_raw is None:
            return

        strict = (strict_raw if strict_raw is not None else base_raw) / 10.0
        relaxed = (
            (relaxed_raw / 10.0)
            if relaxed_raw is not None
            else strict
        )
        if relaxed > strict:
            relaxed = strict

        zone = "below"
        if confidence >= strict:
            zone = "strict"
        elif confidence >= relaxed:
            zone = "relaxed"

        if zone == self._last_confidence_zone:
            return

        extras: dict[str, Any] = {
            "zone": zone,
            "confidence": round(confidence, 3),
            "strict_threshold": round(strict, 3),
            "relaxed_threshold": round(relaxed, 3),
        }
        if self._last_confidence_zone is not None:
            extras["prev_zone"] = self._last_confidence_zone
        self.log.info("state.confidence_zone", extra=extras)
        self._last_confidence_zone = zone

    @staticmethod
    def _score_bucket_label(score: float) -> str:
        if not math.isfinite(score):
            return "nan"
        step = 0.5
        lower = max(0.0, min(score, 15.0))
        start = math.floor(lower / step) * step
        end = min(15.0, start + step)
        if math.isclose(start, end):
            return f"{start:.1f}"
        return f"{start:.1f}-{end:.1f}"

    def emit_heartbeat(self) -> None:
        """Emit a compact heartbeat log with current signal context."""
        if not self.hb_enabled:
            return
        snap = self.telemetry_snapshot()
        sig = snap.get("signal", {})
        micro = sig.get("micro") or {}
        self.log.info(
            "HB eval=%s regime=%s atr%%=%s score=%s spread%%=%s depth=%s block=%s",
            snap.get("eval_count"),
            sig.get("regime"),
            sig.get("atr_pct"),
            sig.get("score"),
            micro.get("spread_pct"),
            micro.get("depth_ok"),
            sig.get("reason_block"),
        )

    def _maybe_hot_reload_cfg(self) -> None:
        """Reload strategy configuration if the underlying file changed."""
        try:
            mtime = os.path.getmtime(self.strategy_config_path)
            if mtime != self.strategy_cfg.mtime:
                new_cfg = try_load(self.strategy_config_path, self.strategy_cfg)
                self.strategy_cfg = new_cfg
                self.tz = ZoneInfo(self.strategy_cfg.tz)
                self.log.info(
                    "CFG reload: %s v%s @ %s",
                    new_cfg.name,
                    new_cfg.version,
                    self.strategy_config_path,
                )
        except Exception as e:
            self.log.warning("CFG reload failed: %s", e)

    def _maybe_reload_events(self) -> None:
        """Hot-reload event calendar if the YAML file changes."""
        if not self.event_cal or not os.path.exists(self.events_path):
            return
        try:
            m = os.path.getmtime(self.events_path)
            if m != self._event_last_mtime:
                self.event_cal = load_calendar(self.events_path)
                self._event_last_mtime = m
                self.log.info(
                    "EVENTS reload: v%s, %s windows",
                    self.event_cal.version,
                    len(self.event_cal.events),
                )
        except Exception as e:
            self.log.warning("EVENTS reload failed: %s", e)

    def _preview_candidate(self, plan: dict, micro: dict | None):
        min_preview = float(getattr(self.settings, "MIN_PREVIEW_SCORE", 8))
        score = float(plan.get("score") or 0.0)
        rb = plan.get("reason_block") or ""
        hard_block = rb in {
            "outside_window",
            "warmup",
            "cooloff",
            "daily_dd",
            "regime_no_trade",
        }
        if hard_block or score < min_preview:
            return
        sig = (
            plan.get("regime"),
            plan.get("option_type"),
            plan.get("strike"),
            round(score, 1),
            round(float(plan.get("rr") or 0.0), 2),
        )
        if sig == self._last_signal_hash:
            return
        self._last_signal_hash = sig
        text = (
            f"\U0001f7e1 Candidate | {plan.get('regime')} {plan.get('option_type')} {plan.get('strike')} "
            f"atm_strike={plan.get('atm_strike')} token={plan.get('option_token')} "
            f"score={score:.1f} rr={plan.get('rr')} entry\u2248{plan.get('entry')} "
            f"sl={plan.get('sl')} tp1={plan.get('tp1')} tp2={plan.get('tp2')} "
            f"opt_sl={plan.get('opt_sl')} opt_tp1={plan.get('opt_tp1')} opt_tp2={plan.get('opt_tp2')} "
            f"reason_block={rb}"
        )
        self.log.info(text)
        if getattr(self.settings, "TELEGRAM__PRETRADE_ALERTS", False):
            try:
                self.telegram.send_message(text)
            except Exception:
                pass

    def _prime_atm_quotes(self) -> tuple[bool, str | None, list[int]]:
        """Ensure CE/PE quotes are primed before scoring and micro checks."""

        if not bool(getattr(self.settings, "enable_live_trading", False)):
            return True, None, []

        ds = getattr(self, "data_source", None)
        broker: Any | None = None
        ce_token: int | None = None
        pe_token: int | None = None

        if ds is not None:
            broker = getattr(ds, "kite", None) or getattr(ds, "broker", None)
            tokens_raw = getattr(ds, "atm_tokens", None)
            ensure_tokens = getattr(ds, "ensure_atm_tokens", None)
            if (not tokens_raw or None in tokens_raw) and callable(ensure_tokens):
                try:
                    ensure_tokens()
                except Exception:
                    logger.debug("quote_prime_ensure_retry", exc_info=True)
                tokens_raw = getattr(ds, "atm_tokens", None)
            if isinstance(tokens_raw, (list, tuple)):
                if len(tokens_raw) > 0 and tokens_raw[0]:
                    ce_token = int(tokens_raw[0])
                if len(tokens_raw) > 1 and tokens_raw[1]:
                    pe_token = int(tokens_raw[1])

        if broker is None:
            broker = self.kite

        needed_tokens: list[int] = [int(t) for t in (ce_token, pe_token) if t]
        if not needed_tokens:
            logger.debug("quote_prime_fail", {"err": "tokens_missing"})
            return False, "tokens_missing", []
        if broker is None:
            logger.debug("quote_prime_fail", {"err": "broker_missing"})
            return False, "broker_missing", needed_tokens

        quotes: Any | None = None
        errors: list[str] = []
        payloads = (needed_tokens, [str(t) for t in needed_tokens])
        for payload in payloads:
            if not payload:
                continue
            if hasattr(broker, "quote"):
                try:
                    quotes = broker.quote(payload)
                except Exception as exc:  # pragma: no cover - network failures
                    errors.append(str(exc))
                    quotes = None
                else:
                    if quotes:
                        break
            if hasattr(broker, "ltp"):
                try:
                    quotes = broker.ltp(payload)
                except Exception as exc:  # pragma: no cover - network failures
                    errors.append(str(exc))
                    quotes = None
                else:
                    if quotes:
                        break

        if not quotes:
            err_msg = ";".join(err for err in errors if err) or "quote_unavailable"
            logger.debug("quote_prime_fail", {"err": err_msg})
            return False, err_msg, needed_tokens

        missing_tokens = self._detect_missing_tokens(needed_tokens, quotes)
        if missing_tokens:
            err_msg = f"missing:{','.join(str(tok) for tok in missing_tokens)}"
            logger.debug("quote_prime_fail", {"err": err_msg})
            return False, err_msg, needed_tokens

        logger.debug("quote_prime_ok", {"n": len(needed_tokens)})
        return True, None, needed_tokens

    def _get_cached_full_quote(
        self, token: int | str | None
    ) -> dict[str, Any] | None:
        """Return a detached copy of the cached FULL quote for ``token``."""

        if token in (None, ""):
            return None

        ds = getattr(self, "data_source", None)
        if ds is None:
            return None

        getter = getattr(ds, "get_cached_full_quote", None)
        if callable(getter):
            try:
                quote = getter(token)
            except Exception:
                self.log.debug("get_cached_full_quote failed", exc_info=True)
            else:
                if isinstance(quote, Mapping):
                    return dict(quote)

        cache = getattr(ds, "_option_quote_cache", None)
        if isinstance(cache, Mapping):
            try:
                token_i = int(token)  # type: ignore[arg-type]
            except Exception:
                return None
            raw = cache.get(token_i)
            if isinstance(raw, Mapping):
                return dict(raw)
        return None

    def _detect_missing_tokens(self, tokens: list[int], quote_payload: Any) -> list[int]:
        """Identify which instrument tokens are absent from a quote payload."""

        missing: list[int] = []
        for token in tokens:
            if not self._quote_has_token(quote_payload, token):
                missing.append(int(token))
        return missing

    @staticmethod
    def _format_two_decimals(value: Any) -> Any:
        """Format numeric values with two decimal places when possible."""

        if value is None or isinstance(value, bool):
            return value

        if isinstance(value, Decimal):
            return f"{float(value):.2f}"

        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"

        if isinstance(value, str):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return value
            return f"{number:.2f}"

        return value

    def _quote_has_token(self, payload: Any, token: int) -> bool:
        """Recursively inspect a quote payload for a specific instrument token."""

        if payload is None:
            return False

        try:
            token_int = int(token)
        except Exception:
            token_int = token

        if isinstance(payload, Mapping):
            if token_int in payload or str(token_int) in payload:
                return True
            data_section = payload.get("data")
            if data_section is not None and self._quote_has_token(data_section, token_int):
                return True
            for key, value in payload.items():
                if key in {"instrument_token", "token", "instrument", "token_id"}:
                    try:
                        if value is not None and int(value) == token_int:
                            return True
                    except Exception:
                        continue
                if isinstance(value, (Mapping, list, tuple, set)):
                    if self._quote_has_token(value, token_int):
                        return True
                elif isinstance(value, str):
                    digits = "".join(ch for ch in value if ch.isdigit())
                    if digits and digits == str(token_int):
                        return True
            return False

        if isinstance(payload, (list, tuple, set)):
            return any(self._quote_has_token(item, token_int) for item in payload)

        if isinstance(payload, str):
            if payload == str(token_int):
                return True
            digits = "".join(ch for ch in payload if ch.isdigit())
            return bool(digits and digits == str(token_int))

        try:
            return int(payload) == token_int
        except Exception:
            return False

    @staticmethod
    def _sync_atm_state(
        ds: Any,
        *,
        option_type: str,
        token: Any,
        strike: Any,
        expiry: Any,
    ) -> None:
        """Update data source ATM metadata with the latest resolved option."""

        if ds is None:
            return

        if strike is not None:
            try:
                setattr(ds, "current_atm_strike", strike)
            except Exception:
                pass

        if expiry is not None:
            try:
                setattr(ds, "current_atm_expiry", expiry)
            except Exception:
                pass

        option_key = str(option_type or "").upper()
        idx_map = {"CE": 0, "PE": 1}
        idx = idx_map.get(option_key)
        if idx is None:
            return

        try:
            coerced_token = int(token)
        except Exception:
            coerced_token = None
        if coerced_token in (None, 0):
            return

        existing = getattr(ds, "atm_tokens", None)
        if isinstance(existing, (list, tuple)):
            tokens: list[int | None] = []
            for raw in list(existing)[:2]:
                try:
                    coerced = int(raw)
                except Exception:
                    coerced = None
                if coerced in (None, 0):
                    tokens.append(None)
                else:
                    tokens.append(coerced)
        else:
            tokens = [None, None]

        while len(tokens) < 2:
            tokens.append(None)

        if tokens[idx] == coerced_token:
            return

        tokens[idx] = coerced_token
        try:
            setattr(ds, "atm_tokens", tuple(tokens))
        except Exception:
            # Fallback for data sources storing tokens as mutable sequences
            try:
                current = list(getattr(ds, "atm_tokens", []))
                while len(current) < 2:
                    current.append(None)
                current[idx] = coerced_token
                setattr(ds, "atm_tokens", tuple(current))
            except Exception:
                pass

    def _record_plan(self, plan: Dict[str, Any]) -> None:
        micro = plan.get("micro") or {"spread_pct": 0.0, "depth_ok": False}
        changed = (
            plan.get("has_signal") != self._last_has_signal
            or plan.get("reason_block") != self._last_reason_block
        )
        if (not self._log_signal_changes_only) or changed:
            if changed or self._gate.should_emit("runner:plan_debug", interval=5.0):
                strike = self._format_two_decimals(plan.get("strike"))
                atm_strike = self._format_two_decimals(plan.get("atm_strike"))
                score = self._format_two_decimals(plan.get("score"))
                atr_pct = self._format_two_decimals(plan.get("atr_pct") or 0.0)
                spread_pct = self._format_two_decimals(micro.get("spread_pct") or 0.0)
                rr = self._format_two_decimals(plan.get("rr") or 0.0)
                sl = self._format_two_decimals(plan.get("sl"))
                tp1 = self._format_two_decimals(plan.get("tp1"))
                tp2 = self._format_two_decimals(plan.get("tp2"))
                opt_sl = self._format_two_decimals(plan.get("opt_sl"))
                opt_tp1 = self._format_two_decimals(plan.get("opt_tp1"))
                opt_tp2 = self._format_two_decimals(plan.get("opt_tp2"))
                self.log.debug(
                    "Signal plan: action=%s %s strike=%s atm_strike=%s token=%s qty=%s regime=%s score=%s atr%%=%s spread%%=%s depth=%s rr=%s sl=%s tp1=%s tp2=%s opt_sl=%s opt_tp1=%s opt_tp2=%s reason_block=%s",
                    plan.get("action"),
                    plan.get("option_type"),
                    strike,
                    atm_strike,
                    plan.get("option_token"),
                    plan.get("qty_lots"),
                    plan.get("regime"),
                    score,
                    atr_pct,
                    spread_pct,
                    micro.get("depth_ok"),
                    rr,
                    sl,
                    tp1,
                    tp2,
                    opt_sl,
                    opt_tp1,
                    opt_tp2,
                    plan.get("reason_block"),
                )
        plan["eval_count"] = self.eval_count
        plan["last_eval_ts"] = self.last_eval_ts
        self._last_reason_block = plan.get("reason_block")
        self._last_has_signal = plan.get("has_signal")
        pw = getattr(self, "plan_probe_window", None)
        plan["probe_window_from"] = pw[0] if pw else None
        plan["probe_window_to"] = pw[1] if pw else None
        plan["shadow_blockers"] = self._shadow_blockers(plan)
        self.last_plan = dict(plan)
        now = time.time()
        if self.hb_enabled and (now - self._last_hb_ts) >= 15 * 60:
            self.emit_heartbeat()
            self._last_hb_ts = now

    def _log_decisive_event(
        self,
        *,
        label: str | None = None,
        signal: dict | None = None,
        reason_block: str | None = None,
        stage: str | None = None,
        decision: str | None = None,
        reason_codes: list[str] | None = None,
        plan: dict | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one structured 'decision' record backed by the latest plan snapshot."""

        try:
            record_input: Any = signal if signal is not None else plan
            if hasattr(self, "_record_plan"):
                try:
                    self._record_plan(record_input)  # safe even if None
                except Exception:
                    pass

            snapshot_obj: Any = getattr(self, "_last_plan", None)
            if snapshot_obj is None:
                if isinstance(plan, Mapping):
                    snapshot_obj = plan
                elif isinstance(signal, Mapping):
                    snapshot_obj = signal

            snapshot: dict[str, Any] = {}
            if isinstance(snapshot_obj, Mapping):
                snapshot = dict(snapshot_obj)

            now = time.time()
            last = getattr(self, "_ts_last_decision_emit", 0.0)
            if now - last < 0.05:
                return
            self._ts_last_decision_emit = now

            def _sanitize(value: Any) -> Any:
                if isinstance(value, Decimal):
                    return float(value)
                if isinstance(value, Mapping):
                    return {k: _sanitize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple, set)):
                    return [_sanitize(v) for v in value]
                if hasattr(value, "isoformat"):
                    try:
                        return value.isoformat()  # type: ignore[attr-defined]
                    except Exception:
                        return str(value)
                try:
                    return float(value) if isinstance(value, Decimal) else value
                except Exception:
                    return value

            reason_block_value = reason_block or snapshot.get("reason_block")
            if reason_codes is not None:
                reason_code_list = list(reason_codes)
            elif reason_block_value:
                reason_code_list = [str(reason_block_value)]
            else:
                reason_code_list = []

            payload: dict[str, Any] = {
                "stage": stage or "process",
                "decision": decision or label or "idle",
                "reason_codes": reason_code_list,
                "reason_block": reason_block_value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if label is not None:
                payload["label"] = label
            if metrics:
                payload["stage_metrics"] = _sanitize(metrics)

            for key, value in snapshot.items():
                if key not in payload:
                    payload[key] = _sanitize(value)

            micro_snapshot = snapshot.get("micro")
            if isinstance(micro_snapshot, Mapping):
                if "spread_pct" not in payload and "spread_pct" in micro_snapshot:
                    payload["spread_pct"] = _sanitize(micro_snapshot.get("spread_pct"))
                if "depth_ok" not in payload and "depth_ok" in micro_snapshot:
                    payload["depth_ok"] = _sanitize(micro_snapshot.get("depth_ok"))

            message = json.dumps(payload, default=lambda obj: str(obj))
            self.log.info("DECISION %s", message)
        except Exception:
            try:
                self.log.debug("decision.emit_error", exc_info=True)
            except Exception:
                pass

    # ---------------- main loop entry ----------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        self.eval_count += 1
        if (self.eval_count % 30) == 0:
            self._maybe_hot_reload_cfg()
            self._maybe_reload_events()
        if (self.eval_count % 60) == 0:
            snap = {
                "now": self.now_ist.isoformat(),
                "risk": (
                    self.risk_engine.snapshot() if hasattr(self, "risk_engine") else {}
                ),
                "open_legs": getattr(
                    self.order_executor, "open_legs_snapshot", lambda: []
                )(),
                "components": self._active_names,
                "last_plan": self.last_plan or {},
            }
            self.journal.save_checkpoint(snap)
        self.last_eval_ts = datetime.now(timezone.utc).isoformat()
        flow: Dict[str, Any] = {
            "within_window": False,
            "paused": self._paused,
            "data_ok": False,
            "bars": 0,
            "signal_ok": False,
            "rr_ok": True,
            "risk_gates": {},
            "sizing": {},
            "qty": 0,
            "executed": False,
            "reason_block": None,
        }
        self._last_error = None
        warm_min = int(
            getattr(settings, "WARMUP_MIN_BARS", getattr(settings, "warmup_min_bars", 0))
        )
        if (
            self.data_source
            and warm_min > 0
            and hasattr(self.data_source, "have_min_bars")
            and not self.data_source.have_min_bars(warm_min)
        ):
            ok = False
            ensure = getattr(self.data_source, "ensure_warmup", None)
            if callable(ensure):
                ok = bool(ensure(warm_min))
            if not ok:
                self.ready = False
                flow["reason_block"] = "warmup"
                self._last_flow_debug = flow
                now_t = time.time()
                if now_t - getattr(self, "_warmup_log_ts", 0.0) > 30:
                    self.log.info(
                        "Waiting for warmup: %s bars required", warm_min
                    )
                    self._warmup_log_ts = now_t
                self._log_decisive_event(
                    label="blocked",
                    signal=None,
                    reason_block=flow.get("reason_block"),
                )
                return
        now = datetime.now(timezone.utc)
        tick_now = datetime.now(TZ)
        for cb in [
            getattr(self.order_executor, "cb_orders", None),
            getattr(self.order_executor, "cb_modify", None),
            getattr(self.data_source, "cb_hist", None),
            getattr(self.data_source, "cb_quote", None),
        ]:
            if cb:
                cb.tick(tick_now)
        try:
            if hasattr(self.order_executor, "step_queue"):
                self.order_executor.step_queue(now)
                if hasattr(self.order_executor, "on_order_timeout_check"):
                    self.order_executor.on_order_timeout_check()
            if getattr(self, "reconciler", None):
                self.reconciler.step(now)
        except Exception:
            self.log.debug("queue/reconcile step failed", exc_info=True)
        if bool(getattr(settings, "enable_live_trading", False)):
            now_ist = self._now_ist()
            t = now_ist.time()
            if t >= self._flatten_time:
                if getattr(self.order_executor, "open_count", 0) > 0:
                    self.log.info(
                        "eod_close (RISK__EOD_FLATTEN_HHMM=%s)",
                        self.risk_engine.cfg.eod_flatten_hhmm,
                    )
                    getattr(self.order_executor, "cancel_all_orders", lambda: None)()
                    getattr(
                        self.order_executor, "close_all_positions_eod", lambda: None
                    )()
                    self._notify(
                        "\U0001F514 EOD flatten — positions closed and orders cancelled"
                    )
                if hasattr(self.telegram, "send_eod_summary"):
                    try:
                        self.telegram.send_eod_summary()
                    except Exception:
                        pass
        try:
            # fetch data first to allow ADX‑based window override
            df = self._fetch_spot_ohlc()
            if not isinstance(df, pd.DataFrame) or df.empty:
                self.log.warning("OHLC fetch returned no data; retrying")
                df = self._fetch_spot_ohlc()
                if not isinstance(df, pd.DataFrame) or df.empty:
                    if (
                        isinstance(getattr(self, "_ohlc_cache", None), pd.DataFrame)
                        and not self._ohlc_cache.empty
                    ):
                        self.log.warning("Using cached OHLC data due to empty fetch")
                        df = self._ohlc_cache
                    else:
                        df = pd.DataFrame()
            flow["bars"] = int(len(df))
            adx_val = None
            try:
                adx_series = df.get("adx")
                if adx_series is None:
                    adx_cols = [c for c in df.columns if c.startswith("adx_")]
                    if adx_cols:
                        adx_series = df[sorted(adx_cols)[-1]]
                if adx_series is not None and len(adx_series):
                    adx_val = float(adx_series.iloc[-1])
            except Exception:
                adx_val = None

            try:
                window_ok = self._within_trading_window(adx_val)
            except TypeError:
                window_ok = self._within_trading_window()
            enable_windows = getattr(settings, "enable_time_windows", True)
            within = (
                (not enable_windows)
                or window_ok
                or bool(settings.allow_offhours_testing)
            )
            # Always process ticks regardless of trading window
            self.within_window = True
            flow["within_window"] = True
            if not within and not self._offhours_notified:
                now = self._now_ist().strftime("%H:%M:%S")
                tz_name = getattr(settings, "tz", "IST")
                self._notify(
                    f"⏰ Tick received outside trading window at {now} {tz_name}"
                )
                self._offhours_notified = True
            elif within:
                self._offhours_notified = False

            # pause
            if self._paused:
                flow["reason_block"] = "paused"
                self._last_flow_debug = flow
                self.log.debug("Skipping tick: runner paused")
                self._log_decisive_event(
                    label="blocked",
                    signal=None,
                    reason_block=flow.get("reason_block"),
                )
                return

            # new day / equity
            self._ensure_day_state()
            self._refresh_equity_if_due()

            # we already fetched df above; validate sufficiency
            self.log.debug("Fetched %s bars", flow["bars"])
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"
                self._last_flow_debug = flow
                self.log.debug(
                    "Signal evaluation skipped: insufficient data (bars=%s, need=%s)",
                    flow["bars"],
                    int(settings.strategy.min_bars_for_signal),
                )
                self._log_decisive_event(
                    label="blocked",
                    signal=None,
                    reason_block=flow.get("reason_block"),
                )
                return
            flow["data_ok"] = True
            # Event guard pre-check to hint strategy
            active_events: List[EventWindow] = []
            event_block = False
            self._event_post_widen = 0.0
            if self.event_guard_enabled and self.event_cal:
                active_events = self.event_cal.active(self.now_ist)
                if active_events:
                    event_block = any(ev.block_trading for ev in active_events)
                    self._event_post_widen = max(
                        (ev.post_widen_spread_pct for ev in active_events),
                        default=0.0,
                    )
            setattr(self.strategy, "_event_post_widen", self._event_post_widen)

            # ---- plan
            plan = self.strategy.generate_signal(df, current_tick=tick)
            atr_period = int(getattr(settings.strategy, "atr_period", 14))
            last_bar_ts_obj = (
                df.index[-1].to_pydatetime()
                if isinstance(df.index, pd.DatetimeIndex) and len(df)
                else None
            )
            plan["reasons"] = plan.get("reasons", [])
            plan["feature_ok"] = True
            plan["bar_count"] = int(len(df)) if isinstance(df, pd.DataFrame) else 0
            plan["last_bar_ts"] = (
                last_bar_ts_obj.isoformat() if last_bar_ts_obj else None
            )
            bar_count = int(plan["bar_count"])
            self._last_bar_count = int(bar_count)
            self._warm = warmup_check(self.strategy_cfg, bar_count)
            self.ready = self._warm.ok
            if not self._warm.ok:
                plan["reason_block"] = "insufficient_bars"
                plan.setdefault("reasons", []).extend(self._warm.reasons)
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            now = self._now_ist()
            self._fresh = compute_freshness(
                now=now,
                last_tick_ts=None,
                last_bar_open_ts=last_bar_ts_obj,
                tf_seconds=60,
                max_tick_lag_s=int(getattr(self.strategy_cfg, "max_tick_lag_s", 8)),
                max_bar_lag_s=int(getattr(self.strategy_cfg, "max_bar_lag_s", 75)),
            )
            plan["tick_lag_s"] = self._fresh.tick_lag_s
            plan["bar_lag_s"] = self._fresh.bar_lag_s
            plan["lag_s"] = self._fresh.bar_lag_s
            if not self._fresh.ok:
                plan["reason_block"] = "data_stale"
                plan.setdefault("reasons", []).append(
                    f"tick_lag={self._fresh.tick_lag_s}"
                )
                plan.setdefault("reasons", []).append(
                    f"bar_lag={self._fresh.bar_lag_s}"
                )
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            atr_val = atr_pct(df, period=atr_period)
            plan["atr_pct_raw"] = float(atr_val) if atr_val is not None else None
            plan["atr_pct"] = (
                round(plan["atr_pct_raw"], 2) if plan["atr_pct_raw"] is not None else None
            )
            if plan["atr_pct_raw"] is None:
                plan["reasons"].append("atr_na")
                plan["reason_block"] = "features"
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            band = resolve_atr_band(self.strategy_cfg, self.under_symbol)
            ok, reason, atr_min, atr_max = check_atr(
                plan["atr_pct_raw"], self.strategy_cfg, self.under_symbol, band=band
            )
            plan["atr_min"] = atr_min
            plan["atr_max"] = atr_max
            plan["atr_band"] = (atr_min, atr_max)
            self._log_regime_state(plan)
            self._log_atr_band_state(plan)
            if not ok:
                if reason and reason not in plan["reasons"]:
                    plan["reasons"].append(reason)
                plan["reason_block"] = "atr_out_of_band"
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                detail = flow.setdefault("reason_details", {})
                detail["atr_out_of_band"] = {
                    "reason": reason,
                    "atr_pct": plan["atr_pct_raw"],
                    "atr_pct_display": plan["atr_pct"],
                    "band": (atr_min, atr_max),
                }
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            prime_ok, prime_err, prime_tokens = self._prime_atm_quotes()
            if not prime_ok:
                reasons = plan.setdefault("reasons", [])
                if "no_quote" not in reasons:
                    reasons.append("no_quote")
                if prime_err:
                    detail = f"quote_prime:{prime_err}"
                    if detail not in reasons:
                        reasons.append(detail)
                plan["reason_block"] = "no_quote"
                plan["has_signal"] = False
                plan["spread_pct"] = None
                plan["depth_ok"] = None
                plan["micro"] = {"spread_pct": None, "depth_ok": None}
                plan["quote_prime_tokens"] = prime_tokens
                if prime_err:
                    plan["quote_prime_error"] = prime_err
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                flow.setdefault("reason_details", {})["no_quote"] = {
                    "tokens": prime_tokens,
                    "error": prime_err,
                }
                self._last_flow_debug = flow
                self.log.info(
                    "Signal blocked: no_quote tokens=%s err=%s",
                    prime_tokens,
                    prime_err,
                )
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            score_val, details = compute_score(
                df, plan.get("regime"), self.strategy_cfg
            )
            plan["score"] = float(score_val or 0.0)
            if details:
                self._score_items = details.parts
                self._score_total = details.total
            else:
                self._score_items = None
                self._score_total = None
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            score_val = float(plan.get("score") or 0.0)
            self._log_score_state(score_val)
            if score_val == 0.0:
                bar_count = int(plan.get("bar_count") or 0)
                regime = plan.get("regime")
                min_bars = int(getattr(settings.strategy, "min_bars_for_signal"))
                if bar_count < min_bars or regime == "NO_TRADE":
                    self.log.debug(
                        "Scoring skipped: bar_count=%s regime=%s",
                        bar_count,
                        regime,
                    )
                else:
                    self.log.warning(
                        "Missing score with bar_count=%s regime=%s",
                        bar_count,
                        regime,
                    )
            min_score = self._min_score_threshold()
            if score_val < min_score and not plan.get("reason_block"):
                plan["reason_block"] = "score_low"
                plan.setdefault("reasons", []).extend(
                    [f"score={score_val:.2f}", f"min={min_score:.2f}"]
                )
            self._maybe_emit_minute_diag(plan)
            if plan.get("reason_block"):
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self.log.debug("No tradable plan: %s", flow["reason_block"])
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            flow["signal_ok"] = True
            metrics.inc_signal()
            flow["plan"] = dict(plan)

            if self.event_guard_enabled and self.event_cal and active_events:
                block = event_block
                widen = self._event_post_widen
                plan.setdefault("reasons", []).append(
                    f"event_guard:{','.join(ev.name for ev in active_events)}"
                )
                plan["event_guard"] = {
                    "active": True,
                    "names": [ev.name for ev in active_events],
                    "post_widen_spread_pct": round(widen, 3),
                    "block": block,
                }
                if block:
                    plan["has_signal"] = False
                    plan["reason_block"] = "event_guard"
                    self.last_plan = plan
                    self._last_flow_debug = flow
                    self._record_plan(plan)
                    self._log_decisive_event(
                        label="blocked",
                        signal=plan,
                        reason_block=flow.get("reason_block"),
                    )
                    return
                else:
                    plan["_event_post_widen"] = float(widen)

            # ---- RR minimum
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(plan.get("rr", 0.0) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                plan["reason_block"] = f"rr<{rr_min}"
                flow["rr_ok"] = False
                flow["reason_block"] = plan["reason_block"]
                flow["plan"] = {**plan, "rr_min": rr_min}
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.info(
                    "Signal skipped: rr %.2f below minimum %.2f", rr_val, rr_min
                )
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            cfg = self.strategy_cfg
            score = plan.get("score") or 0
            if cfg and score >= int(getattr(cfg, "delta_enable_score", 999)):
                try:
                    pick = strike_selector.select_strike_by_delta(
                        spot=plan.get("spot"),
                        opt=plan.get("option"),
                        expiry=plan.get("expiry"),
                        target=float(getattr(cfg, "delta_target", 0.40)),
                        band=float(getattr(cfg, "delta_band", 0.05)),
                        chain=plan.get("option_chain", []),
                    )
                    if pick:
                        plan["strike"] = pick["strike"]
                        plan.setdefault("reasons", []).append(
                            f"delta_pick:{getattr(cfg, 'delta_target', 0.40)}"
                        )
                except Exception:
                    pass

            # store some diagnostics from signal
            flow.update(
                {
                    "regime": plan.get("regime"),
                    "score": plan.get("score"),
                    "rr": plan.get("rr"),
                    "sl": plan.get("stop_loss"),
                    "tp1": plan.get("tp1"),
                    "tp2": plan.get("tp2"),
                }
            )
            under_ltp = 0.0
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    under_ltp = float(df["close"].iloc[-1])
                except Exception:
                    under_ltp = 0.0
            try:
                opt = self.option_resolver.resolve_atm(
                    self.under_symbol,
                    under_ltp,
                    plan.get("side_hint", "CE"),
                    self._now_ist(),
                )
            except Exception:
                plan["reason_block"] = "no_option_token"
                plan.setdefault("reasons", []).append("no_option_token")
                plan["option"] = {}
                plan["expiry"] = None
                plan["option_token"] = None
                plan["token"] = None
                plan["spread_pct"] = None
                plan["depth_ok"] = None
                plan["micro"] = {"spread_pct": None, "depth_ok": None}
                ds = getattr(self, "data_source", None)
                plan["atm_strike"] = getattr(ds, "current_atm_strike", None)
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self._last_option = None
                self.log.exception("Failed to resolve ATM option")
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            token = opt.get("token")
            plan["option"] = opt
            plan["expiry"] = opt.get("expiry")
            plan["option_token"] = token
            plan["token"] = token
            ds = getattr(self, "data_source", None)
            atm_strike = getattr(ds, "current_atm_strike", None)
            if atm_strike is None:
                atm_strike = opt.get("strike")
            atm_expiry = getattr(ds, "current_atm_expiry", None)
            if atm_expiry is None:
                atm_expiry = opt.get("expiry")
            plan["atm_strike"] = atm_strike
            plan["atm_expiry"] = atm_expiry

            def _coerce_token(value: Any) -> int | None:
                try:
                    coerced = int(value)
                except (TypeError, ValueError):
                    return None
                return coerced or None

            ce_token: int | None = None
            pe_token: int | None = None
            if ds is not None:
                tokens_raw = getattr(ds, "atm_tokens", None)
                if isinstance(tokens_raw, (list, tuple)):
                    if len(tokens_raw) > 0:
                        ce_token = _coerce_token(tokens_raw[0])
                    if len(tokens_raw) > 1:
                        pe_token = _coerce_token(tokens_raw[1])

            option_type = str(plan.get("option_type") or plan.get("side_hint") or "").upper()
            token_map: dict[str, int | None] = {"CE": ce_token, "PE": pe_token}
            plan["token"] = token_map.get(option_type)
            if plan["token"] is None and token is not None:
                try:
                    plan["token"] = int(token)
                except (TypeError, ValueError):
                    plan["token"] = None

            if ds is not None:
                self._sync_atm_state(
                    ds,
                    option_type=option_type,
                    token=plan.get("token"),
                    strike=plan.get("atm_strike"),
                    expiry=plan.get("atm_expiry"),
                )

            logger.debug(
                "picked option: type=%s ce=%s pe=%s chosen=%s",
                option_type,
                ce_token,
                pe_token,
                plan["token"],
            )
            if (
                option_type == "PE"
                and pe_token is not None
                and plan.get("token") != pe_token
            ) or (
                option_type == "CE"
                and ce_token is not None
                and plan.get("token") != ce_token
            ):
                plan["reason_block"] = "token_mismatch"
                plan.setdefault("reasons", []).append("token_mismatch")
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            self._last_option = opt
            if not token:
                plan["reason_block"] = "no_option_token"
                plan.setdefault("reasons", []).append("no_option_token")
                plan["token"] = None
                self.log.warning(
                    "No option token: expiry=%s strike=%s kind=%s",
                    opt.get("expiry"),
                    opt.get("strike"),
                    plan.get("side_hint"),
                )
                plan["spread_pct"] = None
                plan["depth_ok"] = None
                plan["micro"] = {"spread_pct": None, "depth_ok": None}
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            ensure_subscribe = getattr(ds, "ensure_token_subscribed", None)
            if callable(ensure_subscribe):
                try:
                    ensure_subscribe(token)
                except Exception:
                    self.log.debug("ensure_token_subscribed failed", exc_info=True)
            prime_price: float | None = None
            prime_src: str | None = None
            prime_ts: int | None = None
            prime_fn = getattr(ds, "prime_option_quote", None)
            require_prime = bool(getattr(self.settings, "enable_live_trading", False))
            token_for_quote = plan.get("token")
            if callable(prime_fn) and token_for_quote:
                try:
                    prime_price, prime_src, prime_ts = prime_fn(token_for_quote)
                except Exception:
                    self.log.debug("prime_option_quote failed", exc_info=True)
                    prime_price = None
                    prime_src = None
                    prime_ts = None
            plan["prime_quote_price"] = prime_price
            plan["prime_quote_src"] = prime_src
            plan["prime_quote_ts"] = prime_ts
            if require_prime and token_for_quote and not prime_price:
                plan["reason_block"] = "no_quote"
                plan.setdefault("reasons", []).append("no_quote")
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.log.info("no_quote token=%s", token_for_quote)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            raw_quote: dict[str, Any] | None = None
            if self.kite is not None and opt.get("tradingsymbol"):
                try:
                    ts = opt["tradingsymbol"]
                    resp = self.kite.quote([f"NFO:{ts}"])  # type: ignore[attr-defined]
                    raw = resp.get(f"NFO:{ts}") if isinstance(resp, dict) else None
                    if isinstance(raw, dict):
                        raw_quote = dict(raw)
                        raw_quote.setdefault("source", "kite")
                except Exception:
                    raw_quote = None

            fetch_ltp = None
            if hasattr(self.data_source, "get_last_price"):
                fetch_ltp = getattr(self.data_source, "get_last_price")

            quote_dict, quote_mode = get_option_quote_safe(
                option=opt,
                quote=raw_quote,
                fetch_ltp=fetch_ltp,
            )
            if not quote_dict:
                plan["reason_block"] = "no_option_quote"
                plan.setdefault("reasons", []).append("no_option_quote")
                plan["spread_pct"] = None
                plan["depth_ok"] = None
                plan["micro"] = {"spread_pct": None, "depth_ok": None}
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            def _positive_number(value: Any) -> bool:
                return isinstance(value, (int, float)) and value > 0

            have_price = any(
                _positive_number(quote_dict.get(key))
                for key in ("mid", "ltp", "bid", "ask")
            )
            have_levels = any(
                _positive_number(quote_dict.get(key))
                for key in ("bid", "ask", "bid_qty", "ask_qty", "bid5_qty", "ask5_qty")
            )
            if not have_levels:
                depth_payload = quote_dict.get("depth")
                if isinstance(depth_payload, Mapping):
                    for side in ("buy", "sell"):
                        levels = depth_payload.get(side)
                        if isinstance(levels, list) and levels:
                            first = levels[0]
                            if isinstance(first, Mapping) and (
                                _positive_number(first.get("price"))
                                or _positive_number(first.get("quantity"))
                            ):
                                have_levels = True
                                break
            if not have_price or not have_levels:
                plan["reason_block"] = "no_quote"
                plan.setdefault("reasons", []).append("no_quote")
                plan["spread_pct"] = None
                plan["depth_ok"] = None
                plan["micro"] = {"spread_pct": None, "depth_ok": None}
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.log.info(
                    "no_quote token=%s missing_price=%s missing_levels=%s",
                    token,
                    not have_price,
                    not have_levels,
                )
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            micro = evaluate_micro(
                q=quote_dict,
                lot_size=opt.get("lot_size", self.lot_size),
                atr_pct=plan.get("atr_pct"),
                cfg=self.strategy_cfg.raw,
                side=plan.get("action"),
                lots=plan.get("qty_lots"),
                depth_multiplier=float(
                    getattr(settings.executor, "depth_multiplier", 1.0)
                ),
                require_depth=bool(getattr(settings.executor, "require_depth", False)),
            )
            plan["spread_pct"] = micro.get("spread_pct")
            plan["depth_ok"] = micro.get("depth_ok")
            plan["micro"] = micro
            plan["quote_src"] = quote_dict.get("source", "kite")
            plan["quote_mode"] = quote_mode
            plan["quote"] = quote_dict
            self.log.debug(
                "quote=ok mid=%.2f bid=%.2f ask=%.2f mode=%s src=%s",
                float(quote_dict.get("mid", 0.0)),
                float(quote_dict.get("bid", 0.0)),
                float(quote_dict.get("ask", 0.0)),
                quote_mode,
                plan["quote_src"],
            )
            if micro.get("spread_pct") is None or micro.get("depth_ok") is None:
                plan["reason_block"] = "no_option_quote"
                plan.setdefault("reasons", []).append("no_option_quote")
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.last_plan = plan
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            if micro.get("would_block"):
                plan.setdefault("reasons", []).append("micro")
                plan["reason_block"] = "micro"
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.last_plan = plan
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            if micro.get("mode") == "SOFT":
                penalty = (
                    0.5
                    if (
                        micro.get("spread_pct")
                        and micro.get("spread_pct") > micro.get("cap_pct")
                    )
                    else 0.0
                )
                plan["score"] = max(0.0, (plan.get("score", 0.0) - penalty))

            acct = risk_gates.AccountState(
                equity_rupees=self._active_equity(),
                dd_rupees=self.risk.day_realized_loss,
                max_daily_loss=self._max_daily_loss_rupees,
                loss_streak=self.risk.consecutive_losses,
            )
            ok, gate_reasons = risk_gates.evaluate(plan, acct, self.strategy_cfg)
            plan["risk_ok"] = ok
            plan.setdefault("reasons", []).extend(gate_reasons)
            if not ok:
                if "daily_dd" in gate_reasons:
                    plan["reason_block"] = "daily_dd"
                if not plan.get("reason_block"):
                    plan["reason_block"] = "risk"
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            # ---- risk gates
            gates = self._risk_gates_for(plan)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                blocked = [k for k, v in gates.items() if not v]
                if "daily_drawdown" in blocked:
                    plan["reason_block"] = "daily_dd_hit"
                elif "loss_streak" in blocked:
                    plan["reason_block"] = "loss_cooloff"
                elif "market_hours" in blocked:
                    plan["reason_block"] = "market_closed"
                else:
                    plan["reason_block"] = "risk_gate_block"
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.info("Signal blocked by risk gates: %s", blocked)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            # ---- limits engine
            exposure = Exposure(
                lots_by_symbol=self._lots_by_symbol(),
                notional_rupees=self._notional_rupees(),
            )
            sym = plan.get("symbol") or plan.get("strike") or ""
            qty_lots = int(plan.get("qty_lots") or 1)
            lot_size = int(getattr(self.settings.instruments, "nifty_lot_size", 75))
            entry = plan.get("entry") or plan.get("entry_price") or 0.0
            sl = plan.get("sl") or plan.get("stop_loss") or 0.0

            planned_delta_units: Optional[float] = None
            if (
                plan.get("has_signal")
                and plan.get("strike")
                and plan.get("option_type")
                and plan.get("qty_lots")
            ):
                parsed = strike_selector.parse_nfo_symbol(plan["strike"])
                if parsed:
                    s = getattr(self, "last_spot", 0.0) or 0.0
                    k = parsed["strike"]
                    opt = parsed["option_type"]
                    mid = plan.get("entry") or plan.get("entry_price") or 0.0
                    rfr = float(getattr(self.settings, "RISK_FREE_RATE", 0.065))
                    est = estimate_greeks_from_mid(
                        s,
                        k,
                        mid,
                        opt,
                        now=self.now_ist,
                        r=rfr,
                        atr_pct=plan.get("atr_pct"),
                    )
                    lot = getattr(self.settings, "LOT_SIZE", 50)
                    planned_delta_units = (
                        (est.delta or 0.0) * lot * int(plan["qty_lots"])
                    )
                    plan["planned_delta_units"] = round(planned_delta_units, 1)

            portfolio_delta_units = self._portfolio_delta_units()
            gmode = self.now_ist.weekday() == 1 and self.now_ist.time() >= dt_time(
                14, 45
            )

            ok, reason, det = self.risk_engine.pre_trade_check(
                equity_rupees=self._active_equity(),
                plan=plan,
                runner=self,
                exposure=exposure,
                intended_symbol=str(sym),
                intended_lots=qty_lots,
                lot_size=lot_size,
                entry_price=float(entry or 0.0),
                stop_loss_price=float(sl or 0.0),
                spot_price=float(getattr(self, "last_spot", 0.0) or 0.0),
                quote=plan.get("quote"),
                planned_delta_units=planned_delta_units,
                portfolio_delta_units=portfolio_delta_units,
                gamma_mode=gmode,
            )
            flow["portfolio_greeks"] = {
                "delta_units": round(portfolio_delta_units, 1),
                "gamma_mode": gmode,
            }
            if planned_delta_units is not None:
                flow["planned_delta_units"] = round(planned_delta_units, 1)

            if not ok:
                plan["has_signal"] = False
                # Preserve any upstream block reason determined before risk checks.
                existing_block_reason = plan.get("reason_block")
                if not existing_block_reason:
                    plan["reason_block"] = reason
                # Snapshot reasons to avoid mutating shared references from upstream signals.
                existing_reasons: List[str] = list(plan.get("reasons", []))
                risk_reason = f"risk:{reason}" if reason else "risk"
                if risk_reason not in existing_reasons:
                    existing_reasons.append(risk_reason)
                plan["reasons"] = existing_reasons
                flow["reason_block"] = plan.get("reason_block") or reason
                reason_details = flow.setdefault("reason_details", {})
                if reason:
                    reason_details.setdefault(reason, det)
                if reason == "cap_lt_one_lot":
                    reason_details["cap_lt_one_lot"] = det
                plan.setdefault("risk_details", det)
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.last_plan = plan
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            plan_token = plan.get("token") or plan.get("option_token")
            micro_guard: Dict[str, Any] = {}
            micro_ok = True
            if plan_token is not None:
                try:
                    micro_ok, micro_guard = limits.pretrade_micro_checks(
                        getattr(self, "source", None),
                        plan_token,
                        self.settings,
                    )
                except Exception:
                    micro_ok = True
                    micro_guard = {}
                    self.log.debug("pretrade_micro_checks failed", exc_info=True)
            else:
                micro_guard = {"reason": "no_token"}
            flow["pretrade_micro"] = micro_guard
            if not micro_ok:
                reason_micro = (
                    micro_guard.get("reason")
                    or micro_guard.get("block_reason")
                    or "micro"
                )
                extras = dict(micro_guard)
                extras["token"] = plan_token
                self.log.info("signal.block_micro", extra=extras)
                reasons = plan.setdefault("reasons", [])
                if reason_micro and reason_micro not in reasons:
                    reasons.append(reason_micro)
                plan["reason_block"] = plan.get("reason_block") or reason_micro
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._record_plan(plan)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            # ---- sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(plan.get("entry")),
                stop=float(plan.get("sl")),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
                spot_price=float(getattr(self, "last_spot", plan.get("entry", 0.0)) or 0.0),
                delta=plan.get("delta"),
                quote=plan.get("quote"),
            )
            flow["sizing"] = diag
            block_reason = diag.get("block_reason")
            flow["qty"] = int(qty)
            flow["equity"] = self._active_equity()
            flow["risk_rupees"] = round(
                float(diag.get("rupee_risk_per_lot", 0.0))
                * float(diag.get("lots_final", 0)),
                2,
            )
            flow["trades_today"] = self.risk.trades_today
            flow["consecutive_losses"] = self.risk.consecutive_losses
            size_ctx = {
                "token": plan_token,
                "entry_price": float(entry or 0.0),
                "lot_size": lot_size,
                "cap_info": diag.get("cap_info"),
                "micro": micro_guard,
                "reason": diag.get("reason")
                or ("ok" if qty > 0 else block_reason or "qty_zero"),
            }
            diagnostics.log_trade_context(self.log, size_ctx)
            if qty <= 0:
                self.log.info("signal.block_size", extra=size_ctx)
                existing_reason = plan.get("reason_block")
                reason = (
                    existing_reason
                    or size_ctx.get("reason")
                    or block_reason
                    or "qty_zero"
                )
                if not existing_reason:
                    plan["reason_block"] = reason
                reasons = plan.setdefault("reasons", [])
                if block_reason == "cap_lt_one_lot":
                    cap_val = diag.get("cap")
                    unit_val = diag.get("unit_notional")
                    eq_val = diag.get("equity")
                    cap_abs_val = diag.get("cap_abs")
                    cap_msg_parts = [
                        f"cap={cap_val}" if cap_val is not None else None,
                        f"unit={unit_val}" if unit_val is not None else None,
                        f"equity={eq_val}" if eq_val is not None else None,
                    ]
                    if cap_abs_val:
                        cap_msg_parts.append(f"cap_abs={cap_abs_val}")
                    cap_msg_detail = " ".join([p for p in cap_msg_parts if p])
                    cap_msg = "sizer:cap_lt_one_lot"
                    if cap_msg_detail:
                        cap_msg = f"{cap_msg} ({cap_msg_detail})"
                    if cap_msg not in reasons:
                        reasons.append(cap_msg)
                    flow.setdefault("reason_details", {})["cap_lt_one_lot"] = {
                        "cap": cap_val,
                        "unit": unit_val,
                        "equity": eq_val,
                        "cap_abs": cap_abs_val,
                        "min_equity_needed": diag.get("min_equity_needed"),
                    }
                    reason = plan.get("reason_block") or block_reason or reason
                else:
                    if reason not in reasons:
                        reasons.append(reason)
                flow["reason_block"] = plan.get("reason_block") or reason
                reason = flow["reason_block"]
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.debug(
                    "Signal skipped: quantity %s <= 0 reason=%s", qty, reason
                )
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            planned_lots = int(qty / int(settings.instruments.nifty_lot_size))
            plan["qty_lots"] = planned_lots

            plan_token = plan.get("token") or plan.get("option_token")
            quote_snapshot: Mapping[str, Any] | None = None
            if plan_token:
                quote_snapshot = self._get_cached_full_quote(plan_token)
                if not quote_snapshot:
                    ds_local = getattr(self, "data_source", None)
                    prime_fn_local = getattr(ds_local, "prime_option_quote", None)
                    if callable(prime_fn_local):
                        try:
                            prime_fn_local(plan_token)
                        except Exception:
                            self.log.debug(
                                "prime_option_quote refresh failed", exc_info=True
                            )
                    quote_snapshot = self._get_cached_full_quote(plan_token)

            def _as_float(val: Any) -> float:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            if not quote_snapshot or (
                _as_float(quote_snapshot.get("bid")) <= 0.0
                and _as_float(quote_snapshot.get("ask")) <= 0.0
            ):
                plan.setdefault("reasons", [])
                if "no_quote" not in plan["reasons"]:
                    plan["reasons"].append("no_quote")
                plan["reason_block"] = "no_quote"
                plan["micro"] = {
                    "block_reason": "no_quote",
                    "spread_pct": None,
                    "depth_ok": None,
                }
                flow["reason_block"] = plan["reason_block"]
                self._record_plan(plan)
                self._last_flow_debug = flow
                self.log.info("micro precheck: no_quote token=%s", plan_token)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            quote_snapshot = dict(quote_snapshot)
            plan["quote"] = quote_snapshot
            plan["quote_src"] = quote_snapshot.get("source")

            ok_micro, micro = self.executor.micro_decision(
                quote=quote_snapshot,
                qty_lots=planned_lots,
                lot_size=int(settings.instruments.nifty_lot_size),
                max_spread_pct=float(
                    getattr(settings.executor, "max_spread_pct", 0.35)
                ),
                depth_mult=int(getattr(settings.executor, "depth_multiplier", 5)),
                side=str(plan.get("action") or ""),
            )
            plan["micro"] = micro
            if micro:
                spread_val = micro.get("spread_pct")
                try:
                    spread_ratio = (
                        float(spread_val) / 100.0 if spread_val is not None else None
                    )
                except (TypeError, ValueError):
                    spread_ratio = None
                plan_token = getattr(plan, "token", None)
                if plan_token is None:
                    plan_token = plan.get("option_token") or plan.get("token")
                quote_obj = plan.get("quote") or {}
                quote_src = None
                if isinstance(quote_obj, Mapping):
                    quote_src = quote_obj.get("source")
                quote_src = quote_src or plan.get("quote_src")
                side_val = plan.get("action") or plan.get("side") or ""
                spread_for_log = (
                    (spread_ratio if spread_ratio is not None else math.nan) * 100.0
                )
                self.log.info(
                    "micro precheck: token=%s side=%s lots=%d quote_src=%s spread=%.2f%%",
                    plan_token,
                    side_val,
                    planned_lots,
                    quote_src,
                    spread_for_log,
                )
            self._preview_candidate(plan, micro)
            score_val = float(plan.get("score") or 0.0)
            plan["score"] = score_val
            self._log_score_state(score_val)
            self._log_confidence_state(plan)
            min_score = self._min_score_threshold()
            if score_val < min_score and not plan.get("reason_block"):
                plan["reason_block"] = "score_low"
                flow["reason_block"] = plan["reason_block"]
                plan.setdefault("reasons", []).extend(
                    [f"score={score_val:.2f}", f"min={min_score:.2f}"]
                )
                self._last_flow_debug = flow
                self._record_plan(plan)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return
            if (
                ok_micro
                and score_val >= int(settings.strategy.min_signal_score)
                and not plan.get("reason_block")
            ):
                plan["has_signal"] = True
                self._emit_diag(plan, micro)
            else:
                plan["has_signal"] = False
                if plan.get("reason_block") in ("", None) and not ok_micro:
                    plan["reason_block"] = "microstructure"
                    if micro:
                        reasons = plan.setdefault("reasons", [])
                        block_reason = micro.get("block_reason") or "fail"
                        detail = (
                            f"micro:{block_reason} req={micro.get('required_qty')}"
                            f" avail={micro.get('depth_available')}"
                        )
                        if detail not in reasons:
                            reasons.append(detail)
                flow["reason_block"] = flow.get("reason_block") or plan.get(
                    "reason_block"
                )
                self._last_flow_debug = flow
                self._record_plan(plan)
                self._log_decisive_event(
                    label="blocked",
                    signal=plan,
                    reason_block=flow.get("reason_block"),
                )
                return

            self._record_plan(plan)

            # ---- execution (support both executors)
            placed_ok = False
            tp_basis = getattr(settings, "tp_basis", "premium").lower()
            entry_px = float(
                plan.get("opt_entry") if tp_basis == "premium" else plan.get("entry")
            )
            sl_px = float(
                plan.get("opt_sl") if tp_basis == "premium" else plan.get("sl")
            )
            tp2_px = float(
                plan.get("opt_tp2") if tp_basis == "premium" else plan.get("tp2")
            )
            if hasattr(self.executor, "place_order"):
                hash_input = {
                    "action": plan["action"],
                    "option_type": plan["option_type"],
                    "strike": float(plan["strike"]),
                    "entry": entry_px,
                    "sl": sl_px,
                    "tp": tp2_px,
                    "qty": int(qty),
                }
                client_oid = hashlib.sha256(
                    json.dumps(hash_input, sort_keys=True).encode()
                ).hexdigest()[:20]
                exec_payload = {
                    "action": plan["action"],
                    "quantity": int(qty),
                    "entry_price": entry_px,
                    "stop_loss": sl_px,
                    "take_profit": tp2_px,
                    "strike": float(plan["strike"]),
                    "option_type": plan["option_type"],
                    "client_oid": client_oid,
                }
                if plan.get("quote"):
                    exec_payload["quote"] = dict(plan["quote"])
                placed_ok = bool(self.order_manager.submit(exec_payload))
                if placed_ok and hasattr(self.executor, "create_trade_fsm"):
                    try:
                        plan_exec = dict(plan)
                        plan_exec["entry"] = entry_px
                        plan_exec["sl"] = sl_px
                        plan_exec["tp2"] = tp2_px
                        plan_exec["client_oid"] = client_oid
                        fsm = self.executor.create_trade_fsm(plan_exec)
                        self.executor.place_trade(fsm)
                    except Exception:
                        self.log.debug("FSM enqueue failed", exc_info=True)
            elif hasattr(self.executor, "place_entry_order"):
                side = "BUY" if str(plan["action"]).upper() == "BUY" else "SELL"
                symbol = getattr(settings.instruments, "trade_symbol", "NIFTY")
                token = int(getattr(settings.instruments, "instrument_token", 0))
                oid = self.executor.place_entry_order(
                    token=token,
                    symbol=symbol,
                    side=side,
                    quantity=int(qty),
                    price=entry_px,
                )
                placed_ok = bool(oid)
                if placed_ok and hasattr(self.executor, "create_trade_fsm"):
                    try:
                        plan_exec = dict(plan)
                        plan_exec["trade_id"] = oid
                        plan_exec["entry"] = entry_px
                        plan_exec["sl"] = sl_px
                        plan_exec["tp2"] = tp2_px
                        fsm = self.executor.create_trade_fsm(plan_exec)
                        self.executor.place_trade(fsm)
                    except Exception:
                        self.log.debug("FSM enqueue failed", exc_info=True)
                if placed_ok and hasattr(self.executor, "setup_gtt_orders"):
                    try:
                        self.executor.setup_gtt_orders(
                            record_id=oid,
                            sl_price=sl_px,
                            tp_price=tp2_px,
                        )
                    except Exception as e:
                        self.log.warning("setup_gtt_orders failed: %s", e)
            else:
                self.log.error("No known execution method found on OrderExecutor")

            flow["executed"] = placed_ok
            if placed_ok:
                self._last_trade_time = self.now_ist
                self._auto_relax_active = False
                metrics.inc_orders(placed=1)
            else:
                metrics.inc_orders(rejected=1)
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")
                err = getattr(self.executor, "last_error", None)
                if err:
                    self._notify(f"⚠️ Execution error: {err}")

            if placed_ok:
                self.risk.trades_today += 1
                self._last_signal_at = time.time()
                self._notify(
                    f"✅ Placed: {plan['action']} {qty} {plan['option_type']} {int(plan['strike'])} "
                    f"@ {float(plan.get('entry')):.2f} (SL {float(plan.get('sl')):.2f}, "
                    f"TP {float(plan.get('tp2')):.2f})"
                )

            self._last_flow_debug = flow
            if placed_ok:
                self._log_decisive_event(
                    label="action",
                    signal=plan,
                    reason_block=None,
                )
                return
            self._log_decisive_event(
                label="blocked",
                signal=plan,
                reason_block=flow.get("reason_block"),
            )
            return

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_error = str(e)
            self._last_flow_debug = flow
            self.log.exception("process_tick error: %s", e)
            signal_payload = locals().get("plan")
            signal_dict = signal_payload if isinstance(signal_payload, dict) else None
            self._log_decisive_event(
                label="blocked",
                signal=signal_dict,
                reason_block=flow.get("reason_block"),
            )
            return
        finally:
            if getattr(self, "trace_ticks_remaining", 0) > 0:
                p = self.last_plan or {}
                m = p.get("micro") or {}
                self.log.info(
                    "TRACE regime=%s score=%s atr%%=%.2f spread%%=%s depth=%s rr=%s entry=%s sl=%s tp1=%s tp2=%s opt_sl=%s opt_tp1=%s opt_tp2=%s block=%s reasons=%s",
                    p.get("regime"),
                    p.get("score"),
                    float(p.get("atr_pct") or 0.0),
                    m.get("spread_pct"),
                    m.get("depth_ok"),
                    p.get("rr"),
                    p.get("entry"),
                    p.get("sl"),
                    p.get("tp1"),
                    p.get("tp2"),
                    p.get("opt_sl"),
                    p.get("opt_tp1"),
                    p.get("opt_tp2"),
                    p.get("reason_block"),
                    p.get("reasons"),
                )
                self.trace_ticks_remaining -= 1

    # one-shot tick used by Telegram
    def runner_tick(self, *, dry: bool = False) -> Dict[str, Any]:
        prev = bool(settings.allow_offhours_testing)
        try:
            if dry:
                setattr(settings, "allow_offhours_testing", True)
            self.process_tick(tick=None)
            return dict(self._last_flow_debug)
        finally:
            setattr(settings, "allow_offhours_testing", prev)

    def get_current_atm(self) -> Dict[str, dict]:
        under = self.last_spot or 0.0
        if under <= 0 or not self.option_resolver:
            return {}
        now = self._now_ist()
        return {
            k: self.option_resolver.resolve_atm(self.under_symbol, under, k, now)
            for k in ("CE", "PE")
        }

    def get_current_l1(self) -> Optional[Dict[str, Any]]:
        if not self.kite:
            return None
        atm = self.get_current_atm().get("CE")
        ts = atm.get("tradingsymbol") if atm else None
        if not ts:
            return None
        try:
            return self.kite.quote([f"NFO:{ts}"]).get(f"NFO:{ts}")  # type: ignore[attr-defined]
        except Exception:
            return None

    def get_probe_info(self) -> Dict[str, Any]:
        plan = self.last_plan or {}
        return {
            "start": plan.get("probe_window_from"),
            "end": plan.get("probe_window_to"),
            "bars": plan.get("bar_count") or plan.get("bars"),
            "last_bar_ts": plan.get("last_bar_ts"),
            "bar_age_s": plan.get("lag_s") or plan.get("last_bar_lag_s"),
            "tick_age_s": None,
            "source": plan.get("data_source"),
        }

    def run_backtest(self, csv_path: Optional[str] = None) -> str:
        """Run backtest on a CSV file and return a summary string."""
        try:
            path = (
                Path(csv_path).expanduser().resolve()
                if csv_path
                else Path(__file__).resolve().parent.parent / "data" / "nifty_ohlc.csv"
            )
            if path.exists():
                try:
                    feed = SpotFeed.from_csv(str(path))
                except KeyError as e:
                    self.log.error("Backtest failed: %s", e)
                    return f"Backtest error: {e}"
            else:
                df = make_synth_1m(start=datetime.now())
                feed = SpotFeed(df=df, tz=ZoneInfo("Asia/Kolkata"))
            cfg = try_load(resolve_config_path(), None)
            risk = RiskEngine(LimitConfig(tz=cfg.tz))
            sim = SimConnector()
            engine = BacktestEngine(
                feed,
                cfg,
                risk,
                sim,
                outdir=tempfile.mkdtemp(prefix="bt_"),
            )
            summary = engine.run()
            trades = summary.get("trades", 0)
            pnl = float(sum(t.get("pnl_rupees", 0.0) for t in engine.trades))
            wins = sum(1 for t in engine.trades if t.get("pnl_rupees", 0.0) > 0)
            win_rate = (wins / trades * 100.0) if trades else 0.0
            return f"Backtest done: trades={trades}, win%={win_rate:.2f}, pnl={pnl:.2f}"
        except Exception as e:
            # Log the error message without the full traceback to keep logs concise
            self.log.error("Backtest failed: %s", e, exc_info=False)
            return f"Backtest error: {e}"

    def health_check(self) -> Dict[str, Any]:
        """Perform lightweight self-checks and return status snapshot."""
        self._refresh_equity_if_due(silent=True)
        try:
            _ = self._fetch_spot_ohlc()
        except Exception as e:
            self.log.debug("Passive data refresh warn: %s", e)
        try:
            if hasattr(self.executor, "health_check"):
                self.executor.health_check()
        except Exception as e:
            self.log.warning("Executor health check warning: %s", e)
        self._last_error = None
        status = self.get_status_snapshot()
        status["within_window"] = bool(self.within_window)
        status["ok"] = True
        return status

    def shutdown(self) -> None:
        """Graceful shutdown used by /stop or process exit."""
        try:
            # IMPORTANT: use cancel_all_orders (compat) instead of nonexistent close_all_positions
            if hasattr(self.executor, "cancel_all_orders"):
                self.executor.cancel_all_orders()
            if hasattr(self.executor, "shutdown"):
                self.executor.shutdown()
        except Exception:
            self.log.warning("Executor shutdown encountered an error", exc_info=True)

    # ---------------- equity & risk ----------------
    def _refresh_equity_if_due(self, silent: bool = False) -> None:
        now = time.time()
        if not settings.risk.use_live_equity:
            self._max_daily_loss_rupees = self._equity_cached_value * float(
                settings.risk.max_daily_drawdown_pct
            )
            return
        if (now - self._equity_last_refresh_ts) < int(
            settings.risk.equity_refresh_seconds
        ):
            return

        new_eq = None
        if self.kite is not None:
            try:
                margins = self.kite.margins()  # type: ignore[attr-defined]
                if isinstance(margins, dict):
                    # Typical structure: {'equity': {'net': ..., 'available': {'cash': ...}}}
                    segment = (
                        margins.get("equity")
                        if isinstance(margins.get("equity"), dict)
                        else margins
                    )
                    if isinstance(segment, dict):
                        # First try direct numeric fields (net/cash/final)
                        for k in ("net", "cash", "final", "equity"):
                            v = segment.get(k)
                            if isinstance(v, (int, float)):
                                new_eq = float(v)
                                break
                        # Then drill into nested 'available' dicts
                        if new_eq is None:
                            avail = segment.get("available")
                            if isinstance(avail, dict):
                                for k in ("cash", "net", "equity", "final"):
                                    v = avail.get(k)
                                    if isinstance(v, (int, float)):
                                        new_eq = float(v)
                                        break
                if new_eq is None:
                    new_eq = float(settings.risk.default_equity)
            except Exception as e:
                if not silent:
                    msg = str(e)
                    if "Incorrect 'api_key' or 'access_token'" in msg:
                        self.log.info(
                            "Equity refresh returned placeholder response; using fallback"
                        )
                    else:
                        self.log.warning("Equity refresh failed; using fallback: %s", e)

        self._equity_cached_value = (
            float(new_eq)
            if (isinstance(new_eq, (int, float)) and new_eq > 0)
            else float(settings.risk.default_equity)
        )
        self._max_daily_loss_rupees = self._equity_cached_value * float(
            settings.risk.max_daily_drawdown_pct
        )
        self._equity_last_refresh_ts = now

        if not silent:
            self.log.info(
                "Equity snapshot: ₹%s | Max daily loss: ₹%s",
                f"{self._equity_cached_value:,.0f}",
                f"{self._max_daily_loss_rupees:,.0f}",
            )

    def _active_equity(self) -> float:
        return (
            float(self._equity_cached_value)
            if settings.risk.use_live_equity
            else float(settings.risk.default_equity)
        )

    def _risk_gates_for(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        gates = {
            "market_hours": True,
            "equity_floor": True,
            "daily_drawdown": True,
            "loss_streak": True,
            "trades_per_day": True,
            "sl_valid": True,
        }
        if not self._within_trading_window():
            gates["market_hours"] = False
        if settings.risk.use_live_equity and self._active_equity() < float(
            settings.risk.min_equity_floor
        ):
            gates["equity_floor"] = False
        if self.risk.day_realized_loss >= self._max_daily_loss_rupees:
            gates["daily_drawdown"] = False
        # loss streak cooldown logic
        now = self._now_ist()
        cooldown_until = self._loss_cooldown.active_until(now)
        if cooldown_until is not None:
            self.risk.loss_cooldown_until = cooldown_until
            gates["loss_streak"] = False
        else:
            if self.risk.loss_cooldown_until is not None:
                self.risk.consecutive_losses = 0
            self.risk.loss_cooldown_until = None
        limit = int(settings.risk.consecutive_loss_limit)
        if gates.get("loss_streak", True) and self.risk.consecutive_losses >= limit:
            gates["loss_streak"] = False
        if self.risk.trades_today >= int(settings.risk.max_trades_per_day):
            gates["trades_per_day"] = False
        entry = signal.get("entry") or signal.get("entry_price")
        stop = signal.get("sl") or signal.get("stop_loss")
        try:
            entry_f = float(entry)
        except (TypeError, ValueError):
            gates["sl_valid"] = False
            self._last_error = f"invalid entry_price: {entry}"
            self.log.warning("Invalid entry_price: %r", entry)
            return {k: bool(v) for k, v in gates.items()}
        try:
            stop_f = float(stop)
        except (TypeError, ValueError):
            gates["sl_valid"] = False
            self._last_error = f"invalid stop_loss: {stop}"
            self.log.warning("Invalid stop_loss: %r", stop)
            return {k: bool(v) for k, v in gates.items()}
        if abs(entry_f - stop_f) <= float(getattr(settings.executor, "tick_size", 0.0)):
            # stop loss must differ from entry by at least one tick
            gates["sl_valid"] = False
        return {k: bool(v) for k, v in gates.items()}

    def _min_score_threshold(self) -> float:
        base = float(getattr(self.strategy_cfg, "min_score", 0.35))
        now = self.now_ist
        last = self._last_trade_time
        if now.tzinfo is None and last.tzinfo is not None:
            last = last.replace(tzinfo=None)
        elif now.tzinfo is not None and last.tzinfo is None:
            last = last.replace(tzinfo=now.tzinfo)
        minutes_since = (now - last).total_seconds() / 60.0
        relax_after = getattr(getattr(self, "strategy", None), "auto_relax_after_min", 30)
        enabled = getattr(getattr(self, "strategy", None), "auto_relax_enabled", False)
        relax = 0.0
        if enabled and relax_after > 0:
            steps = int(minutes_since // float(relax_after))
            relax = min(steps * 0.05, 0.1)
        if relax > 0.0:
            relaxed = max(base - relax, 0.25)
            self._auto_relax_active = True
            runtime_metrics.set_auto_relax(relax)
            _log_throttled(
                "auto_relax",
                logging.INFO,
                "auto_relax active %.1fmin, min_score %.2f",
                minutes_since,
                relaxed,
            )
            return relaxed
        self._auto_relax_active = False
        runtime_metrics.set_auto_relax(0.0)
        return base

    def _calculate_quantity_diag(
        self,
        *,
        entry: float,
        stop: float,
        lot_size: int,
        equity: float,
        spot_price: float | None = None,
        delta: float | None = None,
        quote: Dict | None = None,
    ) -> Tuple[int, Dict]:
        entry_f = float(entry)
        stop_f = float(stop)
        lot_size_i = int(lot_size)
        equity_f = float(equity)

        if spot_price is None:
            spot_price = getattr(self, "last_spot", None)
        spot_price_f = float(spot_price) if spot_price is not None else entry_f

        sl_points_entry = max(0.5, abs(entry_f - stop_f))

        qty, lots, diag = self._position_sizer.size_from_signal(
            entry_price=entry_f,
            stop_loss=stop_f,
            lot_size=lot_size_i,
            equity=equity_f,
            spot_price=spot_price_f,
            spot_sl_points=sl_points_entry,
            delta=None if delta is None else float(delta),
            quote=quote,
        )

        diag_aug = dict(diag)
        diag_aug["rupee_risk_per_lot"] = float(diag.get("risk_per_lot", 0.0))
        diag_aug["lots_raw"] = int(diag.get("calc_lots", lots))
        diag_aug["lots_final"] = int(diag.get("lots_final", diag.get("lots", lots)))
        unit_notional = float(diag.get("unit_notional", 0.0))
        exposure_est = unit_notional * diag_aug["lots_final"]
        diag_aug["exposure_notional_est"] = (
            round(exposure_est, 2) if math.isfinite(exposure_est) else float("inf")
        )
        cap_val = diag.get("cap", diag.get("exposure_cap", 0.0))
        if isinstance(cap_val, (int, float)):
            cap_f = float(cap_val)
            diag_aug["max_notional_cap"] = (
                round(cap_f, 2) if math.isfinite(cap_f) else cap_f
            )
        else:
            diag_aug["max_notional_cap"] = 0.0
        diag_aug["sl_points"] = round(sl_points_entry, 4)
        diag_aug["entry"] = round(entry_f, 4)
        diag_aug["stop"] = round(stop_f, 4)
        diag_aug["cap_info"] = {
            "cap": diag_aug.get("cap"),
            "cap_abs": diag_aug.get("cap_abs"),
            "unit_notional": diag_aug.get("unit_notional"),
            "max_lots_exposure": diag_aug.get("max_lots_exposure"),
            "max_lots_risk": diag_aug.get("max_lots_risk"),
            "min_equity_needed": diag_aug.get("min_equity_needed"),
        }
        diag_aug["reason"] = (
            "ok" if int(qty) > 0 else diag_aug.get("block_reason") or "qty_zero"
        )
        return int(qty), diag_aug

    def _on_trade_closed(self, pnl: float) -> None:
        """Update risk state and forward realised PnL to risk engine."""

        if pnl < 0:
            loss = abs(pnl)
            self._risk_state.realised_loss += loss
            self.risk.day_realized_loss += loss
            self.risk.consecutive_losses += 1
        else:
            self.risk.consecutive_losses = 0
        self.risk.day_realized_pnl += pnl
        self.risk.trades_today += 1
        now = self.now_ist
        self._last_trade_time = now
        cooldown_until = self._loss_cooldown.register_trade(
            now=now,
            pnl=pnl,
            streak=self.risk.consecutive_losses,
            day_loss=self.risk.day_realized_loss,
            max_daily_loss=self._max_daily_loss_rupees,
        )
        self.risk.loss_cooldown_until = cooldown_until
        self._auto_relax_active = False
        try:
            self.risk_engine.on_trade_closed(pnl_R=pnl)
        except Exception:
            self.log.debug("risk_engine on_trade_closed failed", exc_info=True)
        try:
            record_trade(pnl, runtime_metrics.slippage_bps)
        except Exception:
            self.log.debug("record_trade failed", exc_info=True)

    # ---------------- data helpers ----------------
    def _fetch_spot_ohlc(self) -> Optional[pd.DataFrame]:
        """
        Build SPOT OHLC frame using LiveKiteSource with configured lookback.
        If no valid token is configured or broker returns empty data, synthesize a 1-bar DF from LTP.
        """
        if self.data_source is None:
            return None

        def _attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
            if "bb_width" not in df.columns:
                try:
                    df["bb_width"] = calculate_bb_width(
                        df["close"], use_percentage=True
                    )
                except Exception:
                    df["bb_width"] = pd.Series([float("nan")] * len(df), index=df.index)
            if "adx" not in df.columns and not any(
                c.startswith("adx_") for c in df.columns
            ):
                try:
                    if len(df) >= 14:
                        adx, dip, dim = calculate_adx(df)
                        df["adx"], df["di_plus"], df["di_minus"] = adx, dip, dim
                    else:
                        raise ValueError("insufficient")
                except Exception:
                    na = pd.Series([float("nan")] * len(df), index=df.index)
                    df["adx"], df["di_plus"], df["di_minus"] = na, na, na
            return df

        try:
            need = required_bars(self.strategy_cfg)
            pad = int(getattr(settings.data, "lookback_padding_bars", 5))
            lookback = max(int(settings.data.lookback_minutes), need + pad)
            if lookback <= 0:
                self.log.warning("Adjusted OHLC window invalid; aborting")
                return None

            now = floor_to_minute(self._now_ist(), self._now_ist().tzinfo or TZ)

            tz = now.tzinfo or TZ

            # Build today's session using configured start/end times
            session_start = now.replace(
                hour=self._start_time.hour,
                minute=self._start_time.minute,
                second=0,
                microsecond=0,
            )
            session_end = now.replace(
                hour=self._end_time.hour,
                minute=self._end_time.minute,
                second=0,
                microsecond=0,
            )
            if session_end <= session_start:
                session_end += timedelta(days=1)

            if now < session_start or now > session_end:
                # Outside today's session: shift to the previous trading session
                ref = now
                prev_start, _ = prev_session_bounds(ref.astimezone(TZ))
                prev_day = prev_start.astimezone(tz).date()
                session_start = datetime(
                    prev_day.year,
                    prev_day.month,
                    prev_day.day,
                    self._start_time.hour,
                    self._start_time.minute,
                    tzinfo=tz,
                )
                session_end = datetime(
                    prev_day.year,
                    prev_day.month,
                    prev_day.day,
                    self._end_time.hour,
                    self._end_time.minute,
                    tzinfo=tz,
                )
                if session_end <= session_start:
                    session_end += timedelta(days=1)

            end = session_end if now > session_end or now < session_start else now

            start = end - timedelta(minutes=lookback)
            # Only clamp to session bounds when fetching for a completed session.
            if (now < session_start or now > session_end) and start < session_start:
                start = session_start
            if start >= end:
                start = end - timedelta(minutes=1)

            # Resolve token with fallbacks
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
            if token <= 0:
                token = int(getattr(settings.instruments, "spot_token", 0) or 0)

            timeframe = str(getattr(settings.data, "timeframe", "minute"))

            if token > 0:
                df = self.data_source.fetch_ohlc_df(
                    token=token,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                )
                need = {"open", "high", "low", "close", "volume"}
                min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 0))
                valid = need.issubset(df.columns)
                rows = len(df)

                if not valid or rows < min_bars:
                    # First attempt yielded insufficient data; try again with expanded window
                    if rows < min_bars:
                        self.log.warning(
                            "historical_data short %s<%s; refetching with expanded lookback",
                            rows,
                            min_bars,
                        )
                    else:
                        self.log.warning(
                            "historical_data empty for token=%s interval=%s window=%s..%s; refetching with expanded lookback.",
                            token,
                            timeframe,
                            start.isoformat(),
                            end.isoformat(),
                        )

                    start2 = end - timedelta(minutes=lookback * 2)
                    df2 = self.data_source.fetch_ohlc_df(
                        token=token,
                        start=start2,
                        end=end,
                        timeframe=timeframe,
                    )
                    if not df2.empty and need.issubset(df2.columns):
                        df = df2.sort_index()
                        valid = True
                        rows = len(df)

                min_req = int(getattr(settings.strategy, "min_bars_required", min_bars))
                if valid and rows >= min_bars:
                    self._last_fetch_ts = time.time()
                    df = _attach_indicators(df.sort_index())
                    try:
                        self.last_spot = float(df["close"].iloc[-1])
                    except Exception:
                        pass
                    self._ohlc_cache = df
                    return df
                if valid and rows >= min_req:
                    self._last_fetch_ts = time.time()
                    df = _attach_indicators(df.sort_index())
                    try:
                        self.last_spot = float(df["close"].iloc[-1])
                    except Exception:
                        pass
                    self._ohlc_cache = df
                    return df

                self.log.error(
                    "Insufficient historical_data (%s<%s) after expanded fetch.",
                    rows,
                    min_bars,
                )
                self._last_error = "no_historical_data"
                # Avoid spamming Telegram with automatic notifications for missing historical data
                self.log.warning(
                    "⚠️ Historical data unavailable from broker — check credentials or subscription."
                )

            # Fallback: synthesize a single bar from trade symbol/token LTP
            sym = getattr(settings.instruments, "trade_symbol", None)
            ltp = self.data_source.get_last_price(sym if sym else token)
            if isinstance(ltp, (int, float)) and ltp > 0:
                ts = end
                df = pd.DataFrame(
                    {
                        "open": [ltp],
                        "high": [ltp],
                        "low": [ltp],
                        "close": [ltp],
                        "volume": [0],
                    },
                    index=[ts],
                )
                self._last_fetch_ts = time.time()
                try:
                    self.last_spot = float(ltp)
                except Exception:
                    pass
                df = _attach_indicators(df)
                self._ohlc_cache = df
                return df

            # If we get here, we truly have nothing
            self._ohlc_cache = None
            return None

        except Exception as e:
            self.log.warning("OHLC fetch failed: %s", e)
            self._ohlc_cache = None
            return None

    # ---------------- session/window ----------------
    def _ensure_day_state(self) -> None:
        today = self._today_ist()
        if today.date() != self.risk.trading_day.date():
            self.risk = RiskState(trading_day=today)
            self._loss_cooldown.reset_for_new_day()
            self._notify("🔁 New trading day — risk counters reset")

    def _within_trading_window(self, adx_val: Optional[float] = None) -> bool:
        """Return ``True`` if current IST time falls within the configured window.

        Start and end times are sourced from the environment via ``settings``
        so trading hours can be tuned without modifying code.
        """

        _ = adx_val  # legacy arg ignored; window no longer depends on ADX
        now = self._now_ist().time()
        start = getattr(
            self, "_start_time", self._parse_hhmm(settings.risk.trading_window_start)
        )
        end = getattr(
            self, "_end_time", self._parse_hhmm(settings.risk.trading_window_end)
        )
        return start <= now <= end

    @staticmethod
    def _parse_hhmm(text: str):
        from datetime import datetime as _dt

        return _dt.strptime(text, "%H:%M").time()

    def _now_ist(self) -> datetime:
        return datetime.now(self.tz)

    @property
    def now_ist(self) -> datetime:
        """Current time in the configured timezone."""
        return self._now_ist()

    def _today_ist(self) -> datetime:
        now = self._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ---------------- Telegram helpers & diagnostics ----------------
    def telemetry_snapshot(self) -> dict:
        """Return a consolidated snapshot of the runner's current state."""
        plan = self.last_plan or {}
        dp_health = getattr(self.data_source, "api_health", lambda: {})()
        oc_health = getattr(self.order_executor, "api_health", lambda: {})()
        router = getattr(self.order_executor, "router_health", lambda: {})()
        risk = getattr(self, "risk_engine", None)
        last_eval = getattr(self, "last_eval_ts", None)
        if isinstance(last_eval, datetime):
            last_eval = last_eval.isoformat()
        ts = getattr(self, "now_ist", None)
        ts_val = ts.isoformat() if callable(getattr(ts, "isoformat", None)) else None
        return {
            "ts": ts_val,
            "eval_count": getattr(self, "eval_count", 0),
            "last_eval_ts": last_eval,
            "bars": {
                "bar_count": plan.get("bar_count"),
                "last_bar_ts": plan.get("last_bar_ts"),
                "data_source": plan.get("data_source", "broker"),
            },
            "signal": {
                "action": plan.get("action"),
                "regime": plan.get("regime"),
                "score": plan.get("score"),
                "atr_pct": plan.get("atr_pct"),
                "rr": plan.get("rr"),
                "reason_block": plan.get("reason_block"),
                "reasons": plan.get("reasons"),
                "micro": plan.get("micro"),
                "entry": plan.get("entry"),
                "sl": plan.get("sl"),
                "tp1": plan.get("tp1"),
                "tp2": plan.get("tp2"),
            },
            "components": getattr(self, "_active_names", {}),
            "api_health": {
                "hist": dp_health.get("hist"),
                "quote": dp_health.get("quote"),
                "orders": oc_health.get("orders"),
                "modify": oc_health.get("modify"),
            },
            "router": router,
            "risk": risk.snapshot() if risk else {},
            "portfolio": getattr(self, "portfolio_greeks", None),
        }

    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self.last_plan or {})

    def get_recent_bars(self, n: int = 5) -> str:
        if not self.data_source:
            return "data_source_unavailable"

        try:
            df = self.data_source.get_recent_bars(n)
        except Exception as e:  # pragma: no cover - defensive log
            self.log.warning("get_recent_bars failed: %s", e)
            return "no data"

        if df.empty:
            self.log.warning("No bars fetched for lookback=%s", n)
            return "no data"
        if len(df) < n:
            self.log.warning("Fetched %s bars (<%s) from data source", len(df), n)

        from src.data.source import render_last_bars

        return render_last_bars(self.data_source, n)

    def enable_trace(self, n: int) -> None:
        self.trace_ticks_remaining = int(max(0, n))

    def disable_trace(self) -> None:
        self.trace_ticks_remaining = 0

    # detailed bundle used by /check
    def build_diag(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    def get_last_flow_debug(self) -> Dict[str, Any]:
        return dict(self._last_flow_debug)

    def ohlc_window(self) -> Optional[pd.DataFrame]:
        """Return the cached OHLC window, if any."""

        return self._ohlc_cache

    @classmethod
    def get_singleton(cls) -> Optional["StrategyRunner"]:
        """Return the most recently created runner instance, if any."""

        return cls._SINGLETON

    def debug_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight state snapshot for diagnostics."""

        df = self._ohlc_cache
        last_ts = None if df is None or df.empty else df.index[-1]
        gates = self._last_flow_debug.get("risk_gates", {})
        return {
            "now": str(self.now_ist),
            "bars": None if df is None else len(df),
            "last_bar_ts": str(last_ts),
            "lag_s": (
                None if last_ts is None else (self.now_ist - last_ts).total_seconds()
            ),
            "eval_count": getattr(self, "eval_count", None),
            "gates": gates,
            "risk_pct": getattr(self.settings.risk, "risk_per_trade", None),
            "rr_threshold": getattr(self.strategy_cfg, "rr_threshold", None),
        }

    def open_trades_provider(self) -> List[Dict[str, Any]]:
        """Return a snapshot of open legs suitable for Telegram diagnostics."""

        snapshot_fn = getattr(self.order_executor, "open_legs_snapshot", None)
        if callable(snapshot_fn):
            return snapshot_fn()
        return []

    def cancel_trade(self, trade_id: str) -> None:
        """Forward a manual cancel request for the given trade id."""

        canceller = getattr(self.order_executor, "cancel_trade", None)
        if callable(canceller):
            canceller(str(trade_id))

    def reconcile_once(self) -> int:
        """Run a single reconciliation step and return updated leg count."""

        reconciler = getattr(self, "reconciler", None)
        if reconciler is None:
            return 0
        return int(reconciler.step(self.now_ist))

    def _build_diag_bundle(self) -> Dict[str, Any]:
        """Health cards for /diag (compact) and /check (detailed)."""
        checks: List[Dict[str, Any]] = []

        # Strategy and data provider info
        checks.append(
            {
                "name": "Strategy",
                "ok": self.strategy is not None,
                "detail": getattr(self, "strategy_name", "unknown"),
            }
        )
        checks.append(
            {
                "name": "Data provider",
                "ok": self.data_source is not None,
                "detail": getattr(self, "data_provider_name", "none"),
            }
        )

        # Telegram wiring
        checks.append(
            {
                "name": "Telegram wiring",
                "ok": bool(self.telegram is not None),
                "detail": (
                    "controller attached" if self.telegram else "missing controller"
                ),
            }
        )

        # Broker session (live flag + kite object)
        live = bool(settings.enable_live_trading)
        checks.append(
            {
                "name": "Broker session",
                "ok": (self.kite is not None) if live else True,
                "detail": (
                    "live mode with kite"
                    if (live and self.kite)
                    else ("dry mode" if not live else "live but kite=None")
                ),
            }
        )

        # Data feed freshness
        age_s = (time.time() - self._last_fetch_ts) if self._last_fetch_ts else 1e9
        checks.append(
            {
                "name": "Data feed",
                "ok": age_s < 120,  # < 2 minutes considered fresh
                "detail": "fresh" if age_s < 120 else "stale/never",
                "hint": (
                    f"age={int(age_s)}s "
                    f"token={int(getattr(settings.instruments,'instrument_token',0) or getattr(settings.instruments,'spot_token',0) or 0)} "
                    f"tf={getattr(settings.data,'timeframe','minute')} lookback={int(getattr(settings.data,'lookback_minutes',15))}m"
                ),
            }
        )

        # Strategy readiness (min bars)
        ready = isinstance(self._last_flow_debug, dict) and int(
            self._last_flow_debug.get("bars", 0)
        ) >= int(getattr(settings.strategy, "min_bars_for_signal", 15))
        checks.append(
            {
                "name": "Strategy readiness",
                "ok": ready,
                "detail": f"bars={int(self._last_flow_debug.get('bars', 0))}",
                "hint": f"min_bars={int(getattr(settings.strategy, 'min_bars_for_signal', 15))}",
            }
        )

        # Risk gates last view
        gates = (
            self._last_flow_debug.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(self._last_flow_debug, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = gates is RISK_GATES_SKIPPED or (
            isinstance(gates, dict) and bool(gates.get("skipped"))
        )
        gates_dict = gates if isinstance(gates, dict) else {}
        gates_ok = True
        if not skipped and gates_dict:
            gates_ok = all(bool(v) for v in gates_dict.values())
        checks.append(
            {
                "name": "Risk gates",
                "ok": gates_ok,
                "detail": (
                    "skipped"
                    if skipped
                    else (
                        ", ".join(
                            [
                                f"{k}={'OK' if v else 'BLOCK'}"
                                for k, v in gates_dict.items()
                            ]
                        )
                        if gates_dict
                        else "no-eval"
                    )
                ),
            }
        )

        # RR check
        rr_ok = (
            bool(self._last_flow_debug.get("rr_ok", True))
            if isinstance(self._last_flow_debug, dict)
            else True
        )
        checks.append(
            {
                "name": "RR threshold",
                "ok": rr_ok,
                "detail": str(self._last_flow_debug.get("plan", {})),
            }
        )

        # Errors
        checks.append(
            {
                "name": "Errors",
                "ok": self._last_error is None,
                "detail": "none" if self._last_error is None else self._last_error,
            }
        )

        ok = all(c.get("ok", False) for c in checks)
        last_sig = (
            (time.time() - self._last_signal_at) < 900
            if self._last_signal_at
            else False
        )  # 15min
        open_legs = [
            {
                "trade": fsm.trade_id,
                "leg": leg.leg_type.name,
                "sym": leg.symbol,
                "state": leg.state.name,
                "filled": leg.filled_qty,
                "qty": leg.qty,
                "avg": leg.avg_price,
                "age_s": int(
                    (datetime.now(timezone.utc) - leg.created_at).total_seconds()
                ),
                "status": fsm.status,
            }
            for fsm in getattr(self.order_executor, "open_trades", lambda: [])()
            for leg in fsm.open_legs()
        ]
        bundle = {
            "ok": ok,
            "checks": checks,
            "last_signal": last_sig,
            "last_flow": dict(self._last_flow_debug),
            "open_legs": open_legs,
            "risk": self.risk_engine.snapshot(),
            "exposure": {
                "notional_rupees": round(self._notional_rupees(), 2),
                "basis": getattr(self.settings, "EXPOSURE_BASIS", "premium"),
                "lots_by_symbol": self._lots_by_symbol(),
            },
        }
        bundle["strategy_cfg"] = {
            "name": self.strategy_cfg.name,
            "version": self.strategy_cfg.version,
            "tz": self.strategy_cfg.tz,
            "atr_band": [self.strategy_cfg.atr_min, self.strategy_cfg.atr_max],
            "min_score": self.strategy_cfg.raw.get("strategy", {}).get(
                "min_score", 0.35
            ),
        }
        return bundle

    # compact one-line summary for /diag
    def get_compact_diag_summary(self) -> Dict[str, Any]:
        """Concise status for /diag without building the full multiline text."""
        bundle = self._build_diag_bundle()
        flow = bundle.get("last_flow", {}) if isinstance(bundle, dict) else {}

        telegram_obj = getattr(self, "telegram", None)
        telegram_ok = bool(
            telegram_obj and telegram_obj.__class__.__name__ != "_NoopTelegram"
        )
        live = bool(settings.enable_live_trading)
        broker_ok = (self.kite is not None) if live else True
        data_fresh = (time.time() - getattr(self, "_last_fetch_ts", 0.0)) < 120
        bars = int(flow.get("bars", 0) or 0)
        min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 15))
        strat_ready = bars >= min_bars
        gates = (
            flow.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(flow, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = gates is RISK_GATES_SKIPPED or (
            isinstance(gates, dict) and bool(gates.get("skipped"))
        )
        gates_dict = gates if isinstance(gates, dict) else {}
        gates_ok = True
        if not skipped and gates_dict:
            gates_ok = all(bool(v) for v in gates_dict.values())
        rr_ok = bool(flow.get("rr_ok", True))
        no_errors = self._last_error is None

        if skipped:
            gate_status = "skipped"
        elif not gates_dict:
            gate_status = "no-eval"
        else:
            gate_status = "ok" if gates_ok else "blocked"

        return {
            "ok": bool(bundle.get("ok", False)),
            "status_messages": {
                "telegram_wiring": "ok" if telegram_ok else "missing",
                "broker_session": (
                    "ok" if broker_ok else ("dry mode" if not live else "missing")
                ),
                "data_feed": "ok" if data_fresh else "stale",
                "strategy_readiness": "ok" if strat_ready else "not ready",
                "risk_gates": gate_status,
                "rr_threshold": "ok" if rr_ok else "blocked",
                "errors": "ok" if no_errors else "present",
            },
        }

    def get_equity_snapshot(self) -> Dict[str, Any]:
        return {
            "use_live_equity": bool(settings.risk.use_live_equity),
            "equity_cached": round(float(self._equity_cached_value), 2),
            "equity_floor": float(settings.risk.min_equity_floor),
            "max_daily_loss_rupees": round(float(self._max_daily_loss_rupees), 2),
            "refresh_seconds": int(settings.risk.equity_refresh_seconds),
        }

    def get_status_snapshot(self) -> Dict[str, Any]:
        try:
            pos = getattr(self.executor, "get_positions_kite", lambda: {})() or {}
        except Exception:
            pos = {}
        try:
            mr = getattr(self.executor, "get_margins_kite", lambda: {})() or {}
        except Exception:
            mr = {}

        market_open = self._within_trading_window(None)
        within_window = (
            (not getattr(settings, "enable_time_windows", True))
            or market_open
            or bool(settings.allow_offhours_testing)
        )
        diag = {
            "time_ist": self._now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite" if self.kite is not None else "Paper",
            "market_open": market_open,
            "within_window": within_window,
            "daily_dd_hit": self.risk.day_realized_loss >= self._max_daily_loss_rupees,
            "cooloff_until": (
                self.risk.loss_cooldown_until.isoformat()
                if self.risk.loss_cooldown_until
                else "-"
            ),
            "cooloff_severity": round(float(self._loss_cooldown.severity), 2),
            "paused": self._paused,
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "day_realized_loss": round(self.risk.day_realized_loss, 2),
            "day_realized_pnl": round(self.risk.day_realized_pnl, 2),
            "active_orders": (
                getattr(self.executor, "open_count", 0)
                if hasattr(self.executor, "open_count")
                else 0
            ),
            "open_positions": len(pos),
            "last_signal_score": (
                float(self._last_signal_debug.get("score") or 0.0)
                if isinstance(self._last_signal_debug, dict)
                else 0.0
            ),
            "strategy": getattr(self, "strategy_name", "unknown"),
            "data_provider": getattr(self, "data_provider_name", "none"),
        }
        if mr:
            diag["margins"] = mr

        mins_since = (self.now_ist - self._last_trade_time).total_seconds() / 60.0
        diag["minutes_since_last_trade"] = round(mins_since, 1)
        if self._auto_relax_active:
            diag.setdefault("banners", []).append("auto_relax")

        plan = self.last_plan or {}
        diag["last_signal"] = {
            "entry": plan.get("entry"),
            "sl": plan.get("sl"),
            "tp1": plan.get("tp1"),
            "tp2": plan.get("tp2"),
            "opt_entry": plan.get("opt_entry"),
            "opt_sl": plan.get("opt_sl"),
            "opt_tp1": plan.get("opt_tp1"),
            "opt_tp2": plan.get("opt_tp2"),
            "opt_atr": plan.get("opt_atr"),
            "opt_atr_pct": plan.get("opt_atr_pct"),
        }
        diag["tp_basis"] = getattr(settings, "tp_basis", "premium")

        portfolio_delta_units = self._portfolio_delta_units()
        gmode = self.now_ist.weekday() == 1 and self.now_ist.time() >= dt_time(14, 45)
        diag["portfolio_greeks"] = {
            "delta_units": round(portfolio_delta_units, 1),
            "gamma_mode": gmode,
        }
        if self.last_plan and self.last_plan.get("planned_delta_units") is not None:
            diag["planned_delta_units"] = float(self.last_plan["planned_delta_units"])

        diag["components"] = {
            "strategy": self._active_names.get("strategy"),
            "data_provider": self._active_names.get("data_provider"),
            "order_connector": self._active_names.get("order_connector"),
        }
        try:
            diag["data_provider_health"] = getattr(
                self.data_source, "health", lambda: {"status": "NA"}
            )()
        except Exception:
            diag["data_provider_health"] = {"status": "NA"}
        try:
            diag["order_connector_health"] = getattr(
                self.order_executor, "health", lambda: {"status": "NA"}
            )()
        except Exception:
            diag["order_connector_health"] = {"status": "NA"}
        diag["api_health"] = {
            "orders": getattr(self.order_executor, "api_health", lambda: {})().get(
                "orders", {}
            ),
            "modify": getattr(self.order_executor, "api_health", lambda: {})().get(
                "modify", {}
            ),
            "hist": getattr(self.data_source, "api_health", lambda: {})().get(
                "hist", {}
            ),
            "quote": getattr(self.data_source, "api_health", lambda: {})().get(
                "quote", {}
            ),
        }
        diag["router"] = getattr(self.order_executor, "router_health", lambda: {})()

        plan = self.last_plan or {}
        diag["event_guard"] = plan.get("event_guard", {"active": False})
        if self.event_cal:
            next_ev = self.event_cal.next_event(self.now_ist)
        else:
            next_ev = None
        if next_ev:
            diag["next_event"] = {
                "name": next_ev.name,
                "guard_start": next_ev.guard_start().isoformat(),
                "guard_end": next_ev.guard_end().isoformat(),
            }
        return diag

    def risk_snapshot(self) -> Dict[str, Any]:
        snap = self.risk_engine.snapshot()
        snap["exposure"] = {
            "notional_rupees": round(self._notional_rupees(), 2),
            "basis": getattr(self.settings, "EXPOSURE_BASIS", "premium"),
            "lots_by_symbol": self._lots_by_symbol(),
        }
        return snap

    def risk_reset_today(self) -> None:
        self.risk_engine.state.cum_R_today = 0.0
        self.risk_engine.state.trades_today = 0
        self.risk_engine.state.consecutive_losses = 0
        self.risk_engine.state.cooloff_until = None
        self.risk_engine.state.roll_R_last10.clear()

    def _lots_by_symbol(self) -> Dict[str, int]:
        lots: Dict[str, int] = {}
        for fsm in getattr(self.order_executor, "open_trades", lambda: [])():
            for leg in fsm.open_legs():
                lots[leg.symbol] = lots.get(leg.symbol, 0) + math.ceil(
                    leg.qty / int(self.settings.instruments.nifty_lot_size)
                )
        return lots

    def _notional_rupees(self) -> float:
        if getattr(self.settings, "EXPOSURE_BASIS", "premium") == "underlying":
            lot_size = int(getattr(self.settings.instruments, "nifty_lot_size", 75))
            total_lots = sum(self._lots_by_symbol().values())
            spot = self.last_spot or 0.0
            return spot * lot_size * total_lots
        total = 0.0
        basis = getattr(self.settings, "EXPOSURE_BASIS", "premium")
        spot = float(getattr(self, "last_spot", 0.0) or 0.0)
        for fsm in getattr(self.order_executor, "open_trades", lambda: [])():
            for leg in fsm.open_legs():
                price = leg.limit_price or leg.avg_price or 0.0
                unit = price if basis == "premium" else spot
                total += unit * leg.qty
        return total

    def _portfolio_delta_units(self) -> float:
        """Return total delta exposure in units across open legs."""
        total = 0.0
        lot = getattr(self.settings, "LOT_SIZE", 50)
        r = float(getattr(self.settings, "RISK_FREE_RATE", 0.065))
        for fsm in getattr(self.order_executor, "open_trades", lambda: [])():
            for leg in fsm.open_legs():
                parsed = (
                    strike_selector.parse_nfo_symbol(leg.symbol)
                    if hasattr(strike_selector, "parse_nfo_symbol")
                    else None
                )
                if not parsed:
                    continue
                k = parsed["strike"]
                opt = parsed["option_type"]
                q = getattr(self.order_executor, "fetch_quote_with_depth", None)
                mid = leg.avg_price or leg.limit_price or 0.0
                if q:
                    qt = q(self.order_executor.kite, leg.symbol)
                    b, a = qt.get("bid"), qt.get("ask")
                    if b and a:
                        mid = (b + a) / 2
                s = getattr(self, "last_spot", 0.0) or 0.0
                atr_pct = (
                    self.last_plan.get("atr_pct", None) if self.last_plan else None
                )
                est = estimate_greeks_from_mid(
                    s, k, mid, opt, now=self.now_ist, r=r, atr_pct=atr_pct
                )
                delta = est.delta or 0.0
                side = 1 if leg.side.name == "BUY" else -1
                contracts = int(leg.qty / lot)
                total += side * delta * lot * contracts
        return total

    def sizing_test(self, entry: float, sl: float) -> Dict[str, Any]:
        qty, diag = self._calculate_quantity_diag(
            entry=float(entry),
            stop=float(sl),
            lot_size=int(settings.instruments.nifty_lot_size),
            equity=self._active_equity(),
        )
        return {"qty": int(qty), "diag": diag}

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    # ---------------- strategy tuning helpers ----------------
    def _strategy_raw_section(self) -> dict[str, Any]:
        raw = getattr(self.strategy_cfg, "raw", None)
        if isinstance(raw, dict):
            return raw.setdefault("strategy", {})
        return {}

    def _strategy_cfg_set(self, key: str, value: Any) -> None:
        cfg = getattr(self, "strategy_cfg", None)
        if cfg is None:
            return
        try:
            setattr(cfg, key, value)
        except Exception:
            self.log.debug("Failed to set strategy_cfg.%s", key, exc_info=True)

    def _set_strategy_setting(self, key: str, value: Any) -> None:
        target = getattr(settings, "strategy", None)
        if target is None:
            return
        try:
            setattr(target, key, value)
        except (AttributeError, ValueError):
            try:
                object.__setattr__(target, key, value)
            except Exception:
                self.log.debug("Unable to set settings.strategy.%s", key, exc_info=True)

    def set_min_score(self, value: int) -> None:
        """Adjust the strict and relaxed signal score thresholds at runtime."""

        try:
            score = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("min score must be an integer") from exc
        if not 0 <= score <= 10:
            raise ValueError("min score must be between 0 and 10")
        relaxed_default = max(2, score - 1)
        relaxed = min(score, relaxed_default)

        self._set_strategy_setting("min_signal_score", score)
        self._set_strategy_setting("min_signal_score_relaxed", relaxed)

        if getattr(self, "strategy", None):
            self.strategy.min_score_strict = score
            self.strategy.min_score_relaxed = relaxed

        self._strategy_cfg_set("min_signal_score", score)
        self._strategy_cfg_set("min_signal_score_relaxed", relaxed)
        raw = self._strategy_raw_section()
        raw["min_signal_score"] = score
        raw["min_signal_score_relaxed"] = relaxed
        self.log.info("min_signal_score -> %s (relaxed=%s)", score, relaxed)

    def set_conf_threshold(self, value: float) -> None:
        """Update the confidence thresholds (strict + relaxed) used by the strategy."""

        try:
            strict = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("confidence threshold must be numeric") from exc
        if not 0.0 <= strict <= 100.0:
            raise ValueError("confidence threshold must be within 0..100")

        current_strict = float(getattr(settings.strategy, "confidence_threshold", 55.0))
        current_relaxed = float(
            getattr(
                settings.strategy,
                "confidence_threshold_relaxed",
                max(0.0, current_strict - 20.0),
            )
        )
        gap = max(0.0, current_strict - current_relaxed)
        relaxed = max(0.0, strict - gap)
        relaxed = min(strict, relaxed)

        self._set_strategy_setting("confidence_threshold", strict)
        self._set_strategy_setting("confidence_threshold_relaxed", relaxed)

        if getattr(self, "strategy", None):
            self.strategy.min_conf_strict = strict / 10.0
            self.strategy.min_conf_relaxed = relaxed / 10.0

        self._strategy_cfg_set("confidence_threshold", strict)
        self._strategy_cfg_set("confidence_threshold_relaxed", relaxed)
        raw = self._strategy_raw_section()
        raw["confidence_threshold"] = strict
        raw["confidence_threshold_relaxed"] = relaxed
        self.log.info(
            "confidence_threshold -> %.2f (relaxed=%.2f)", strict, relaxed
        )

    def set_atr_period(self, value: int) -> None:
        """Change the ATR lookback period used by the signal engine."""

        try:
            period = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("ATR period must be an integer") from exc
        if not 1 <= period <= 200:
            raise ValueError("ATR period must be between 1 and 200")

        self._set_strategy_setting("atr_period", period)

        if getattr(self, "strategy", None):
            self.strategy.atr_period = period

        self._strategy_cfg_set("atr_period", period)
        raw = self._strategy_raw_section()
        raw["atr_period"] = period
        self.log.info("atr_period -> %s", period)

    def set_sl_mult(self, value: float) -> None:
        """Adjust the ATR stop-loss multiplier with validation against TP."""

        try:
            sl_mult = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("SL multiplier must be numeric") from exc
        if not 0.1 <= sl_mult <= 10.0:
            raise ValueError("SL multiplier must be between 0.1 and 10.0")

        tp_mult = float(
            getattr(settings.strategy, "atr_tp_multiplier", getattr(self.strategy, "base_tp_mult", 2.0))
        )
        if sl_mult >= tp_mult:
            raise ValueError("SL multiplier must be less than TP multiplier")

        self._set_strategy_setting("atr_sl_multiplier", sl_mult)

        if getattr(self, "strategy", None):
            self.strategy.base_sl_mult = sl_mult

        self._strategy_cfg_set("atr_sl_multiplier", sl_mult)
        raw = self._strategy_raw_section()
        raw["atr_sl_multiplier"] = sl_mult
        self.log.info("atr_sl_multiplier -> %.2f", sl_mult)

    def set_tp_mult(self, value: float) -> None:
        """Adjust the ATR take-profit multiplier keeping it above SL."""

        try:
            tp_mult = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("TP multiplier must be numeric") from exc
        if not 0.1 <= tp_mult <= 15.0:
            raise ValueError("TP multiplier must be between 0.1 and 15.0")

        sl_mult = float(
            getattr(settings.strategy, "atr_sl_multiplier", getattr(self.strategy, "base_sl_mult", 1.0))
        )
        if tp_mult <= sl_mult:
            raise ValueError("TP multiplier must be greater than SL multiplier")

        self._set_strategy_setting("atr_tp_multiplier", tp_mult)

        if getattr(self, "strategy", None):
            self.strategy.base_tp_mult = tp_mult

        self._strategy_cfg_set("atr_tp_multiplier", tp_mult)
        raw = self._strategy_raw_section()
        raw["atr_tp_multiplier"] = tp_mult
        self.log.info("atr_tp_multiplier -> %.2f", tp_mult)

    def set_trend_boosts(self, tp_boost: float, sl_relax: float) -> None:
        """Tune trend regime adjustments for TP and SL multipliers."""

        tp_adj = float(tp_boost)
        sl_adj = float(sl_relax)
        if not -5.0 <= tp_adj <= 5.0:
            raise ValueError("trend TP boost must be between -5.0 and 5.0")
        if not -5.0 <= sl_adj <= 5.0:
            raise ValueError("trend SL relax must be between -5.0 and 5.0")

        self._set_strategy_setting("trend_tp_boost", tp_adj)
        self._set_strategy_setting("trend_sl_relax", sl_adj)

        if getattr(self, "strategy", None):
            self.strategy.trend_tp_boost = tp_adj
            self.strategy.trend_sl_relax = sl_adj

        self._strategy_cfg_set("trend_tp_boost", tp_adj)
        self._strategy_cfg_set("trend_sl_relax", sl_adj)
        raw = self._strategy_raw_section()
        raw["trend_tp_boost"] = tp_adj
        raw["trend_sl_relax"] = sl_adj
        self.log.info("trend boosts -> tp=%+.2f sl=%+.2f", tp_adj, sl_adj)

    def set_range_tighten(self, tp_tighten: float, sl_tighten: float) -> None:
        """Tune range regime tightening factors for TP and SL."""

        tp_adj = float(tp_tighten)
        sl_adj = float(sl_tighten)
        if not -5.0 <= tp_adj <= 5.0:
            raise ValueError("range TP tighten must be between -5.0 and 5.0")
        if not -5.0 <= sl_adj <= 5.0:
            raise ValueError("range SL tighten must be between -5.0 and 5.0")

        self._set_strategy_setting("range_tp_tighten", tp_adj)
        self._set_strategy_setting("range_sl_tighten", sl_adj)

        if getattr(self, "strategy", None):
            self.strategy.range_tp_tighten = tp_adj
            self.strategy.range_sl_tighten = sl_adj

        self._strategy_cfg_set("range_tp_tighten", tp_adj)
        self._strategy_cfg_set("range_sl_tighten", sl_adj)
        raw = self._strategy_raw_section()
        raw["range_tp_tighten"] = tp_adj
        raw["range_sl_tighten"] = sl_adj
        self.log.info("range tighten -> tp=%+.2f sl=%+.2f", tp_adj, sl_adj)

    # ---------------- live-mode wiring ----------------
    def _create_kite_from_settings(self):
        """Create a KiteConnect session from settings.

        Raises:
            RuntimeError: If the broker SDK is missing or credentials are not
                provided.
        """
        if KiteConnect is None:
            msg = "kiteconnect not installed; cannot enter live."
            self.log.warning(msg)
            raise RuntimeError(msg)

        api_key = getattr(settings.zerodha, "api_key", None)
        access_token = getattr(settings.zerodha, "access_token", None)
        if not api_key or not access_token:
            msg = "Zerodha credentials missing; cannot enter live."
            self.log.error(msg)
            raise RuntimeError(msg)

        try:
            k = KiteConnect(api_key=str(api_key))
            k.set_access_token(str(access_token))
            return k
        except Exception as e:
            msg = f"Failed to create KiteConnect session: {e}"
            self.log.warning(msg)
            raise RuntimeError(msg)

    def set_live_mode(self, val: bool) -> None:
        """
        Flip live mode and (if enabling) ensure a live broker session is present, then rewire executor and data source safely.
        """
        try:
            setattr(settings, "enable_live_trading", bool(val))
        except Exception:
            self.log.debug("Unable to set enable_live_trading flag", exc_info=True)

        if not val:
            self.log.info("🔒 Dry mode — paper trading only.")
            return

        # Enabling LIVE: ensure we have a Kite session
        if not self.kite:
            try:
                self.kite = self._create_kite_from_settings()
            except Exception as e:
                msg = f"Broker init failed: {e}"
                self.log.error(msg)
                try:
                    if self.telegram:
                        self.telegram.send_message(msg)
                except Exception:
                    self.log.warning(
                        "Failed to notify Telegram about broker init failure",
                        exc_info=True,
                    )
                # Re-raise so callers know live mode failed
                raise

        # Rewire executor
        try:
            if hasattr(self.executor, "set_live_broker"):
                self.executor.set_live_broker(self.kite)
            elif hasattr(self.executor, "set_kite"):
                self.executor.set_kite(self.kite)
            else:
                self.executor.kite = self.kite  # best-effort
        except Exception as e:
            self.log.warning("Executor rewire failed: %s", e)

        # Rewire or initialize data source
        if self.data_source is not None:
            try:
                if hasattr(self.data_source, "set_kite"):
                    self.data_source.set_kite(self.kite)
                else:
                    setattr(self.data_source, "kite", self.kite)
                self.data_source.connect()
            except Exception as e:
                self.log.warning("Data source connect failed: %s", e)
        elif LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()
                try:
                    self._fetch_spot_ohlc()
                except Exception as e:
                    self.log.debug("Initial OHLC fetch failed: %s", e)
            except Exception as e:
                self.log.warning("Data source init failed: %s", e)
                self.data_source = None

        self.log.info("🔓 Live mode ON — broker session initialized.")

    # ---------------- notify ----------------
    def _notify(self, msg: str) -> None:
        now = time.time()
        last_msg, last_ts = self._last_notification
        # Skip duplicate messages within a short time window
        if msg == last_msg and (now - last_ts) < 300:
            return
        self._last_notification = (msg, now)
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            self.log.debug("Failed to send Telegram notification", exc_info=True)
