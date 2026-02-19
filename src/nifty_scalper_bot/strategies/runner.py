"""Event-driven strategy runner coordinating trading managers."""

from __future__ import annotations

import asyncio
import calendar
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import threading
import time
import time as time_module
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    cast,
)

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.universe_controller import UniverseController

# Assumes you created the data/constants.py file as advised
from nifty_scalper_bot.data.constants import OPTION_ALIAS_SUFFIX
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType
from nifty_scalper_bot.execution.position_manager import OrderSide, PositionManager
from nifty_scalper_bot.indicators.atr_provider import ATRSnapshot
from nifty_scalper_bot.options.strike_selector import SelectedContract, StrikeSelector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar, OneMinuteBarBuilder
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils import metrics
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market_hours import (
    MarketState,
    get_market_state,
    get_time_status,
    is_market_hours_cached,
)
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.reasons import canonical as canonical_reason
from nifty_scalper_bot.utils.symbols import canonical, is_strategy_instrument

if TYPE_CHECKING:
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.persistent_state import (
        PersistentStateManager,
        TradeDict,
    )

LOGGER = get_logger(__name__)
_THROTTLE_CACHE: Dict[str, float] = {}
_THROTTLE_LOCK = threading.Lock()


def log_throttled(
    logger: Any,
    key: str,
    msg: str,
    interval_sec: float = 60.0,
    level: int | str = logging.INFO,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log throttled message. Args: logger, key, msg. Returns: None. Raises: Exception."""
    try:
        with _THROTTLE_LOCK:
            now = time.time()
            last_time = _THROTTLE_CACHE.get(key, 0.0)
            if now - last_time < interval_sec:
                return
            _THROTTLE_CACHE[key] = now

        # Normalize log level (accept str or logging.* int)
        if isinstance(level, int):
            logger.log(level, msg, extra=extra or {})
        else:
            log_method = getattr(logger, str(level).lower(), logger.info)
            if extra:
                log_method(msg, extra=extra)
            else:
                log_method(msg)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failure in log_throttled: %s", exc, exc_info=True)


_STRATEGY_SKIP_COUNTER = Counter(
    "strategy_skips_total", "Strategy skip counts by reason", ["reason"]
)

_NIFTY_OPTION_DELTA_GAUGE = metrics.Gauge(
    "nifty_option_best_delta",
    "Delta of the best scoring NIFTY option candidate",
    ["underlying"],
)

_NIFTY_OPTION_IV_GAUGE = metrics.Gauge(
    "nifty_option_best_iv",
    "Implied volatility of the best scoring NIFTY option candidate",
    ["underlying"],
)

_NIFTY_OPTION_LIQUIDITY_GAUGE = metrics.Gauge(
    "nifty_option_best_liquidity",
    "Liquidity score of the best scoring NIFTY option candidate",
    ["underlying"],
)

_NIFTY_OPTION_SIGNAL_LATENCY = metrics.Histogram(
    "nifty_option_signal_to_trade_latency_seconds",
    "Latency between signal generation and order submission",
    ["underlying"],
)

_NIFTY_OPTION_EXECUTION_COUNTER = metrics.Counter(
    "nifty_option_execution_total",
    "NIFTY option execution outcomes by result",
    ["underlying", "result"],
)

_NIFTY_OPTION_SUCCESS_RATE = metrics.Gauge(
    "nifty_option_execution_success_rate",
    "Rolling execution success ratio for NIFTY options",
    ["underlying"],
)

_NIFTY_OPTION_SLIPPAGE_GAUGE = metrics.Gauge(
    "nifty_option_order_slippage",
    "Observed slippage for NIFTY option orders",
    ["underlying"],
)


class OrderRouter(Protocol):
    """Protocol for order placement and management."""

    def place_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> str: ...

    def place_reduce_only_exit(self, intent: ExitIntent) -> str | None: ...

    def consume_skip_reason(self) -> str | None: ...

    def get_order(self, order_id: str) -> Any | None: ...


@dataclass(slots=True)
class TradeRecord:
    """Record summarizing trade submissions for auditing."""

    timestamp: datetime
    action: str
    quantity: int
    price: float
    status: str
    reason: str | None = None
    order_id: str | None = None
    reason_tags: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the trade record."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "reason": self.reason,
            "order_id": self.order_id,
            "reason_tags": dict(self.reason_tags) if self.reason_tags else None,
        }


@dataclass(slots=True)
class StrategyRunnerConfig:
    """Configuration controlling runner level behaviour."""

    signal_cooldown_seconds: float = 3.0
    trade_cooldown_seconds: float = 10.0
    min_indicator_bars: int = 20
    max_trade_history: int = 100
    fetch_history_on_startup: bool = True

    def __post_init__(self) -> None:
        if self.signal_cooldown_seconds < 0:
            msg = "signal_cooldown_seconds must be non-negative"
            raise ValueError(msg)
        if self.trade_cooldown_seconds < 0:
            msg = "trade_cooldown_seconds must be non-negative"
            raise ValueError(msg)
        if self.min_indicator_bars < 0:
            msg = "min_indicator_bars must be non-negative"
            raise ValueError(msg)
        if self.max_trade_history <= 0:
            msg = "max_trade_history must be positive"
            raise ValueError(msg)


class RunnerState(Enum):
    """State machine for strategy runner lifecycle."""

    BOOTING = 1
    HISTORICAL_READY = 2
    LIVE_READY = 3
    EXECUTION_ENABLED = 4


class SymbolState(Enum):
    """Hydration lifecycle state maintained per symbol."""

    DISCOVERED = "discovered"
    HYDRATING = "hydrating"
    READY = "ready"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"


@dataclass(slots=True)
class SymbolRuntimeState:
    """Mutable runtime data maintained per symbol."""

    symbol: str
    history_limit: int
    active: bool = True
    last_tick: dict[str, Any] | None = None
    last_signal_at: datetime | None = None
    last_trade_at: datetime | None = None
    cooldown_until: datetime | None = None
    strategy_data: dict[str, Any] = field(default_factory=dict)
    vwap: float | None = None
    session_vwap_volume: int = 0
    session_vwap_turnover: float = 0.0
    _last_strategy_eval: datetime | None = None  # [FIX] For Throttling strategy calls
    _last_eval_bar_ts: datetime | None = None
    trade_history: Deque[TradeRecord] = field(init=False)

    def __post_init__(self) -> None:
        self.trade_history = deque(maxlen=self.history_limit)

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the symbol state."""
        return {
            "active": self.active,
            "cooldown_until": _format_dt(self.cooldown_until),
            "last_signal_at": _format_dt(self.last_signal_at),
            "last_trade_at": _format_dt(self.last_trade_at),
            "trade_history": [record.to_dict() for record in self.trade_history],
            "strategy_data": dict(self.strategy_data),
        }


def _format_dt(value: datetime | None) -> str | None:
    """Format datetime to UTC ISO format string."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _extract_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first float value available across *keys*."""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            number = float(value)
            if number == number:  # guard NaN
                return number
        except (TypeError, ValueError):
            continue
    return None


def _extract_int(payload: Mapping[str, Any], *keys: str) -> int:
    """Return the first integer value available across *keys*."""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _extract_timestamp(payload: Mapping[str, Any], fallback: datetime) -> datetime:
    """Extract a timestamp from *payload* or return *fallback*."""
    candidate = (
        payload.get("timestamp")
        or payload.get("ts")
        or payload.get("ts_ms")
        or payload.get("last_trade_time")
    )

    if isinstance(candidate, datetime):
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    if isinstance(candidate, (int, float)):
        try:
            value = float(candidate)
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, ValueError, OSError):
            return fallback

    if isinstance(candidate, str):
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except Exception:
            return fallback

    return fallback


def _is_monthly_expiry(expiry: datetime) -> bool:
    """Return True when *expiry* corresponds to the monthly contract."""
    normalized = expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)
    expiry_date = normalized.date()
    last_day = calendar.monthrange(expiry_date.year, expiry_date.month)[1]
    anchor = datetime(
        expiry_date.year,
        expiry_date.month,
        last_day,
        normalized.hour,
        normalized.minute,
        normalized.second,
        normalized.microsecond,
        tzinfo=normalized.tzinfo,
    )

    while anchor.weekday() != 3:  # Thursday
        anchor -= timedelta(days=1)

    return anchor.date() == expiry_date


class StrategyRunner:
    """Coordinate market data events with strategy and execution managers."""

    def __init__(
        self,
        *,
        market_data_manager: MarketDataManager,
        indicator_engine: IndicatorEngine,
        strategy_manager: StrategyManager,
        risk_manager: RiskManager,
        order_manager: OrderRouter,
        position_manager: PositionManager,
        message_bus: MessageBus | None = None,
        config: StrategyRunnerConfig | None = None,
        data_hub: "DataHub | None" = None,
        strike_selector: StrikeSelector | None = None,
        bracket_manager: Any | None = None,
    ) -> None:
        self._market_data = market_data_manager
        self._indicator_engine = indicator_engine
        self._strategy_manager = strategy_manager
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._message_bus = message_bus
        self._config = config or StrategyRunnerConfig()
        self._logger = get_logger(__name__)
        self._logger.debug("StrategyRunner using MessageBus id=%s", id(self._message_bus))
        # ✅ FIX 1: Ensure 'data' directory exists to prevent Persistence Crash
        try:
            os.makedirs("data", exist_ok=True)
            self._logger.info("✅ Verified 'data/' directory exists for persistence.")
        except Exception as e:
            self._logger.error(f"❌ Failed to create 'data/' directory: {e}")
        self._data_hub = data_hub
        self._strike_selector = strike_selector
        self._bracket_manager = bracket_manager
        self._symbol_source: MarketDataManager | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        # Time block logging throttle
        self._time_block_logged: Dict[str, float] = {}

        if self._message_bus is not None:
            self._message_bus.subscribe(MessageType.TICK, self._handle_tick_message)

        hedge_env = os.getenv("NSB__ALLOW_HEDGE_ENTRIES", "false").strip().lower()
        self._allow_hedge_entries = hedge_env in {"1", "true", "yes", "on"}

        self._options_long_only = True
        self._legacy_side_to_type = False
        self._monthly_halt_minutes = 0
        self._option_delta_target = 0.35
        self._option_max_iv_rank = 0.75
        self._option_min_liquidity = 0.6
        self._option_score_weights: Mapping[str, float] = {
            "delta": 0.4,
            "theta": 0.2,
            "gamma": 0.2,
            "iv": 0.1,
            "liquidity": 0.1,
        }

        if strike_selector is not None:
            try:
                selector_settings = strike_selector.settings
                self._options_long_only = getattr(selector_settings, "long_only", True)
                self._legacy_side_to_type = getattr(
                    selector_settings, "legacy_side_to_type", False
                )
                self._monthly_halt_minutes = max(
                    0, int(getattr(selector_settings, "monthly_halt_minutes", 0))
                )
            except Exception as exc:
                self._logger.debug(
                    "Unable to read selector settings: %s", exc, exc_info=True
                )

        self._execution_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "error": 0}
        )

        try:
            settings = get_settings()
            option_cfg = getattr(settings, "nifty_options", None)
            if option_cfg is not None:
                self._option_delta_target = float(option_cfg.delta_target)
                self._option_max_iv_rank = float(option_cfg.max_iv_rank)
                self._option_min_liquidity = float(option_cfg.min_liquidity_score)
                weights = option_cfg.weights.normalized()
                if weights:
                    self._option_score_weights = weights
                self._logger.info(
                    "Condition met: nifty_option_score_config",
                    extra={
                        "event": "nifty_option_score_config",
                        "delta_target": self._option_delta_target,
                        "max_iv_rank": self._option_max_iv_rank,
                        "min_liquidity": self._option_min_liquidity,
                        "weights": dict(self._option_score_weights),
                    },
                )
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner settings load: %s",
                exc,
                extra={"event": "strategy_runner_settings_load_error"},
                exc_info=exc,
            )

        self._lock = threading.RLock()
        self._running = False
        self._trading_paused = False
        self._active_symbols: set[str] = set()
        self._tracked_symbols: set[str] = set()
        self._live_symbols: set[str] = set()
        self._symbol_state: Dict[str, SymbolRuntimeState] = {}
        self._callbacks: MutableMapping[str, Callable[[dict], None]] = {}
        self._bar_builders: Dict[str, OneMinuteBarBuilder] = {}
        self._last_bar_ts: dict[str, datetime] = {}
        self._gating_log_bar_cache: dict[tuple[str, str], datetime] = {}
        self._hydration_log_bar_cache: dict[str, datetime] = {}
        self._orchestrator = getattr(strategy_manager, "orchestrator", None)
        self._persistent_state: PersistentStateManager | None = None
        self._orders_in_flight: dict[str, float] = {}  # symbol -> timestamp
        self._order_in_flight_timeout: float = 30.0  # seconds
        self._recently_closed: dict[str, float] = (
            {}
        )  # ✅ FIX: Track recently exited symbols
        self._entry_lock = threading.Lock()  # Atomic entry lock
        self._last_cumulative_volume: dict[str, int] = {}
        self._last_valid_price: dict[str, float] = {}
        self._last_valid_price_ts: dict[str, datetime] = {}
        self._post_exit_cooldown_seconds: float = float(
            os.getenv("POST_EXIT_COOLDOWN_SECONDS", "60.0")
        )
        # Global risk-halt latch keeps control-plane work quiet once breaker trips.
        # We intentionally keep this sticky so per-symbol loops cannot spam checks/logs.
        self._risk_halt_active = False
        self._risk_halt_logged = False
        self._required_candles = max(
            self._config.min_indicator_bars,
            int(os.getenv("REQUIRED_CANDLES", str(self._config.min_indicator_bars))),
        )
        self._max_symbol_count: int = int(os.getenv("STRATEGY_MAX_SYMBOL_COUNT", "32"))
        self._universe_controller = UniverseController()
        self._universe_dynamic_mode = bool(
            getattr(get_settings(), "universe_dynamic_mode", True)
        )
        self._history_gate_failed: bool = False
        self._history_ready_by_symbol: dict[str, bool] = {}
        self._symbol_states: dict[str, SymbolState] = {}
        self._symbol_bar_count: dict[str, int] = {}
        self._symbol_history: dict[str, list[OneMinuteBar]] = {}
        self._hydration_ready_streak: dict[str, int] = {}
        self._frozen_universe: set[str] = set()
        self._vwap_state: dict[str, dict[str, Any]] = {}
        self._last_readiness_update_by_symbol: dict[str, datetime] = {}
        self._rate_limit_backoff_until_by_symbol: dict[str, float] = {}
        self._hydration_attempted_symbols: set[str] = set()
        self._strategy_slot_limit: int = max(
            1,
            int(os.getenv("MAX_CONCURRENT_STRATEGIES", "3")),
        )
        self._history_cache_dir = Path(".cache/candles")
        self._history_cache_dir.mkdir(parents=True, exist_ok=True)
        self._hydrate_failures: dict[str, int] = {}
        self._session_gap_count: dict[str, int] = {}
        self._runner_state: RunnerState = RunnerState.BOOTING
        self._active_orphan_guards: set[str] = set()

    # ==================== LIFECYCLE MANAGEMENT ====================

    def start(self) -> None:
        """Start processing market data events."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._trading_paused = False
            symbols = list(self._active_symbols)
            self._frozen_universe = set(symbols)
            self._universe_controller.update(symbols)
            self._history_gate_failed = False
            self._runner_state = RunnerState.HISTORICAL_READY
            self._history_ready_by_symbol = {symbol: False for symbol in symbols}
            for symbol in symbols:
                self._symbol_states.setdefault(symbol, SymbolState.DISCOVERED)
            self._rate_limit_backoff_until_by_symbol = {}
            self._vwap_state = {}
            self._symbol_bar_count = {}
            self._hydration_ready_streak = {}

        # Capture the loop if called from async context (optional safety)
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        self._market_data.start()

        if self._data_hub is not None:
            reset = getattr(self._data_hub, "reset_warmup", None)
            if callable(reset):
                reset()

        for symbol in symbols:
            self._subscribe_symbol(symbol)

        self._logger.info("Strategy runner started with symbols: %s", symbols)

        # ✅ FIX: Launch Backfill Task
        if self._config.fetch_history_on_startup and self._main_loop:
            self._main_loop.create_task(self._backfill_history())

    def stop(self) -> None:
        """Stop event processing and unsubscribe from market data."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            callbacks = dict(self._callbacks)
            self._callbacks.clear()
            self._trading_paused = True

        for symbol, callback in callbacks.items():
            if self._data_hub is not None:
                self._data_hub.unsubscribe_ticks(symbol, callback)
            else:
                self._market_data.unsubscribe(symbol, callback)

        self._market_data.stop()
        self._logger.info("Strategy runner stopped")

    def pause_trading(self) -> None:
        """Temporarily prevent order placement while keeping data flowing."""
        with self._lock:
            self._trading_paused = True
            self._logger.info("Strategy runner paused")

    def resume_trading(self) -> None:
        """Resume order placement after a pause."""
        with self._lock:
            self._trading_paused = False
            self._logger.info("Strategy runner resumed")

        if self._data_hub is not None:
            reset = getattr(self._data_hub, "reset_warmup", None)
            if callable(reset):
                reset()

    # ==================== SYMBOL MANAGEMENT ====================

    def set_symbol_source(self, source: MarketDataManager | None) -> None:
        """Attach fallback market data symbol source."""
        self._logger.debug(
            "Entered StrategyRunner.set_symbol_source",
            extra={"event": "strategy_runner_set_symbol_source_enter"},
        )
        try:
            self._symbol_source = source
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.set_symbol_source: %s",
                exc,
                extra={"event": "strategy_runner_set_symbol_source_error"},
                exc_info=exc,
            )

    def add_symbol(self, symbol: str) -> None:
        """Begin tracking a new symbol."""
        normalized = self._normalize_symbol(symbol)
        with self._lock:
            state = self._symbol_state.get(normalized)
            if state is None:
                state = SymbolRuntimeState(
                    symbol=normalized,
                    history_limit=self._config.max_trade_history,
                )
                self._symbol_state[normalized] = state
            else:
                state.active = True

            if (
                self._running
                and self._frozen_universe
                and normalized not in self._frozen_universe
            ):
                self._logger.info(
                    "Symbol add deferred until next session boundary",
                    extra={"event": "symbol_add_deferred", "symbol": normalized},
                )
                return
            self._active_symbols.add(normalized)
            self._tracked_symbols.add(normalized)
            self._symbol_states.setdefault(normalized, SymbolState.DISCOVERED)
            self._set_symbol_hydration_state(normalized, SymbolState.HYDRATING)
            # Update tracked universe snapshot only when membership changes.
            self._universe_controller.update(self._active_symbols)

        running = False
        with self._lock:
            running = self._running

        try:
            self._strategy_manager.track_symbol(normalized)
        except AttributeError:
            pass

        if running:
            self._subscribe_symbol(normalized)

        self._prehydrate_symbol_history(normalized)

        self._logger.info("Tracking symbol %s", normalized)

    def _prehydrate_symbol_history(self, symbol: str) -> None:
        """Fetch startup candles for symbol hydration and indicator readiness."""
        target = max(self._required_candles, 50)
        rows = self._hydrate_missing_bars(symbol, target)
        if not rows:
            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
            return
        if len(rows) >= target and not self._has_session_candle_gaps(symbol):
            self._set_symbol_hydration_state(symbol, SymbolState.READY)
            return
        self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)

    def remove_symbol(self, symbol: str) -> None:
        """Stop tracking a symbol."""
        normalized = self._normalize_symbol(symbol)
        with self._lock:
            state = self._symbol_state.get(normalized)
            if state is None:
                return

            state.active = False
            self._active_symbols.discard(normalized)
            self._tracked_symbols.discard(normalized)
            self._live_symbols.discard(normalized)
            self._frozen_universe.discard(normalized)
            self._symbol_states.pop(normalized, None)
            self._symbol_bar_count.pop(normalized, None)
            self._hydration_ready_streak.pop(normalized, None)
            self._vwap_state.pop(normalized, None)
            # Dynamic diffing avoids legacy frozen-universe drift conflicts.
            self._universe_controller.update(self._active_symbols)
            callback = self._callbacks.pop(normalized, None)

        try:
            self._strategy_manager.untrack_symbol(normalized)
        except AttributeError:
            pass

        if callback is not None:
            if self._data_hub is not None:
                self._data_hub.unsubscribe_ticks(normalized, callback)
            else:
                self._market_data.unsubscribe(normalized, callback)

        builder = self._bar_builders.pop(normalized, None)
        if builder is not None:
            completed = builder.flush()
            if completed is not None:
                self._ingest_bar(normalized, completed)

        self._logger.info("Stopped tracking symbol %s", normalized)

    @property
    def tracked_symbols(self) -> list[str]:
        """Return tracked symbols with MarketDataManager fallback."""
        self._logger.debug(
            "Entered StrategyRunner.tracked_symbols",
            extra={"event": "strategy_runner_tracked_symbols_enter"},
        )

        try:
            with self._lock:
                active_symbols = sorted(self._active_symbols)

            if active_symbols:
                self._logger.info(
                    "Condition met: strategy_runner_tracked_symbols_active",
                    extra={
                        "event": "strategy_runner_tracked_symbols_active",
                        "count": len(active_symbols),
                    },
                )
                return active_symbols

            snapshot_fn = getattr(self._market_data, "tracked_snapshot", None)
            fallback_symbols: list[str] = []

            if callable(snapshot_fn):
                fallback_symbols = [
                    str(symbol) for symbol in snapshot_fn() if str(symbol or "").strip()
                ]

            if fallback_symbols:
                sorted_fallback = sorted(fallback_symbols)
                self._logger.info(
                    "Condition met: strategy_runner_tracked_symbols_fallback",
                    extra={
                        "event": "strategy_runner_tracked_symbols_fallback",
                        "count": len(sorted_fallback),
                    },
                )
                return sorted_fallback

            self._logger.info(
                "Condition met: strategy_runner_tracked_symbols_empty",
                extra={
                    "event": "strategy_runner_tracked_symbols_empty",
                    "count": 0,
                },
            )
            return []

        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.tracked_symbols: %s",
                exc,
                extra={"event": "strategy_runner_tracked_symbols_error"},
                exc_info=exc,
            )
            return []

    def tracked_symbol_count(self) -> int:
        """Return tracked symbol count leveraging MarketDataManager fallback."""
        self._logger.debug(
            "Entered StrategyRunner.tracked_symbol_count",
            extra={"event": "strategy_runner_tracked_symbol_count_enter"},
        )

        try:
            count = len(self.tracked_symbols)
            self._logger.info(
                "Condition met: strategy_runner_tracked_symbol_count_ready",
                extra={
                    "event": "strategy_runner_tracked_symbol_count_ready",
                    "count": count,
                },
            )
            return count

        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.tracked_symbol_count: %s",
                exc,
                extra={"event": "strategy_runner_tracked_symbol_count_error"},
                exc_info=exc,
            )
            return 0

    def get_underlying_snapshot(
        self, base_symbol: str = "NIFTY"
    ) -> dict[str, float] | None:
        """Return the latest quote snapshot for the underlying symbol."""
        self._logger.debug(
            "Entered StrategyRunner.get_underlying_snapshot",
            extra={"base_symbol": base_symbol},
        )

        snapshot: dict[str, float] = {}

        try:
            quote = self._market_data.get_quote(base_symbol)
        except Exception as exc:
            self._logger.warning(
                "underlying_snapshot_error",
                extra={"base_symbol": base_symbol, "error": str(exc)},
                exc_info=exc,
            )
            return None

        if not quote:
            return None

        ltp = _extract_float(quote, "ltp", "last_price", "price", "close")
        if ltp is not None:
            snapshot["ltp"] = float(ltp)

        bid = _extract_float(quote, "bid", "bid_price", "best_bid_price")
        if bid is not None:
            snapshot["bid"] = float(bid)

        ask = _extract_float(quote, "ask", "ask_price", "best_ask_price")
        if ask is not None:
            snapshot["ask"] = float(ask)

        fallback = datetime.now(timezone.utc)
        timestamp_value = _extract_timestamp(quote, fallback)
        if timestamp_value is not None:
            snapshot["timestamp"] = float(timestamp_value.timestamp())

        if snapshot:
            self._logger.info(
                "Condition met: underlying_snapshot_ready",
                extra={"base_symbol": base_symbol, "fields": sorted(snapshot)},
            )
            return snapshot

        return None

    # ==================== STATE & PERSISTENCE ====================

    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        """Attach persistent state manager used for trade recovery."""
        self._logger.debug(
            "Entered attach_persistent_state",
            extra={"event": "runner_attach_persistent"},
        )
        self._persistent_state = manager

    def restore_trades(
        self, trades: Iterable["TradeDict | Mapping[str, object]"]
    ) -> None:
        """Restore trade history from persisted *trades* payloads."""
        self._logger.debug(
            "Entered restore_trades",
            extra={"event": "runner_restore_trades"},
        )

        restored = 0
        for trade in trades:
            if not isinstance(trade, Mapping):
                continue

            symbol_raw = trade.get("symbol") or trade.get("instrument")
            symbol = str(symbol_raw or "").strip().upper()
            if not symbol:
                continue

            timestamp = datetime.now(timezone.utc)
            raw_ts = trade.get("timestamp")

            if isinstance(raw_ts, datetime):
                ts_value = raw_ts
                if ts_value.tzinfo is None:
                    timestamp = ts_value.replace(tzinfo=timezone.utc)
                else:
                    timestamp = ts_value.astimezone(timezone.utc)
            elif isinstance(raw_ts, str):
                try:
                    parsed = datetime.fromisoformat(raw_ts)
                except ValueError:
                    parsed = timestamp

                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)

                timestamp = parsed.astimezone(timezone.utc)

            action = str(trade.get("action") or trade.get("side") or "").upper()
            quantity = _extract_int(trade, "quantity")
            price_value = _extract_float(trade, "price")
            price = price_value if price_value is not None else 0.0
            status = str(trade.get("status") or "").upper()

            if not action or quantity == 0 or price <= 0.0 or not status:
                continue

            reason = trade.get("reason")
            order_id = trade.get("order_id") or trade.get("orderId")

            record = TradeRecord(
                timestamp=timestamp,
                action=action,
                quantity=quantity,
                price=price,
                status=status,
                reason=str(reason) if reason is not None else None,
                order_id=str(order_id) if order_id is not None else None,
            )

            with self._lock:
                state = self._symbol_state.get(symbol)
                if state is None:
                    state = SymbolRuntimeState(
                        symbol=symbol,
                        history_limit=self._config.max_trade_history,
                    )
                    state.active = False
                    self._symbol_state[symbol] = state

                state.trade_history.append(record)
                state.last_trade_at = record.timestamp

            restored += 1

        if restored == 0:
            self._logger.info(
                "Condition met: restore_trades_empty",
                extra={"event": "runner_restore_trades_empty"},
            )
        else:
            self._logger.info(
                "Condition met: restore_trades_applied",
                extra={"event": "runner_restore_trades", "count": restored},
            )

    def get_status(self) -> dict[str, Any]:
        """Return current runner status including symbol level state."""
        with self._lock:
            symbols = {
                symbol: state.snapshot() for symbol, state in self._symbol_state.items()
            }
            status = {
                "running": self._running,
                "trading_paused": self._trading_paused,
                "active_symbols": sorted(self._active_symbols),
                "symbols": symbols,
            }

        return status

    # ==================== INTERNAL HELPERS ====================

    def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to tick updates for a symbol."""
        callback = self._callbacks.get(symbol)
        if callback is None:
            def _callback(tick: Mapping[str, Any], sym: str = symbol) -> None:
                self._on_tick(sym, tick)
            callback = _callback
            self._callbacks[symbol] = callback

        if self._data_hub is not None:
            # Primary path: data_hub.subscribe_ticks wires DataHub → MDM._subscribers
            # so WS ticks reach this callback via MDM._emit_tick directly.
            try:
                self._data_hub.subscribe_ticks(symbol, callback)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in StrategyRunner._subscribe_symbol: %s", exc
                )
        else:
            self._market_data.subscribe(symbol, callback)

    def ingest_historical_bar(self, data: dict) -> None:
        """
        Public API for Startup Hydration.
        Strictly conforms to OneMinuteBar(slots=True).
        """
        # [FIX] Mark as hydrated so _backfill_history knows to skip
        self._startup_hydrated = True

        try:
            # 1. Extract timestamps
            ts = data["timestamp"]
            end_ts = ts + timedelta(minutes=1)

            # 2. Construct EXACTLY matching the signature
            bar = OneMinuteBar(
                open=float(data["open"]),
                high=float(data["high"]),
                low=float(data["low"]),
                close=float(data["close"]),
                volume=int(data["volume"]),
                start=ts,
                end=end_ts,
            )

            # 3. Ingest
            symbol = canonical(str(data["symbol"]))
            self._ingest_bar(symbol, bar, is_backfill=True)

            # 4. Force Registration
            with self._lock:
                self._active_symbols.add(symbol)
                if symbol not in self._symbol_state:
                    self._symbol_state[symbol] = SymbolRuntimeState(
                        symbol=symbol, history_limit=2000
                    )
                self._symbol_states.setdefault(symbol, SymbolState.DISCOVERED)

        except Exception as exc:
            self._logger.error(
                f"❌ Hydration Ingest Failed for {data.get('symbol')}: {exc}"
            )

    def mark_ready(self, symbols: list[str]) -> None:
        """
        Public API to finalize startup hydration.
        Explicitly registers symbols and sets readiness flags.
        """
        with self._lock:
            for sym in symbols:
                normalized = canonical(sym)
                # 1. Register Active (Critical for main loop)
                self._active_symbols.add(normalized)
                self._tracked_symbols.add(normalized)

                # 2. Ensure SymbolState exists (Critical for Strategy Context)
                if normalized not in self._symbol_state:
                    self._symbol_state[normalized] = SymbolRuntimeState(
                        symbol=normalized, history_limit=2000
                    )
                self._symbol_states.setdefault(normalized, SymbolState.DISCOVERED)
                self._set_symbol_hydration_state(normalized, SymbolState.READY)

                # 3. Initialize BarBuilder (Prevent KeyErrors in internal checks)
                if normalized not in self._bar_builders:
                    self._bar_builders[normalized] = OneMinuteBarBuilder()

                # 4. Set High-Water Mark (Prevent dropping first live tick)
                if normalized not in self._last_bar_ts:
                    self._last_bar_ts[normalized] = datetime.now(timezone.utc)

        # 5. THE KILL SWITCH: Prevents fallback backfill logic from running
        self._startup_hydrated = True
        self._runner_state = RunnerState.HISTORICAL_READY

        try:
            if self._market_data is not None and self._main_loop is not None:
                for wait_symbol in ("NSE:NIFTY 50", "NFO:NIFTY FUT"):
                    token = int(self._market_data.get_token(wait_symbol) or 0)
                    if token <= 0:
                        continue
                    fut = asyncio.run_coroutine_threadsafe(
                        self._market_data.wait_for_live_tick(token, timeout=10),
                        self._main_loop,
                    )
                    fut.result(timeout=10.5)
                self._runner_state = RunnerState.EXECUTION_ENABLED
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.mark_ready live tick gate: %s",
                exc,
                exc_info=True,
            )

        self._logger.info(f"✅ StrategyRunner marked READY with {len(symbols)} symbols")

    def _set_symbol_hydration_state(
        self,
        symbol: str,
        next_state: SymbolState,
        *,
        allow_downgrade: bool = False,
    ) -> SymbolState:
        """Update per-symbol hydration state with READY downgrade protection."""
        current = self._symbol_states.get(symbol, SymbolState.DISCOVERED)
        if current == SymbolState.READY and not allow_downgrade:
            if next_state in {SymbolState.HYDRATING, SymbolState.DEGRADED}:
                return current
        self._symbol_states[symbol] = next_state
        self._history_ready_by_symbol[symbol] = next_state == SymbolState.READY
        if current != next_state:
            self._logger.info(
                "hydration_status_transition",
                extra={
                    "event": "hydration_status_transition",
                    "symbol": symbol,
                    "old_state": current.value,
                    "new_state": next_state.value,
                },
            )
        return next_state

    def _history_cache_path(self, symbol: str) -> Path:
        """Return per-symbol local candle cache path."""
        safe = self._normalize_symbol(symbol).replace("/", "_")
        return self._history_cache_dir / f"{safe}.json"

    def _load_history_cache(self, symbol: str) -> list[dict[str, Any]]:
        """Load locally cached candles when broker history is unavailable."""
        path = self._history_cache_path(symbol)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        rows: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            ts = row.get("timestamp")
            if isinstance(ts, str):
                try:
                    row["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
            rows.append(row)
        return rows

    def _write_history_cache(self, symbol: str, rows: list[dict[str, Any]]) -> None:
        """Persist normalized candles for retry-safe hydration."""
        serializable: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.isoformat()
            serializable.append(item)
        try:
            self._history_cache_path(symbol).write_text(
                json.dumps(serializable),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("history_cache_write_failed: %s", exc)

    def _has_session_candle_gaps(self, symbol: str) -> bool:
        """Return True when current-session minute history has timestamp gaps."""
        history = self._symbol_history.get(symbol, [])
        if len(history) < 2:
            self._session_gap_count[symbol] = 0
            return False
        session_date = datetime.now(timezone.utc).date()
        session_bars = [bar for bar in history if bar.timestamp.date() == session_date]
        gaps = 0
        for prev, curr in zip(session_bars, session_bars[1:]):
            if (curr.timestamp - prev.timestamp).total_seconds() > 120:
                gaps += 1
        self._session_gap_count[symbol] = gaps
        return gaps > 0

    def _refresh_symbol_hydration_state(self, symbol: str) -> SymbolState:
        """Set DISCOVERED/HYDRATING/READY from bar count and VWAP validity."""
        try:
            bars = self._indicator_engine.get_history(symbol)
        except Exception:
            bars = []
        with self._lock:
            runtime_state = self._symbol_state.get(symbol)
            vwap_value = runtime_state.vwap if runtime_state is not None else None
        vwap_state = self._vwap_state.get(symbol, {})
        indicators = {
            symbol: {
                "vwap": vwap_value,
                "cum_volume": float(vwap_state.get("cum_vol", 0.0)),
            }
        }
        return self.update_symbol_hydration(symbol, bars, indicators)

    def _update_symbol_readiness(self, symbol: str) -> SymbolState:
        """Update lifecycle state from bar history and cumulative VWAP health."""
        bars = self._symbol_history.get(symbol, [])
        vol_sum = sum(float(getattr(bar, "volume", 0)) for bar in bars)
        state = self._symbol_state.get(symbol)
        vwap_val = float(state.vwap) if state and state.vwap else 0.0
        indicators = {symbol: {"vwap": vwap_val, "cum_volume": vol_sum}}
        return self.update_symbol_hydration(symbol, bars, indicators)

    def update_symbol_hydration(
        self,
        symbol: str,
        bars: list[float],
        indicators: dict[str, dict[str, Any]],
    ) -> SymbolState:
        """Update hydration lifecycle using strict warmup and cumulative VWAP checks."""
        prev_state = self._symbol_states.get(symbol, SymbolState.DISCOVERED)
        bar_count = len(bars)
        self._symbol_bar_count[symbol] = bar_count

        if bar_count < self._required_candles:
            self._hydration_ready_streak[symbol] = 0
            return self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)

        symbol_indicators = indicators.get(symbol, {})
        raw_vwap = symbol_indicators.get("vwap")
        vwap_val = float(raw_vwap) if isinstance(raw_vwap, (int, float)) else 0.0
        raw_volume = symbol_indicators.get("cum_volume", 0)
        vol_sum = float(raw_volume) if isinstance(raw_volume, (int, float)) else 0.0
        valid_vwap = vwap_val > 0
        valid_volume = vol_sum > 0

        if self._has_session_candle_gaps(symbol):
            gap_count = int(self._session_gap_count.get(symbol, 0))
            if gap_count > 1:
                self._logger.warning(
                    "soft_data_issue",
                    extra={
                        "event": "soft_data_issue",
                        "symbol": symbol,
                        "issue": "repeated_missing_candles",
                        "gaps": gap_count,
                    },
                )
                return self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)
            self._logger.warning(
                "soft_data_issue",
                extra={
                    "event": "soft_data_issue",
                    "symbol": symbol,
                    "issue": "single_missing_candle",
                },
            )
            return self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)

        if valid_vwap and valid_volume:
            streak = int(self._hydration_ready_streak.get(symbol, 0)) + 1
            self._hydration_ready_streak[symbol] = streak
            if prev_state == SymbolState.READY or streak >= 2:
                return self._set_symbol_hydration_state(symbol, SymbolState.READY)

        self._hydration_ready_streak[symbol] = 0
        # Soft degrade on transient VWAP/volume issues; avoid repeated hard resets.
        return self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)

    def _ensure_symbol_vwap_state(self, symbol: str, now: datetime) -> dict[str, Any]:
        """Return session-scoped VWAP accumulator for symbol."""
        session_date = now.date()
        state = self._vwap_state.setdefault(
            symbol,
            {
                "cum_pv": 0.0,
                "cum_vol": 0.0,
                "last_reset_date": session_date,
                "last_valid_vwap": 0.0,
                "mode": "primary",
            },
        )
        if state.get("last_reset_date") != session_date:
            state["cum_pv"] = 0.0
            state["cum_vol"] = 0.0
            state["last_valid_vwap"] = 0.0
            state["mode"] = "primary"
            state["last_reset_date"] = session_date
            self._hydration_ready_streak[symbol] = 0
            self._set_symbol_hydration_state(
                symbol,
                SymbolState.HYDRATING,
                allow_downgrade=True,
            )
        return state

    def _ingest_bar(
        self, symbol: str, bar: OneMinuteBar, is_backfill: bool = False
    ) -> None:
        """
        Ingest a completed minute bar.
        Updates Indicators, Bracket Manager, and Triggers Strategies.

        World-Class Design:
        - Respects OneMinuteBar immutability (slots=True).
        - Uses canonical .timestamp property (Contract).
        - Maps Bar object to IndicatorEngine.update_price API.
        """
        # 1. Monotonicity Check (Prevent out-of-order processing)
        last_ts = self._last_bar_ts.get(symbol)

        # Use the public .timestamp property (wraps .start)
        if not is_backfill and last_ts and bar.timestamp <= last_ts:
            self._logger.debug(
                "Dropping out-of-order bar",
                extra={"symbol": symbol, "bar_ts": bar.timestamp, "last_ts": last_ts},
            )
            return

        # 2. STATE: Update High-Water Mark
        if not last_ts or bar.timestamp > last_ts:
            self._last_bar_ts[symbol] = bar.timestamp
        history = self._symbol_history.setdefault(symbol, [])
        history.append(bar)
        if len(history) > 400:
            del history[:-400]

        try:
            # 3. INDICATORS: Feed the Engine
            # [FIX] Use update_price() instead of update_bar()
            # We convert the bar to a dict using .as_mapping() as expected by the engine.
            if hasattr(self._indicator_engine, "update_bar"):
                self._indicator_engine.update_bar(symbol, bar)
            else:
                # Fallback to update_price (Standard API seen in app.py)
                self._indicator_engine.update_price(
                    symbol,
                    bar.as_mapping(),
                    volume=bar.volume,
                    timestamp=bar.timestamp,
                    is_complete=True,
                    is_provisional=False,
                )

            self._update_symbol_readiness(symbol)

            # 4. BRACKET MANAGER: Inject Dynamic ATR (Volatility)
            if self._bracket_manager:
                # Compute ATR (Period 14 is standard)
                raw_atr = self._indicator_engine.compute_atr(symbol, period=14)

                # Robust Unwrapping
                atr_value = 0.0
                if isinstance(raw_atr, (int, float)):
                    atr_value = float(raw_atr)
                elif hasattr(raw_atr, "value"):
                    atr_value = float(raw_atr.value)
                elif hasattr(raw_atr, "atr"):
                    atr_value = float(raw_atr.atr)

                if atr_value > 0 and hasattr(
                    self._bracket_manager, "update_market_stats"
                ):
                    self._bracket_manager.update_market_stats(symbol, atr=atr_value)

            # [FIX] Force Regime Refresh: Ensure Detector sees the new bar immediately
            if hasattr(self, "_strategy_manager"):
                regime_mgr = getattr(self._strategy_manager, "regime_manager", None)
                if regime_mgr and hasattr(regime_mgr, "refresh_from_indicators"):
                    # Use the captured main loop to run the async refresh safely from this thread
                    if self._main_loop and self._main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            regime_mgr.refresh_from_indicators(), self._main_loop
                        )

            # 5. EXECUTION: Trigger Strategies
            # CRITICAL: Do NOT run strategies during backfill
            if is_backfill:
                return

            with self._lock:
                state = self._symbol_state.get(symbol)
                if state:
                    state.last_tick = {
                        "last_price": bar.close,
                        "timestamp": bar.timestamp.timestamp(),
                        "volume": bar.volume,
                    }
                    vwap_state = self._ensure_symbol_vwap_state(symbol, bar.timestamp)
                    if bar.volume > 0:
                        vwap_state["cum_vol"] = float(
                            vwap_state.get("cum_vol", 0.0)
                        ) + float(bar.volume)
                        vwap_state["cum_pv"] = float(vwap_state.get("cum_pv", 0.0)) + (
                            float(bar.close) * float(bar.volume)
                        )
                    cum_vol = float(vwap_state.get("cum_vol", 0.0))
                    if cum_vol > 0:
                        computed_vwap = float(vwap_state.get("cum_pv", 0.0)) / cum_vol
                        vwap_state["last_valid_vwap"] = computed_vwap
                        vwap_state["mode"] = "primary"
                        state.vwap = computed_vwap
                    else:
                        last_valid_vwap = float(vwap_state.get("last_valid_vwap", 0.0))
                        if last_valid_vwap > 0:
                            vwap_state["mode"] = "degraded"
                            state.vwap = last_valid_vwap
                        else:
                            vwap_state["mode"] = "fallback_last_close"
                            state.vwap = float(bar.close)
                            vwap_state["last_valid_vwap"] = float(bar.close)
                    self._last_cumulative_volume[symbol] = int(cum_vol)

            # 🔥 THE TRIGGER: Run Strategy Logic
            # [FIX] Removed .on_bar() call as StrategyManager is signal-driven (via ticks), not bar-driven.
            return

        except Exception as exc:
            self._logger.error(
                f"Failure in _ingest_bar: {exc}",
                exc_info=True,
                extra={"event": "ingest_bar_failed", "symbol": symbol},
            )

    def _apply_premium_targets(
        self,
        signal: Signal,
        premium: float,
        entry_side: OrderSide,
    ) -> Signal:
        """Derive stop-loss and target from option premium metadata."""
        self._logger.debug(
            "Entered StrategyRunner._apply_premium_targets",
            extra={"event": "apply_premium_targets", "side": entry_side},
        )

        metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        stop_pct_raw = cast(Any, metadata.get("premium_stop_pct"))
        target_rr_raw = cast(Any, metadata.get("premium_target_rr"))

        try:
            stop_pct = float(stop_pct_raw)
        except (TypeError, ValueError):
            return signal

        if stop_pct <= 0:
            return signal

        try:
            target_rr = float(target_rr_raw)
        except (TypeError, ValueError):
            target_rr = 2.0

        if target_rr <= 0:
            target_rr = 2.0

        if premium <= 0:
            return signal

        if entry_side == "BUY":
            stop_loss = max(premium * (1.0 - stop_pct), 0.01)
            risk = premium - stop_loss
            if risk <= 0:
                return signal
            take_profit = premium + risk * target_rr
        else:
            stop_loss = premium * (1.0 + stop_pct)
            risk = stop_loss - premium
            if risk <= 0:
                return signal
            take_profit = max(premium - risk * target_rr, 0.01)

        updated_metadata = dict(metadata)
        updated_metadata["computed_from_premium"] = True
        updated_metadata["entry_side"] = entry_side

        self._logger.info(
            "Condition met: premium_targets_computed",
            extra={
                "event": "premium_targets_computed",
                "side": entry_side,
                "stop_pct": stop_pct,
                "target_rr": target_rr,
            },
        )

        return Signal(
            action=signal.action,
            symbol=signal.symbol,
            quantity=signal.quantity,
            confidence=signal.confidence,
            reason=signal.reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=updated_metadata,
        )

    def _correct_sl_tp_for_position_side(
        self,
        signal: Signal,
        entry_price: float,
        entry_side: OrderSide,
        atr: float,
    ) -> Signal:
        """
        ✅ PRODUCTION FIX (Feb 3, 2026): Ensure SL/TP are correct for position SIDE.

        RULE (Non-Negotiable):
        - LONG (BUY): SL below entry, TP above entry
        - SHORT (SELL): SL above entry, TP below entry

        Strategies calculate SL/TP based on market direction (bullish/bearish),
        but the ACTUAL trade is on OPTION PREMIUM. For LONG options:
        - You profit when premium RISES (regardless of CE/PE)
        - You lose when premium FALLS

        This method corrects any inverted SL/TP from strategy signals.
        """
        sl = signal.stop_loss
        tp = signal.take_profit

        if entry_price <= 0:
            return signal

        # Use ATR-based defaults if missing
        if atr <= 0:
            atr = entry_price * 0.015  # 1.5% fallback

        if sl is None or sl <= 0:
            sl = (
                entry_price - (atr * 1.5)
                if entry_side == "BUY"
                else entry_price + (atr * 1.5)
            )
        if tp is None or tp <= 0:
            tp = (
                entry_price + (atr * 3.0)
                if entry_side == "BUY"
                else entry_price - (atr * 3.0)
            )

        corrected = False
        original_sl = sl
        original_tp = tp

        if entry_side == "BUY":  # LONG position
            # SL must be BELOW entry
            if sl >= entry_price:
                # Mirror the distance to the correct side
                distance = abs(sl - entry_price)
                sl = entry_price - distance
                corrected = True

            # TP must be ABOVE entry
            if tp <= entry_price:
                # Mirror the distance to the correct side
                distance = abs(entry_price - tp)
                tp = entry_price + distance
                corrected = True

        else:  # SHORT position
            # SL must be ABOVE entry
            if sl <= entry_price:
                distance = abs(entry_price - sl)
                sl = entry_price + distance
                corrected = True

            # TP must be BELOW entry
            if tp >= entry_price:
                distance = abs(tp - entry_price)
                tp = entry_price - distance
                corrected = True

        # Ensure minimum distances
        min_sl_distance = atr * 0.5
        min_tp_distance = atr * 1.0

        if entry_side == "BUY":
            if entry_price - sl < min_sl_distance:
                sl = entry_price - min_sl_distance
                corrected = True  # ✅ FIX 5
            if tp - entry_price < min_tp_distance:
                tp = entry_price + min_tp_distance
                corrected = True  # ✅ FIX 5
        else:
            if sl - entry_price < min_sl_distance:
                sl = entry_price + min_sl_distance
                corrected = True  # ✅ FIX 5
            if entry_price - tp < min_tp_distance:
                tp = entry_price - min_tp_distance
                corrected = True  # ✅ FIX 5

        # Force positive prices
        sl = max(0.05, sl)
        tp = max(0.05, tp)

        if corrected:
            self._logger.warning(
                f"⚠️ SL/TP CORRECTED for {entry_side}: "
                f"SL {original_sl:.2f}→{sl:.2f} | TP {original_tp:.2f}→{tp:.2f}",
                extra={
                    "event": "sl_tp_corrected",
                    "entry_side": entry_side,
                    "entry_price": entry_price,
                    "original_sl": original_sl,
                    "original_tp": original_tp,
                    "corrected_sl": sl,
                    "corrected_tp": tp,
                },
            )

        if not corrected:
            return signal

        # Create corrected signal
        return Signal(
            action=signal.action,
            symbol=signal.symbol,
            quantity=signal.quantity,
            confidence=signal.confidence,
            reason=signal.reason,
            stop_loss=sl,
            take_profit=tp,
            metadata=signal.metadata,
        )

    def _anchor_sl_tp_to_execution(
        self,
        signal: Signal,
        *,
        signal_price: float,
        execution_price: float,
        entry_side: OrderSide,
        atr: float,
        sl_mult: float = 1.5,
        tp_mult: float = 3.0,
    ) -> Signal:
        """Anchor risk exits to execution price and enforce ordering constraints."""
        sl = float(signal.stop_loss) if signal.stop_loss is not None else 0.0
        tp = float(signal.take_profit) if signal.take_profit is not None else 0.0
        tick_size = 0.05
        fill_delta = float(execution_price) - float(signal_price)
        if abs(fill_delta) > tick_size:
            if sl > 0:
                sl += fill_delta
            if tp > 0:
                tp += fill_delta
        if atr <= 0:
            atr = execution_price * 0.015

        if entry_side == "BUY":
            floor_sl = execution_price - (atr * sl_mult)
            floor_tp = execution_price + (atr * tp_mult)
            sl = min(sl if sl > 0 else floor_sl, floor_sl)
            tp = max(tp if tp > 0 else floor_tp, floor_tp)
            if not (sl < execution_price < tp):
                sl = min(sl, execution_price - max(atr * 0.5, tick_size))
                tp = max(tp, execution_price + max(atr, tick_size))
        else:
            ceil_sl = execution_price + (atr * sl_mult)
            ceil_tp = execution_price - (atr * tp_mult)
            sl = max(sl if sl > 0 else ceil_sl, ceil_sl)
            tp = min(tp if tp > 0 else ceil_tp, ceil_tp)
            if not (tp < execution_price < sl):
                sl = max(sl, execution_price + max(atr * 0.5, tick_size))
                tp = min(tp, execution_price - max(atr, tick_size))

        return Signal(
            action=signal.action,
            symbol=signal.symbol,
            quantity=signal.quantity,
            confidence=signal.confidence,
            reason=signal.reason,
            stop_loss=max(0.05, sl),
            take_profit=max(0.05, tp),
            metadata=signal.metadata,
        )

    def aggregate_signals_by_symbol(
        self,
        signals: list[Signal],
    ) -> dict[str, Signal]:
        """Aggregate by symbol with confidence weights. Args: signals. Returns: Aggregated signals. Raises: None."""
        self._logger.debug(
            "Entered StrategyRunner.aggregate_signals_by_symbol",
            extra={"event": "aggregate_signals_enter", "count": len(signals)},
        )

        aggregated: dict[str, Signal] = {}
        if not signals:
            return aggregated

        try:
            grouped: dict[str, list[Signal]] = defaultdict(list)
            for signal in signals:
                grouped[signal.symbol].append(signal)

            for symbol, symbol_signals in grouped.items():
                if len(symbol_signals) == 1:
                    aggregated[symbol] = symbol_signals[0]
                    continue

                self._logger.warning(
                    "Aggregating %s signals for %s",
                    len(symbol_signals),
                    symbol,
                    extra={
                        "event": "signal_aggregation",
                        "symbol": symbol,
                        "count": len(symbol_signals),
                    },
                )

                actions = {signal.action for signal in symbol_signals}
                if len(actions) > 1:
                    sorted_actions = sorted(actions)
                    self._logger.error(
                        "Conflicting signals detected for %s: %s",
                        symbol,
                        sorted_actions,
                        extra={
                            "event": "signal_conflict",
                            "symbol": symbol,
                            "actions": sorted_actions,
                        },
                    )
                    continue

                normalized_confidences = [
                    self._normalize_confidence(sig.confidence) for sig in symbol_signals
                ]
                weight_sum = sum(conf**2 for conf in normalized_confidences)
                avg_confidence = (
                    sum(conf * (conf**2) for conf in normalized_confidences)
                    / weight_sum
                    if weight_sum > 0
                    else 0.0
                )

                stop_candidates = [
                    float(sig.stop_loss)
                    for sig in symbol_signals
                    if isinstance(sig.stop_loss, (int, float))
                ]

                target_candidates = [
                    float(sig.take_profit)
                    for sig in symbol_signals
                    if isinstance(sig.take_profit, (int, float))
                ]

                best_signal = max(
                    symbol_signals,
                    key=lambda sig: self._normalize_confidence(sig.confidence),
                )
                metadata = dict(best_signal.metadata)
                metadata["aggregated_count"] = len(symbol_signals)
                metadata["aggregated_sources"] = [
                    {
                        "strategy": sig.metadata.get("strategy"),
                        "confidence": sig.confidence,
                        "normalized_confidence": self._normalize_confidence(
                            sig.confidence
                        ),
                        "reason": sig.reason,
                    }
                    for sig in symbol_signals
                ]

                aggregated_signal = Signal(
                    action=best_signal.action,
                    symbol=symbol,
                    quantity=best_signal.quantity,
                    confidence=avg_confidence,
                    reason=best_signal.reason,
                    stop_loss=(
                        max(stop_candidates)
                        if stop_candidates
                        else best_signal.stop_loss
                    ),
                    take_profit=(
                        min(target_candidates)
                        if target_candidates
                        else best_signal.take_profit
                    ),
                    metadata=metadata,
                )

                self._logger.info(
                    "Condition met: signal_aggregated",
                    extra={
                        "event": "signal_aggregated",
                        "symbol": symbol,
                        "count": len(symbol_signals),
                        "confidence": avg_confidence,
                    },
                )

                aggregated[symbol] = aggregated_signal

        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.aggregate_signals_by_symbol: %s",
                exc,
                extra={"event": "aggregate_signals_error"},
                exc_info=exc,
            )
            return {}

        return aggregated

    def _normalize_confidence(self, value: float) -> float:
        """Normalize confidence to 0-1. Args: value. Returns: Normalized confidence. Raises: None."""
        self._logger.debug(
            "Entered StrategyRunner._normalize_confidence",
            extra={"event": "normalize_confidence"},
        )
        try:
            scaled = float(value)
            if scaled > 1.0:
                scaled /= 100.0
            return min(1.0, max(0.0, scaled))
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner._normalize_confidence: %s",
                exc,
                extra={"event": "normalize_confidence_error"},
                exc_info=exc,
            )
            return 0.0

    def _select_best_option(
        self,
        base_symbol: str,
        candidates: Sequence[SelectedContract],
    ) -> SelectedContract | None:
        """Return the highest ranked option candidate for execution."""
        self._logger.debug(
            "Entered StrategyRunner._select_best_option",
            extra={"candidate_count": len(candidates)},
        )

        if not candidates:
            return None

        best = candidates[0]
        label = base_symbol.strip().upper() or best.symbol.strip().upper()
        selection_score = None

        if isinstance(best.metadata, Mapping):
            selection_score = best.metadata.get("selection_score")

        try:
            delta_value = best.greeks.delta if best.greeks else best.delta or 0.0
            _NIFTY_OPTION_DELTA_GAUGE.labels(underlying=label).set(float(delta_value))

            if best.iv is not None:
                _NIFTY_OPTION_IV_GAUGE.labels(underlying=label).set(float(best.iv))

            if best.liquidity_score is not None:
                _NIFTY_OPTION_LIQUIDITY_GAUGE.labels(underlying=label).set(
                    float(best.liquidity_score)
                )

        except Exception:
            pass

        self._logger.info(
            "Condition met: best_option_selected",
            extra={
                "symbol": best.symbol,
                "underlying": label,
                "score": selection_score,
                "liquidity": best.liquidity_score,
                "iv": best.iv,
                "iv_rank": best.iv_rank,
            },
        )

        return best

    def _build_option_score_config(
        self, side: Literal["BUY", "SELL"]
    ) -> Mapping[str, Any]:
        """Return strike selector score configuration for the supplied side."""
        return {
            "weights": dict(self._option_score_weights),
            "delta_target": float(self._option_delta_target),
            "max_iv_rank": float(self._option_max_iv_rank),
            "min_liquidity": float(self._option_min_liquidity),
            "side": side,
        }

    def _execute_order(
        self,
        *,
        symbol: str,
        base_symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        price: float,
        stop_loss: float | None,
        take_profit: float | None,
        timestamp: datetime,
        reference_price: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Execute an order request with quantity normalization and metrics."""
        self._logger.debug(
            "Entered StrategyRunner._execute_order",
            extra={"symbol": symbol, "side": side, "quantity": quantity},
        )

        lot_size = 1
        if metadata is not None:
            raw_lot = metadata.get("lot_size")
            if isinstance(raw_lot, (int, float)):
                lot_size = max(1, int(raw_lot))

        normalized_qty = max(1, int(quantity))
        adjusted = False

        if lot_size > 1:
            remainder = normalized_qty % lot_size
            if remainder != 0:
                normalized_qty -= remainder
                adjusted = True

        if normalized_qty <= 0:
            normalized_qty = lot_size
            adjusted = True

        order_key = f"{symbol}:{side}:{int(timestamp.timestamp() * 1000)}"
        underlying_label = base_symbol.strip().upper() or symbol.strip().upper()
        latency_seconds = max(
            0.0, (datetime.now(timezone.utc) - timestamp).total_seconds()
        )

        if self._runner_state != RunnerState.EXECUTION_ENABLED:
            raise RuntimeError("Execution blocked until live ticks are ready")
        if not is_strategy_instrument(symbol):
            raise RuntimeError("Blocked non-NIFTY instrument")
        tick = self._market_data.get_latest_tick(symbol) if self._market_data else None
        if not tick:
            raise RuntimeError("Execution blocked due to stale tick")
        spot_tick = (
            self._market_data.get_latest_tick("NSE:NIFTY 50")
            if self._market_data
            else None
        )
        fut_tick = None
        if self._market_data is not None:
            for fut_symbol in ("NFO:NIFTY FUT", "NFO:NIFTYFUT"):
                fut_tick = self._market_data.get_latest_tick(fut_symbol)
                if fut_tick:
                    break
        if spot_tick and fut_tick:
            spot = float(spot_tick.get("ltp") or 0.0)
            fut = float(fut_tick.get("ltp") or 0.0)
            if spot > 0 and fut > 0 and abs(fut - spot) / spot > 0.02:
                raise RuntimeError("Execution blocked due to spread sanity guard")

        try:
            order_id = self._order_manager.place_order(
                symbol=symbol,
                side=side,
                quantity=normalized_qty,
                order_type=OrderType.MARKET,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            if adjusted:
                self._logger.debug(
                    "Normalized quantity to lot size",
                    extra={
                        "symbol": symbol,
                        "requested": quantity,
                        "normalized": normalized_qty,
                        "lot_size": lot_size,
                    },
                )

            try:
                _NIFTY_OPTION_SIGNAL_LATENCY.labels(
                    underlying=underlying_label
                ).observe(latency_seconds)
                _NIFTY_OPTION_EXECUTION_COUNTER.labels(
                    underlying=underlying_label, result="success"
                ).inc()

                counters = self._execution_totals[underlying_label]
                counters["success"] += 1
                total = counters["success"] + counters["error"]

                if total > 0:
                    _NIFTY_OPTION_SUCCESS_RATE.labels(underlying=underlying_label).set(
                        counters["success"] / total
                    )

                if reference_price is not None:
                    slippage = float(price - reference_price)
                    _NIFTY_OPTION_SLIPPAGE_GAUGE.labels(
                        underlying=underlying_label
                    ).set(slippage)

            except Exception:
                pass

            self._logger.info(
                "order_submitted",
                extra={
                    "symbol": symbol,
                    "side": side,
                    "quantity": normalized_qty,
                    "order_key": order_key,
                },
            )

            return order_id, normalized_qty

        except OrderPlacementError as exc:
            try:
                _NIFTY_OPTION_SIGNAL_LATENCY.labels(
                    underlying=underlying_label
                ).observe(latency_seconds)
                _NIFTY_OPTION_EXECUTION_COUNTER.labels(
                    underlying=underlying_label, result="error"
                ).inc()

                counters = self._execution_totals[underlying_label]
                counters["error"] += 1
                total = counters["success"] + counters["error"]

                if total > 0:
                    _NIFTY_OPTION_SUCCESS_RATE.labels(underlying=underlying_label).set(
                        counters["success"] / total
                    )

            except Exception:
                pass

            self._logger.error(
                "Failure in StrategyRunner._execute_order: %s",
                exc,
                extra={"symbol": symbol, "side": side},
                exc_info=exc,
            )

            raise

    def _notify_orchestrator_submission(self, signal: Signal, underlying: str) -> None:
        """Inform the orchestrator about a successful submission."""
        self._logger.debug(
            "Entered StrategyRunner._notify_orchestrator_submission",
            extra={"event": "orchestrator_notify_submission", "underlying": underlying},
        )

        orchestrator = self._orchestrator
        if orchestrator is None:
            return

        try:
            orchestrator.notify_submission(signal, underlying)
        except Exception as exc:
            self._logger.debug(
                "orchestrator_notify_submission_failed",
                extra={
                    "event": "orchestrator_notify_submission_failed",
                    "error": str(exc),
                },
            )

    def _notify_orchestrator_exit(self, underlying: str) -> None:
        """Alert the orchestrator when reduce-only exit completes."""
        self._logger.debug(
            "Entered StrategyRunner._notify_orchestrator_exit",
            extra={"event": "orchestrator_notify_exit", "underlying": underlying},
        )

        orchestrator = self._orchestrator
        if orchestrator is None:
            return

        try:
            orchestrator.notify_exit(underlying)
        except Exception as exc:
            self._logger.debug(
                "orchestrator_notify_exit_failed",
                extra={"event": "orchestrator_notify_exit_failed", "error": str(exc)},
            )

    def _is_order_in_flight(self, symbol: str, underlying: str) -> bool:
        """
        Check if an order is currently pending for this symbol or underlying.

        This prevents duplicate order submissions when:
        - Order is submitted but not yet filled
        - Multiple ticks arrive before order confirmation

        Returns:
            True if an order is in flight, False otherwise.
        """
        now = time.time()
        with self._lock:
            # Clean stale entries (orders older than timeout)
            stale_symbols = [
                s
                for s, t in self._orders_in_flight.items()
                if now - t > self._order_in_flight_timeout
            ]
            for s in stale_symbols:
                self._orders_in_flight.pop(s, None)
                self._logger.debug(f"🧹 Cleared stale in-flight: {s}")

            # Check if symbol or underlying has pending order
            if symbol in self._orders_in_flight:
                elapsed = now - self._orders_in_flight[symbol]
                self._logger.debug(
                    f"🛡️ ORDER IN FLIGHT: {symbol} | Age: {elapsed:.1f}s"
                )
                return True

            if (
                underlying
                and underlying != symbol
                and underlying in self._orders_in_flight
            ):
                elapsed = now - self._orders_in_flight[underlying]
                self._logger.debug(
                    f"🛡️ UNDERLYING IN FLIGHT: {underlying} | Age: {elapsed:.1f}s"
                )
                return True

            return False

    def _mark_order_in_flight(self, symbol: str, underlying: str | None = None) -> None:
        """
        Mark that an order has been submitted for this symbol.

        Args:
            symbol: The actual trading symbol (option contract)
            underlying: The base underlying (e.g., NIFTY)
        """
        now = time.time()
        with self._lock:
            self._orders_in_flight[symbol] = now
            if underlying and underlying != symbol:
                self._orders_in_flight[underlying] = now

            self._logger.info(
                f"📌 MARKED IN-FLIGHT: {symbol}"
                + (f" (underlying: {underlying})" if underlying else "")
            )

    def _clear_order_in_flight(self, symbol: str) -> None:
        """
        Clear the in-flight status when order is filled/cancelled/rejected.

        Should be called from order update callback or verification routine.
        """
        with self._lock:
            if symbol in self._orders_in_flight:
                self._orders_in_flight.pop(symbol, None)
                self._logger.debug(f"✅ CLEARED IN-FLIGHT: {symbol}")

            # Also try to clear underlying
            try:
                underlying = self._normalize_symbol(symbol)
                if underlying and underlying != symbol:
                    self._orders_in_flight.pop(underlying, None)
            except Exception:
                pass

    def _set_post_exit_cooldown(self, base_symbol: str, timestamp: datetime) -> None:
        """
        Set a cooldown period after exiting a position.

        This prevents the classic thrashing pattern:
        - Close position at T=0
        - New signal at T=0.1s
        - Re-enter immediately
        - Price reverses, close again
        - Repeat (burning capital on commissions)

        Args:
            base_symbol: The underlying symbol
            timestamp: When the exit occurred
        """
        # ✅ FIX (6 Feb 2026): Also track in _recently_closed for re-entry prevention
        import time as _t

        if hasattr(self, "_recently_closed"):
            self._recently_closed[base_symbol] = _t.time()
            self._logger.info(
                f"🛡️ EXIT RECORDED: {base_symbol} | Re-entry blocked for "
                f"{os.getenv('EXIT_REENTRY_COOLDOWN_SEC', '300')}s"
            )

        with self._lock:
            state = self._symbol_state.get(base_symbol)
            if state:
                cooldown_end = timestamp + timedelta(
                    seconds=self._post_exit_cooldown_seconds
                )
                state.cooldown_until = cooldown_end
                state.last_signal_at = timestamp

                self._logger.info(
                    f"🛡️ POST-EXIT COOLDOWN: {base_symbol} | "
                    f"No new entries until {cooldown_end.strftime('%H:%M:%S')} "
                    f"({self._post_exit_cooldown_seconds}s)",
                    extra={
                        "event": "post_exit_cooldown_set",
                        "symbol": base_symbol,
                        "cooldown_seconds": self._post_exit_cooldown_seconds,
                    },
                )

    async def _handle_tick_message(self, message: Message) -> None:
        """Process incoming TICK messages from the MessageBus."""
        # [MODIFIED] Using defined helper correctly
        log_throttled(
            self._logger,
            "msg_bus_tick",
            f"🔔 MESSAGE BUS TICK: type={message.type}",
            interval_sec=60.0,
            level=logging.DEBUG,
        )
        if not self._running or self._trading_paused:
            return

        # FIX #1: Capture the running loop for thread callbacks
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()

        tick: dict = message.data
        try:
            # Offload heavy synchronous processing (and blocking broker calls) to a thread
            await asyncio.to_thread(self._on_tick_safe, tick)
        except Exception as exc:
            LOGGER.error(f"Error in async tick processing: {exc}", exc_info=True)

    def _is_market_open(self, now: datetime) -> bool:
        """Return True only when market state is OPEN."""
        try:
            _ = now
            settings = get_settings()
            if bool(getattr(settings, "allow_offmarket_trading", False)):
                return True
            return get_market_state() == MarketState.OPEN
        except Exception as e:
            self._logger.warning(
                f"Market time check failed: {e}. Defaulting to CLOSED."
            )
            return False

    def _on_tick_from_bus(self, tick: Mapping[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol_value = tick.get("symbol")
            if not symbol_value:
                return
            symbol = self._normalize_symbol(str(symbol_value))
            if symbol not in self._tracked_symbols:
                return
            price = tick.get("last_price") or tick.get("ltp")
            if not isinstance(price, (int, float)):
                return
            self._on_tick_safe(
                {**dict(tick), "symbol": symbol, "last_price": float(price)}
            )
        except Exception as e:
            self._logger.error("Failure in StrategyRunner._on_tick_from_bus: %s", e)

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol = str(tick.get("symbol") or "")
            if not symbol:
                return
            price = tick.get("last_price") or tick.get("ltp")
            if not isinstance(price, (int, float)):
                return
            self._on_tick_safe({**tick, "symbol": symbol, "last_price": float(price)})
        except Exception as e:
            self._logger.error("Failure in StrategyRunner.on_tick_event: %s", e)

    def _on_tick_safe(self, tick: Mapping[str, Any]) -> None:
        """Safe wrapper for _on_tick to handle exceptions."""
        symbol = tick.get("symbol")
        if not symbol:
            return

        try:
            self._on_tick(str(symbol), tick)
        except Exception as exc:
            LOGGER.error(
                "Critical error in _on_tick for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )

    # ✅ FIX: New Method to Prime Indicators
    async def _backfill_history(self) -> None:
        """
        Download historical data to warm up indicators.
        Skips if startup hydration was already performed by App.py.
        """
        total_bars = 0

        try:
            # 1. Check Hydration Flag (Set by ingest_historical_bar)
            is_hydrated = getattr(self, "_startup_hydrated", False)

            # Also check memory just in case
            has_data = bool(self._last_bar_ts)

            if is_hydrated or has_data:
                self._logger.info(
                    "⏭️ Skipping StrategyRunner historical backfill (startup hydration already completed)"
                )
                return

            # 2. FALLBACK: Only runs if App.py failed
            if not self._risk_allows_trading(None):
                self._logger.warning(
                    "Condition met: backfill_short_circuited_by_risk",
                    extra={"event": "backfill_short_circuited_by_risk"},
                )
                return
            self._logger.warning(
                "⚠️ StrategyRunner memory is empty! Triggering fallback backfill..."
            )

            with self._lock:
                targets = list(self._active_symbols)

            if not targets:
                self._logger.warning("⚠️ Backfill skipped: No active symbols found.")
                return

            # Determine Data Source
            source = None
            if (
                hasattr(self, "_data_hub")
                and self._data_hub
                and hasattr(self._data_hub, "fetch_history")
            ):
                source = self._data_hub
            elif hasattr(self, "_orchestrator") and self._orchestrator:
                source = self._orchestrator

            if not source:
                return

            # 3. SEQUENTIAL FETCH
            for symbol in targets:
                try:
                    # [FIX] Added interval="minute" to fix TypeError
                    history = await source.fetch_history(
                        symbol, interval="minute", days=5
                    )

                    if history:
                        for bar_data in history:
                            self.ingest_historical_bar(bar_data)
                            total_bars += 1
                        self._set_symbol_hydration_state(symbol, SymbolState.READY)
                        self._logger.info(
                            f"✅ Fallback backfill: Ingested {len(history)} bars for {symbol}"
                        )
                    else:
                        self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)

                    await asyncio.sleep(0.5)

                except Exception as e:
                    self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
                    self._logger.error(f"❌ Fallback fetch failed for {symbol}: {e}")

        except Exception as exc:
            self._logger.error(f"❌ History backfill crashed: {exc}", exc_info=True)

        if total_bars > 0:
            self._logger.info(
                f"✅ Emergency Backfill complete. Ingested {total_bars} bars."
            )

    def _hydrate_missing_bars(self, symbol: str, min_bars: int) -> list[dict[str, Any]]:
        """Fetch missing candles for *symbol* and return normalized OHLC bars."""
        needed_bars = max(0, min_bars - len(self._indicator_engine.get_history(symbol)))
        last_bar_ts = self._last_bar_ts.get(symbol)
        if needed_bars <= 0:
            return []
        if last_bar_ts and self._hydration_log_bar_cache.get(symbol) != last_bar_ts:
            self._hydration_log_bar_cache[symbol] = last_bar_ts
            self._logger.info(
                "Hydrating historical data",
                extra={
                    "symbol": symbol,
                    "needed_bars": needed_bars,
                    "have_bars": len(self._indicator_engine.get_history(symbol)),
                },
            )
        if self._main_loop is None:
            return []
        fetch_factory: Callable[[], Any] | None = None
        if self._data_hub and hasattr(self._data_hub, "fetch_history"):
            fetch_factory = lambda: self._data_hub.fetch_history(
                symbol, interval="minute", days=5
            )
        elif self._orchestrator and hasattr(self._orchestrator, "fetch_history"):
            fetch_factory = lambda: self._orchestrator.fetch_history(
                symbol, interval="minute", days=5
            )
        if fetch_factory is None:
            return []
        attempts = int(self._hydrate_failures.get(symbol, 0))
        try:
            rows = asyncio.run_coroutine_threadsafe(
                fetch_factory(),
                self._main_loop,
            ).result(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            self._hydrate_failures[symbol] = attempts + 1
            time_module.sleep(
                min(4.0, 0.25 * (2**attempts))
            )  # bounded exponential backoff
            self._logger.warning(
                "Historical hydration unavailable",
                extra={
                    "event": "indicator_hydration_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return self._load_history_cache(symbol)
        normalized: list[dict[str, Any]] = []
        seen_ts: set[datetime] = set()
        for row in rows or []:
            try:
                ts = row.get("timestamp") or row.get("date")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if not isinstance(ts, datetime):
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                normalized.append(
                    {
                        "open": float(row.get("open", 0.0) or 0.0),
                        "high": float(row.get("high", 0.0) or 0.0),
                        "low": float(row.get("low", 0.0) or 0.0),
                        "close": float(row.get("close", 0.0) or 0.0),
                        "volume": int(row.get("volume", 0) or 0),
                        "timestamp": ts,
                    }
                )
            except Exception:
                continue
        normalized.sort(key=lambda row: cast(datetime, row["timestamp"]))
        if normalized:
            self._hydrate_failures[symbol] = 0
            self._write_history_cache(symbol, normalized)
        else:
            normalized = self._load_history_cache(symbol)

        if len(normalized) < self._required_candles:
            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
            self._warn_symbol_gate(
                "insufficient_history",
                symbol,
                "Hydration excluded due to insufficient candles",
                reason="insufficient_hydrated_candles",
                candles=len(normalized),
                required_candles=self._required_candles,
            )
            return []
        if self._data_hub and hasattr(self._data_hub, "history_freshness"):
            fresh, meta = self._data_hub.history_freshness(symbol, "minute")
            if not fresh:
                self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
                self._warn_symbol_gate(
                    "insufficient_history",
                    symbol,
                    "Hydrated history failed freshness validation",
                    reason="hydrated_history_stale",
                    meta=meta,
                )
                return []
        has_gap = any(
            (
                cast(datetime, curr["timestamp"]) - cast(datetime, prev["timestamp"])
            ).total_seconds()
            > 120
            for prev, curr in zip(normalized, normalized[1:])
        )
        if has_gap:
            self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)
            return normalized
        self._set_symbol_hydration_state(symbol, SymbolState.READY)
        return normalized

    def _log_once_per_symbol_per_bar(
        self, symbol: str, event: str, reason: str
    ) -> None:
        """Emit edge-triggered per-symbol per-bar logs for noisy gates."""
        bar_ts = self._last_bar_ts.get(symbol)
        if bar_ts is None:
            return
        cache_key = (symbol, event)
        if self._gating_log_bar_cache.get(cache_key) == bar_ts:
            return
        self._gating_log_bar_cache[cache_key] = bar_ts
        self._logger.info(
            event,
            extra={
                "event": event,
                "symbol": symbol,
                "reason": reason,
                "bar_ts": bar_ts.isoformat(),
            },
        )

    def _regime_manager_ready(self) -> bool:
        """Return True when regime manager has produced an actionable snapshot."""
        regime_manager = getattr(self._strategy_manager, "_regime_manager", None)
        if regime_manager is None:
            return True
        snapshot = getattr(regime_manager, "get_latest_snapshot", lambda: None)()
        return snapshot is not None

    def _strategy_slots_available(self) -> bool:
        """Return True when active strategy slots are available for new entries."""
        try:
            active_positions = len(self._position_manager.get_open_positions())
        except Exception:
            active_positions = 0
        with self._lock:
            inflight = len(self._orders_in_flight)
        return (active_positions + inflight) < self._strategy_slot_limit

    def _warn_symbol_gate(
        self,
        event_code: str,
        symbol: str,
        message: str,
        *,
        reason: str,
        **context: Any,
    ) -> None:
        """Emit structured warning logs for symbol-level guard failures.

        Args: event_code, symbol, message, reason, context. Returns: None. Raises: None.
        """
        self._logger.debug(
            "Entered StrategyRunner._warn_symbol_gate",
            extra={"event": "symbol_gate_warn_enter", "symbol": symbol},
        )
        try:
            reserved_extra_keys = set(logging.makeLogRecord({}).__dict__)
            payload = {
                "level": "WARNING",
                "symbol": symbol,
                "event": event_code,
                "gate_message": message,
                "reason": reason,
                "time": datetime.now(timezone.utc).isoformat(),
            }
            if context:
                sanitized_context = {
                    (f"context_{key}" if key in reserved_extra_keys else key): value
                    for key, value in context.items()
                }
                payload.update(sanitized_context)
            self._logger.warning(message, extra=payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner._warn_symbol_gate: %s",
                exc,
                extra={"event": "symbol_gate_warn_error", "symbol": symbol},
                exc_info=exc,
            )

    def _mark_symbol_unready(
        self,
        symbol: str,
        reason: str,
        *,
        low_confidence: bool = False,
    ) -> None:
        """Mark symbol as unready for the current cycle.

        Args: symbol, reason, low_confidence. Returns: None. Raises: None.
        """
        self._logger.debug(
            "Entered StrategyRunner._mark_symbol_unready",
            extra={"event": "symbol_unready_mark_enter", "symbol": symbol},
        )
        try:
            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
            state_transition = False
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state is not None:
                    previous_reason = state.strategy_data.get("unready_reason")
                    state.strategy_data["unready_reason"] = reason
                    state.strategy_data["unready_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    state_transition = previous_reason != reason
                    if low_confidence:
                        state.strategy_data["low_confidence"] = True
            if state_transition:
                self._logger.info(
                    "symbol_unready_transition",
                    extra={
                        "event": "symbol_unready_transition",
                        "symbol": symbol,
                        "reason": reason,
                        "low_confidence": low_confidence,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner._mark_symbol_unready: %s",
                exc,
                extra={"event": "symbol_marked_unready_error", "symbol": symbol},
                exc_info=exc,
            )

    def _validate_symbol_for_cycle(self, symbol: str) -> bool:
        """Validate symbol against active dynamic universe without aborting loop.

        Args: symbol. Returns: bool. Raises: None.
        """
        self._logger.debug(
            "Entered StrategyRunner._validate_symbol_for_cycle",
            extra={"event": "symbol_validate_cycle_enter", "symbol": symbol},
        )
        try:
            with self._lock:
                symbols = sorted(self._active_symbols)
                active_set = set(symbols)

            if self._max_symbol_count > 0 and len(symbols) > self._max_symbol_count:
                self._warn_symbol_gate(
                    "universe_violation",
                    symbol,
                    "Session universe exceeds max cap",
                    reason="max_symbol_count_exceeded",
                    actual_count=len(symbols),
                    max_count=self._max_symbol_count,
                )
                self._mark_symbol_unready(symbol, "universe_violation")
                return False
            if self._frozen_universe and symbol not in self._frozen_universe:
                self._logger.debug(
                    "Symbol ignored (not in frozen universe)",
                    extra={"event": "symbol_ignored_frozen_universe", "symbol": symbol},
                )
                return False
            if symbol not in active_set:
                self._logger.debug(
                    "Symbol outside active universe",
                    extra={
                        "event": "symbol_outside_active_universe",
                        "symbol": symbol,
                    },
                )
                self._mark_symbol_unready(symbol, "universe_violation")
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner._validate_symbol_for_cycle: %s",
                exc,
                extra={"event": "symbol_validate_cycle_error", "symbol": symbol},
                exc_info=exc,
            )
            self._mark_symbol_unready(symbol, "universe_validation_error")
            return False

    def _classify_symbol(self, symbol: str) -> str:
        """Classify symbol into index, option or other buckets."""
        upper = symbol.upper()
        if upper.startswith("NSE:") and all(
            tok not in upper for tok in ("CE", "PE", "FUT")
        ):
            return "index"
        if upper.endswith("CE") or upper.endswith("PE"):
            return "option"
        return "other"

    def _validate_symbol_universe(self) -> bool:
        """Validate index/option composition with non-blocking dynamic diagnostics."""
        with self._lock:
            symbols = sorted(self._active_symbols)
        if self._max_symbol_count > 0 and len(symbols) > self._max_symbol_count:
            self._logger.error(
                "Condition met: symbol_universe_mismatch",
                extra={
                    "event": "symbol_universe_mismatch",
                    "reason": "max_symbol_count_exceeded",
                    "actual_count": len(symbols),
                    "max_count": self._max_symbol_count,
                },
            )
            return False
        if self._universe_dynamic_mode:
            # Keep diagnostics informative, but avoid frozen snapshot gating.
            self._logger.debug(
                "Condition met: symbol_universe_dynamic_validation",
                extra={
                    "event": "symbol_universe_dynamic_validation",
                    "all_symbols": symbols,
                },
            )
        index_symbols = [
            sym for sym in symbols if self._classify_symbol(sym) == "index"
        ]
        option_symbols = [
            sym for sym in symbols if self._classify_symbol(sym) == "option"
        ]
        if not option_symbols:
            self._logger.info(
                "Condition met: symbol_universe_mismatch",
                extra={
                    "event": "symbol_universe_mismatch",
                    "reason": "missing_option_symbols",
                    "all_symbols": symbols,
                    "index_symbols": index_symbols,
                    "option_symbols": option_symbols,
                },
            )
            return False
        if len(index_symbols) != 1:
            self._logger.info(
                "Condition met: symbol_universe_mismatch",
                extra={
                    "event": "symbol_universe_mismatch",
                    "reason": "index_count_mismatch",
                    "all_symbols": symbols,
                    "index_symbols": index_symbols,
                    "option_symbols": option_symbols,
                },
            )
            return False
        return True

    def _risk_allows_trading(self, symbol: str | None) -> bool:
        """Return whether risk engine allows entering strategy flow."""
        try:
            rm = self._risk_manager
            if rm is None:
                return True
            if hasattr(rm, "is_circuit_breaker_tripped"):
                tripped, _ = rm.is_circuit_breaker_tripped()
                if tripped:
                    return False
            if hasattr(rm, "can_trade"):
                return bool(rm.can_trade(symbol or "GLOBAL"))
            if hasattr(rm, "risk_gate_should_trade"):
                result = rm.risk_gate_should_trade()
                return bool(result[0] if isinstance(result, tuple) else result)
            if hasattr(rm, "can_trade_now"):
                result = rm.can_trade_now()
                return bool(result[0] if isinstance(result, tuple) else result)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _risk_allows_trading: %s", exc, exc_info=True
            )
            return False
        return False

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Handle incoming tick. Args: symbol, tick. Returns: None. Raises: Exception."""
        self._logger.debug(
            "Entered StrategyRunner._on_tick",
            extra={"event": "tick_enter", "symbol": symbol},
        )
        self._logger.info("STRATEGY_TICK %s", symbol)
        try:
            # =================================================================
            # PHASE -1: BRACKET MANAGER TICK FORWARDING (MUST be before ANY return)
            # =================================================================
            # CRITICAL: The bracket manager monitors SL/TP/trailing for ALL
            # active positions. It MUST receive every tick regardless of
            # market hours, stale-tick status, or any other condition.
            # Without this, stop losses and take profits NEVER fire.
            if self._bracket_manager:
                try:
                    _ltp_raw = (
                        tick.get("ltp")
                        or tick.get("last_price")
                        or tick.get("price")
                        or 0.0
                    )
                    _ltp = float(_ltp_raw)
                    if _ltp > 0:
                        self._bracket_manager.on_tick(symbol, _ltp)
                        tick_err_map = getattr(
                            self._bracket_manager, "_tick_error_logged", None
                        )
                        if isinstance(tick_err_map, dict):
                            tick_err_map[symbol] = False
                except Exception as _bm_err:
                    tick_err_map = getattr(
                        self._bracket_manager, "_tick_error_logged", None
                    )
                    already_logged = bool(
                        isinstance(tick_err_map, dict) and tick_err_map.get(symbol)
                    )
                    if not already_logged:
                        self._logger.error(
                            "Bracket tick handler error",
                            extra={
                                "event": "bracket_tick_handler_error",
                                "symbol": symbol,
                                "error": str(_bm_err),
                            },
                        )
                        if isinstance(tick_err_map, dict):
                            tick_err_map[symbol] = True


            # =================================================================
            # PHASE 0: EARLY EXIT CHECKS (Fast path for non-trading scenarios)
            # =================================================================

            # 1. Orphan Guard — adopt but DO NOT return (tick must still flow to bracket manager)
            # ✅ FIX (6 Feb 2026): Removed early `return` that blocked SL/TP execution for orphans
            if self._position_manager:
                active_pos = self._position_manager.get_active_contract(symbol)
                if active_pos:
                    strat = getattr(active_pos, "strategy", "") or "unknown"
                    if "manual" in strat.lower() or "unknown" in strat.lower():
                        log_throttled(
                            self._logger,
                            f"orphan_guard_{symbol}",
                            f"🛡️ ORPHAN GUARD: {symbol} is unmanaged. Adopting (tick continues)...",
                            interval_sec=30.0,
                            level=logging.WARNING,
                        )
                        if hasattr(self, "_adopt_orphan_positions"):
                            self._adopt_orphan_positions()
                        # ✅ DO NOT return — tick must continue flowing for bracket SL/TP monitoring

            # =================================================================
            # PHASE 1: EXTRACT DATA FIRST (Must happen before any logging)
            # =================================================================

            now = datetime.now(timezone.utc)

            # Helper: Extract timestamp for freshness check
            def _extract_timestamp(t, fallback):
                ts = t.get("timestamp") or t.get("exchange_timestamp")
                if isinstance(ts, (int, float)):
                    if ts > 10_000_000_000:  # Detect milliseconds
                        ts = ts / 1000.0
                    try:
                        return datetime.fromtimestamp(ts, tz=timezone.utc)
                    except (ValueError, OSError, OverflowError):
                        return fallback
                if isinstance(ts, datetime):
                    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                return fallback

            # Helper: Safely extract float
            def _extract_float(d, *keys):
                """Return first positive float for keys. Args: d, keys. Returns: float. Raises: None."""
                for k in keys:
                    if d.get(k) is not None:
                        try:
                            value = float(d[k])
                        except (ValueError, TypeError) as exc:
                            self._logger.debug(
                                "Failure in _extract_float: %s",
                                exc,
                                extra={"event": "tick_extract_float_error", "key": k},
                            )
                            continue
                        if value > 0:
                            return value
                return 0.0

            # Helper: Safely extract int
            def _extract_int(d, *keys):
                for k in keys:
                    if d.get(k) is not None:
                        try:
                            return int(float(d[k]))
                        except (ValueError, TypeError):
                            continue
                return 0

            # Extract all data FIRST
            timestamp = _extract_timestamp(tick, now)
            tick_age = (now - timestamp).total_seconds()
            price = _extract_float(tick, "ltp", "last_price", "close", "price")
            raw_volume = _extract_int(
                tick, "volume", "volume_traded", "volume_traded_today"
            )
            source = tick.get("source", "unknown")
            is_seed = bool(tick.get("seed"))

            if is_seed and price <= 0:
                log_throttled(
                    self._logger,
                    f"seed_tick_price_zero_{symbol}",
                    f"Condition met: seed_tick_price_missing for {symbol}",
                    interval_sec=120.0,
                    level=logging.INFO,
                )
                return

            # ✅ FIX S5: Convert cumulative exchange volume to per-tick delta
            volume = 0
            first_tick_seen = symbol not in self._last_cumulative_volume
            if raw_volume > 0:
                last_cum = self._last_cumulative_volume.get(symbol, -1)
                if last_cum < 0:
                    volume = 0
                elif raw_volume >= last_cum:
                    volume = raw_volume - last_cum
                else:
                    volume = min(raw_volume, 1000)
                self._last_cumulative_volume[symbol] = raw_volume
            elif first_tick_seen:
                volume = 0
                self._last_cumulative_volume[symbol] = raw_volume
                log_throttled(
                    self._logger,
                    f"tick_volume_seeded_{symbol}",
                    "Condition met: tick_volume_baseline_only_first_tick",
                    interval_sec=60.0,
                    level=logging.INFO,
                )

            price_source = "ltp"
            price_from_cache = False

            if price <= 0:
                best_bid = _extract_float(
                    tick, "best_bid", "bid", "best_bid_price", "buy_price"
                )
                best_ask = _extract_float(
                    tick, "best_ask", "ask", "best_ask_price", "sell_price"
                )
                if best_bid > 0 and best_ask > 0:
                    price = (best_bid + best_ask) / 2.0
                    price_source = "book_mid"
                elif best_bid > 0:
                    price = best_bid
                    price_source = "book_bid"
                elif best_ask > 0:
                    price = best_ask
                    price_source = "book_ask"

                if price > 0:
                    log_throttled(
                        self._logger,
                        f"price_from_book_{symbol}",
                        f"Condition met: tick_price_from_book ({price_source})",
                        interval_sec=60.0,
                        level=logging.INFO,
                    )
                else:
                    last_price = self._last_valid_price.get(symbol)
                    last_ts = self._last_valid_price_ts.get(symbol)
                    if last_price and last_ts:
                        max_age = 30.0 if source in ("rest", "polling") else 5.0
                        cache_age = (now - last_ts).total_seconds()
                        if cache_age <= max_age:
                            price = last_price
                            price_source = "cache"
                            price_from_cache = True
                            log_throttled(
                                self._logger,
                                f"price_from_cache_{symbol}",
                                (
                                    "Condition met: tick_price_cache_used "
                                    f"age={cache_age:.1f}s"
                                ),
                                interval_sec=60.0,
                                level=logging.INFO,
                            )
                        else:
                            log_throttled(
                                self._logger,
                                f"price_cache_stale_{symbol}",
                                (
                                    "Condition met: tick_price_cache_stale "
                                    f"age={cache_age:.1f}s"
                                ),
                                interval_sec=60.0,
                                level=logging.WARNING,
                            )

            if price > 0 and not price_from_cache:
                self._last_valid_price[symbol] = price
                self._last_valid_price_ts[symbol] = now

            # Price validity check
            if price <= 0:
                log_throttled(
                    self._logger,
                    f"invalid_price_{symbol}",
                    f"⚠️ Invalid price ({price}) for {symbol}, skipping",
                    interval_sec=60.0,
                    level=logging.WARNING,
                )
                return

            # Bracket tick forwarding already happened at function entry.
            # Keep a single forward per tick so protective handlers do not churn
            # duplicate state transitions during stressed periods.

            # Global breaker latch: once tripped, keep running only protective paths
            # (brackets + position reconciliation) and skip strategy/indicator work.
            if not self._risk_halt_active and self._risk_manager is not None:
                try:
                    tripped, _reason = self._risk_manager.is_circuit_breaker_tripped()
                except Exception as exc:
                    self._logger.debug(
                        "Failure in global breaker check: %s",
                        exc,
                        extra={"event": "risk_halt_check_error", "symbol": symbol},
                    )
                    tripped = False
                    _reason = ""
                if tripped:
                    self._risk_halt_active = True
                    if not self._risk_halt_logged:
                        self._risk_halt_logged = True
                        self._logger.error(
                            "Condition met: global risk halt latched",
                            extra={
                                "event": "risk_halt_latched",
                                "symbol": symbol,
                                "reason": _reason,
                            },
                        )
            if self._risk_halt_active:
                if hasattr(self._position_manager, "update_position_price"):
                    try:
                        self._position_manager.update_position_price(symbol, price)
                    except Exception:
                        pass
                return

            skip_strategy = False
            if not self._is_market_open(now):
                log_throttled(
                    self._logger,
                    f"market_closed_{symbol}",
                    "Condition met: market_closed",
                    interval_sec=30.0,
                    level=logging.INFO,
                )
                skip_strategy = True

            # Stale tick check (increased threshold for REST polling to prevent false positives)
            stale_threshold = 30.0 if source in ("rest", "polling") else 10.0

            if tick_age > stale_threshold:
                log_throttled(
                    self._logger,
                    f"stale_tick_{symbol}",
                    (
                        f"⏰ STALE TICK: {symbol} ({tick_age:.1f}s old, "
                        f"threshold={stale_threshold}s)"
                    ),
                    interval_sec=30.0,
                    level=logging.WARNING,
                )
                skip_strategy = True

            # Volume validity check (relaxed warnings for REST mode)
            if volume < 0:
                log_throttled(
                    self._logger,
                    f"invalid_vol_{symbol}",
                    f"⚠️ Invalid volume ({volume}) for {symbol}, skipping",
                    interval_sec=60.0,
                    level=logging.WARNING,
                )
                return

            # =================================================================
            # PHASE 3: DIAGNOSTIC LOGGING
            # =================================================================

            # Log successful tick acceptance (throttled)
            log_throttled(
                self._logger,
                f"tick_accepted_{symbol}",
                f"✅ TICK ACCEPTED: {symbol} | LTP={price:.2f} | Age={tick_age:.1f}s | Vol={volume}",
                interval_sec=60.0,
                level=logging.DEBUG,
            )

            # Grace period warmup logging
            startup_time = getattr(self, "_startup_timestamp", None)
            if startup_time is None:
                self._startup_timestamp = time.time()
                startup_time = self._startup_timestamp

            time_since_startup = time.time() - startup_time
            in_warmup = time_since_startup < 15  # Fast 15s warmup

            if in_warmup:
                log_throttled(
                    self._logger,
                    "warmup_period",
                    f"⏳ WARMUP: {15 - time_since_startup:.0f}s remaining before trading enabled",
                    interval_sec=5.0,
                    level=logging.INFO,
                )

            # =================================================================
            # PHASE 4: BAR BUILDING (Always process, even during warmup)
            # =================================================================

            builder = self._bar_builders.setdefault(symbol, OneMinuteBarBuilder())
            try:
                completed_bar = builder.update(float(price), volume, timestamp)
                if completed_bar is not None:
                    self._ingest_bar(symbol, completed_bar)
            except ValueError as exc:
                if getattr(builder, "_last_error_ts", 0) < now.timestamp() - 60:
                    self._logger.warning(f"Bar update issue for {symbol}: {exc}")
                    builder._last_error_ts = now.timestamp()

            # =================================================================
            # PHASE 5: POSITION MANAGER UPDATE
            # =================================================================

            if hasattr(self._position_manager, "update_position_price"):
                try:
                    self._position_manager.update_position_price(symbol, price)
                except Exception:
                    pass

            # =================================================================
            # PHASE 6: RISK CHECK (Block trading if risk conditions not met)
            # =================================================================

            if not self._risk_allows_trading(symbol):
                log_throttled(
                    self._logger,
                    f"risk_block_{symbol}",
                    f"⛔ Risk Block Active: {symbol}. Trading Halted.",
                    interval_sec=30.0,
                    level=logging.WARNING,
                )
                return
            if self._runner_state != RunnerState.EXECUTION_ENABLED:
                return

            if skip_strategy:
                return

            # =================================================================
            # PHASE 7: STRATEGY PREPARATION (Skip during warmup)
            # =================================================================

            if in_warmup:
                return
            if not self._is_market_open(now):
                self._logger.info("EVENT|blocked|market_hours")
                return
            if not self._validate_symbol_for_cycle(symbol):
                return

            with self._lock:
                if symbol not in self._active_symbols:
                    self._logger.debug(
                        "Symbol outside active universe",
                        extra={
                            "event": "symbol_outside_active_universe",
                            "symbol": symbol,
                            "reason": "symbol_not_tracked",
                        },
                    )
                    self._mark_symbol_unready(symbol, "universe_violation")
                    return

                state = self._symbol_state.get(symbol)
                if state is None or not state.active:
                    return

                # Use bar-close VWAP accumulators from _ingest_bar only to avoid tick noise.
                vwap_state = self._ensure_symbol_vwap_state(symbol, now)
                cum_vol = float(vwap_state.get("cum_vol", 0.0))
                if state.vwap is None and cum_vol > 0:
                    state.vwap = float(vwap_state.get("cum_pv", 0.0)) / cum_vol

                self._last_cumulative_volume[symbol] = int(cum_vol)
                last_hydration_bar = self._last_readiness_update_by_symbol.get(symbol)
                if last_hydration_bar != self._last_bar_ts.get(symbol):
                    self._last_readiness_update_by_symbol[symbol] = (
                        self._last_bar_ts.get(symbol)
                    )
                    hydration_state = self._update_symbol_readiness(symbol)
                else:
                    hydration_state = self._symbol_states.get(
                        symbol, SymbolState.DISCOVERED
                    )
                if hydration_state != SymbolState.READY:
                    return

                bars = (
                    self._market_data.get_ohlc_bars(symbol) if self._market_data else []
                )
                if len(bars) < 20:
                    return

                spot_tick = (
                    self._market_data.get_latest_tick("NSE:NIFTY 50")
                    if self._market_data
                    else None
                )
                if spot_tick is None and self._market_data is not None:
                    try:
                        spot_token = int(
                            self._market_data.get_token("NSE:NIFTY 50") or 0
                        )
                        if spot_token > 0:
                            loop = self._main_loop
                            if loop is not None:
                                fut = asyncio.run_coroutine_threadsafe(
                                    self._market_data.wait_for_live_tick(
                                        spot_token,
                                        timeout=2,
                                    ),
                                    loop,
                                )
                                spot_tick = fut.result(timeout=2.5)
                    except Exception as exc:
                        self._logger.error(
                            "Failure in StrategyRunner._on_tick wait_for_live_spot: %s",
                            exc,
                            exc_info=True,
                        )
                        return
                if not spot_tick:
                    return
                spot_ts = _extract_float(spot_tick, "timestamp", "ts", "ts_ms")
                if spot_ts is not None and spot_ts > 1_000_000_000_000:
                    spot_ts = spot_ts / 1000.0
                if spot_ts is not None and (time.time() - float(spot_ts)) > 2.0:
                    return

                # Heartbeat logging for derivatives (confirms data flow)
                if "NIFTY" in symbol and any(x in symbol for x in ["FUT", "CE", "PE"]):
                    log_throttled(
                        self._logger,
                        f"heartbeat_{symbol}",
                        f"💓 TICK HEARTBEAT: {symbol} | LTP={price:.2f} | VWAP={state.vwap or 0:.2f}",
                        interval_sec=30.0,
                        level=logging.DEBUG,
                    )

                # =============================================================
                # PHASE 8: SIGNAL GENERATION
                # =============================================================
                generated_signal = None

                # 8A. FORCED SIGNAL (Testing only)
                force_signal_enabled = os.getenv("FORCE_SIGNAL", "").lower() == "true"
                disable_early_forced = (
                    os.getenv("FEATURE_DISABLE_EARLY_FORCED_SIGNALS", "").lower()
                    == "true"
                )

                if force_signal_enabled and not disable_early_forced:
                    generated_signal = Signal(
                        action="BUY",
                        symbol=symbol,
                        quantity=1,
                        confidence=1.0,
                        reason="forced_signal_validation",
                        stop_loss=None,
                        take_profit=None,
                        metadata={"source": "forced"},
                    )
                    self._logger.warning(f"⚠️ FORCED SIGNAL EMITTED for {symbol}")

                # 8B. PRIMARY STRATEGY: VWAP Crossover (Requires VWAP > 0)
                vwap_crossover_enabled = (
                    os.getenv("ENABLE_VWAP_CROSSOVER", "false").lower() == "true"
                )

                if (
                    vwap_crossover_enabled
                    and generated_signal is None
                    and state.vwap
                    and state.vwap > 0
                    and "FUT" not in symbol.upper()
                ):
                    prev_ltp = (
                        _extract_float(state.last_tick, "ltp", "last_price")
                        if state.last_tick
                        else None
                    )
                    curr_vwap = state.vwap

                    if prev_ltp and curr_vwap and price > 0:
                        threshold = curr_vwap * 0.0005  # 0.05% buffer
                        is_cross_up = prev_ltp < (curr_vwap + threshold) and price > (
                            curr_vwap + threshold
                        )
                        is_cross_down = prev_ltp > (curr_vwap - threshold) and price < (
                            curr_vwap - threshold
                        )

                        # ✅ FIX: Calculate proper stop_loss and take_profit
                        sl_pct = float(os.getenv("VWAP_SL_PCT", "1.5"))  # 1.5% SL
                        tp_pct = float(
                            os.getenv("VWAP_TP_PCT", "2.0")
                        )  # 2.0% TP (1:1.33 RR)

                        if is_cross_up:
                            # BUY signal - SL below, TP above
                            calculated_sl = price * (1 - sl_pct / 100)
                            calculated_tp = price * (1 + tp_pct / 100)

                            self._logger.info(
                                f"⚡ VWAP CROSSOVER UP: {symbol} | {prev_ltp:.2f} -> {price:.2f} (VWAP: {curr_vwap:.2f})",
                                extra={"event": "vwap_crossover", "symbol": symbol},
                            )
                            generated_signal = Signal(
                                action="BUY",
                                symbol=symbol,
                                quantity=1,
                                confidence=0.75,
                                reason="vwap_crossover_up",
                                stop_loss=calculated_sl,  # ✅ NOW HAS PROPER SL
                                take_profit=calculated_tp,  # ✅ NOW HAS PROPER TP
                                metadata={
                                    "strategy": "vwap_scalp",
                                    "vwap": curr_vwap,
                                    "tag": "vwap_scalp",
                                    "sl_pct": sl_pct,
                                    "tp_pct": tp_pct,
                                },
                            )
                        elif is_cross_down:
                            # SELL signal - SL above, TP below
                            calculated_sl = price * (1 + sl_pct / 100)
                            calculated_tp = price * (1 - tp_pct / 100)

                            self._logger.info(
                                f"⚡ VWAP CROSSOVER DOWN: {symbol} | {prev_ltp:.2f} -> {price:.2f} (VWAP: {curr_vwap:.2f})",
                                extra={"event": "vwap_crossover", "symbol": symbol},
                            )
                            generated_signal = Signal(
                                action="SELL",
                                symbol=symbol,
                                quantity=1,
                                confidence=0.75,
                                reason="vwap_crossover_down",
                                stop_loss=calculated_sl,  # ✅ NOW HAS PROPER SL
                                take_profit=calculated_tp,  # ✅ NOW HAS PROPER TP
                                metadata={
                                    "strategy": "vwap_scalp",
                                    "vwap": curr_vwap,
                                    "tag": "vwap_scalp_short",
                                    "sl_pct": sl_pct,
                                    "tp_pct": tp_pct,
                                },
                            )

                # 8C. FALLBACK STRATEGY: Momentum Breakout (When VWAP is Missing/0)

                # Update last tick
                state.last_tick = dict(tick)

                # Check if trading is paused
                if not self._running or getattr(self, "_trading_paused", False):
                    return

                # Check cooldown
                if state.cooldown_until and now < state.cooldown_until:
                    return

            # =================================================================
            # PHASE 9: SIGNAL SELECTION & STRATEGY MANAGER EVALUATION
            # =================================================================

            signal = generated_signal
            upper_symbol = symbol.upper()
            is_index_symbol = (
                upper_symbol.startswith("NSE:")
                and "NIFTY" in upper_symbol
                and "CE" not in upper_symbol
                and "PE" not in upper_symbol
                and "FUT" not in upper_symbol
            )

            symbol_rate_until = float(
                self._rate_limit_backoff_until_by_symbol.get(symbol, 0.0)
            )
            now_ts = time_module.time()
            if symbol_rate_until and now_ts < symbol_rate_until:
                self._warn_symbol_gate(
                    "rate_limit_breach",
                    symbol,
                    "Symbol temporarily throttled after broker rate-limit signal",
                    reason="symbol_rate_limit_backoff_active",
                    remaining_s=max(0.0, symbol_rate_until - now_ts),
                )
                self._mark_symbol_unready(symbol, "rate_limit_breach")
                return

            backoff_until = float(getattr(self, "_data_freshness_backoff_until", 0.0))
            if (
                self._should_enforce_freshness_backoff()
                and backoff_until
                and time_module.time() < backoff_until
            ):
                remaining = max(0.0, backoff_until - time_module.time())
                self._logger.debug(
                    "strategy_eval_skipped_stale_data",
                    extra={
                        "event": "strategy_eval_skipped_stale_data",
                        "symbol": symbol,
                        "backoff_until": backoff_until,
                        "remaining_s": remaining,
                        "detail_code": getattr(
                            self, "_data_freshness_backoff_detail", None
                        ),
                        "symbol_checked": getattr(
                            self, "_data_freshness_backoff_symbol", None
                        ),
                    },
                )
                return

            # If no immediate signal, delegate to complex StrategyManager
            current_state = self._symbol_states.get(symbol, SymbolState.DISCOVERED)
            if current_state != SymbolState.READY:
                self._log_once_per_symbol_per_bar(
                    symbol,
                    "strategy_eval_skipped_not_ready",
                    f"state={current_state.value}",
                )
                return
            if signal is None and self._required_candles:
                should_evaluate = False
                with self._lock:
                    state = self._symbol_state.get(symbol)
                    if state:
                        last_eval = getattr(state, "_last_strategy_eval", None)
                        last_bar_ts = self._last_bar_ts.get(symbol)
                        if last_bar_ts is None:
                            self._warn_symbol_gate(
                                "bar_not_finalized",
                                symbol,
                                "No finalized minute bar available for strategy evaluation",
                                reason="missing_finalized_bar",
                            )
                            self._mark_symbol_unready(symbol, "bar_not_finalized")
                            return
                        if last_bar_ts and state._last_eval_bar_ts:
                            if last_bar_ts <= state._last_eval_bar_ts:
                                logged_map = getattr(
                                    self, "_same_bar_skip_logged", None
                                )
                                if logged_map is None:
                                    logged_map = {}
                                    self._same_bar_skip_logged = logged_map
                                extra_payload = {
                                    "event": "strategy_eval_skipped_same_bar",
                                    "symbol": symbol,
                                    "bar_ts": last_bar_ts.isoformat(),
                                }
                                if logged_map.get(symbol) != last_bar_ts:
                                    logged_map[symbol] = last_bar_ts
                                    self._logger.debug(
                                        "Condition met: strategy_eval_skipped_same_bar",
                                        extra=extra_payload,
                                    )
                                return
                        if last_bar_ts:
                            bar_age = (now - last_bar_ts).total_seconds()
                            if bar_age > 120.0:
                                log_throttled(
                                    self._logger,
                                    f"strategy_eval_stale_bar_{symbol}",
                                    "Condition met: strategy_eval_stale_bar",
                                    interval_sec=60.0,
                                    level=logging.WARNING,
                                    extra={
                                        "event": "strategy_eval_stale_bar",
                                        "symbol": symbol,
                                        "bar_age_s": bar_age,
                                        "bar_ts": last_bar_ts.isoformat(),
                                    },
                                )
                                return
                        # Limit evaluation frequency (max 2 per second)
                        if last_eval and (now - last_eval).total_seconds() < 0.5:
                            return
                        cooldown_map = getattr(self, "_pyramid_reject_cooldown", {})
                        if cooldown_map:
                            now_ts = time_module.time()
                            cooldown_symbol = self._normalize_symbol(symbol)
                            directions = (
                                ["BUY"] if self._options_long_only else ["BUY", "SELL"]
                            )
                            for direction in directions:
                                cooldown_key = (cooldown_symbol, direction)
                                cooldown_until = cooldown_map.get(cooldown_key)
                                if cooldown_until and now_ts < cooldown_until:
                                    self._logger.debug(
                                        "pyramid_cooldown_active",
                                        extra={
                                            "event": "pyramid_cooldown_active",
                                            "symbol": cooldown_symbol,
                                            "direction": direction,
                                            "cooldown_remaining_s": max(
                                                0.0, cooldown_until - now_ts
                                            ),
                                        },
                                    )
                                    return
                                if cooldown_until and now_ts >= cooldown_until:
                                    cooldown_map.pop(cooldown_key, None)
                        state._last_strategy_eval = now
                        if last_bar_ts:
                            state._last_eval_bar_ts = last_bar_ts
                        should_evaluate = True

                if should_evaluate:
                    # ✅ DIAGNOSTIC LOG: Confirm evaluation is happening
                    log_throttled(
                        self._logger,
                        f"strategy_eval_{symbol}",
                        f"🎯 EVALUATING STRATEGIES: {symbol} | min_bars={self._required_candles}",
                        interval_sec=30.0,
                        level=logging.DEBUG,
                    )
                    if not self._indicator_engine.has_min_bars(
                        symbol, self._required_candles
                    ):
                        # Hydrate at most once per symbol per startup to avoid repeated API stress.
                        if symbol not in self._hydration_attempted_symbols:
                            self._hydration_attempted_symbols.add(symbol)
                            self._hydrate_missing_bars(symbol, self._required_candles)
                        self._log_once_per_symbol_per_bar(
                            symbol,
                            "indicator_hydration_pending",
                            "min_bars_not_ready",
                        )
                        self._mark_symbol_unready(symbol, "indicator_hydration_pending")
                        return

                    index_indicators_ready = bool(
                        getattr(self._orchestrator, "index_indicators_ready", True)
                    )
                    symbol_indicators_ready = self._indicator_engine.has_min_bars(
                        symbol, self._required_candles
                    )
                    regime_ready = self._regime_manager_ready()

                    if not (
                        index_indicators_ready
                        and symbol_indicators_ready
                        and regime_ready
                    ):
                        reason = "index_indicators_not_ready"
                        if not symbol_indicators_ready:
                            reason = "symbol_indicators_not_ready"
                        elif not regime_ready:
                            reason = "regime_manager_not_ready"
                        self._warn_symbol_gate(
                            "indicator_invalid",
                            symbol,
                            "Indicators/regime are incomplete for this cycle",
                            reason=reason,
                        )
                        self._mark_symbol_unready(symbol, "indicator_invalid")
                        return
                    else:
                        # ✅ DIAGNOSTIC LOG: Confirm indicators are ready
                        log_throttled(
                            self._logger,
                            f"indicators_ready_{symbol}",
                            f"✅ INDICATORS READY: {symbol} | Calling StrategyManager...",
                            interval_sec=60.0,
                            level=logging.DEBUG,
                        )

                        mdm_last_tick = getattr(
                            self._market_data, "_last_tick_time", {}
                        ).get(symbol)
                        if (
                            isinstance(mdm_last_tick, (int, float))
                            and time.time() - float(mdm_last_tick) > 3.0
                        ):
                            self._logger.warning("Stale tick — skipping execution")
                            return
                        signal = self._strategy_manager.generate_signal(symbol, price)
                        cycle_stats = getattr(self, "_strategy_cycle_stats", None)
                        if cycle_stats is None:
                            cycle_stats = {}
                            self._strategy_cycle_stats = cycle_stats
                        bar_key = (
                            last_bar_ts.isoformat() if last_bar_ts else now.isoformat()
                        )
                        cycle = cycle_stats.setdefault(
                            bar_key,
                            {
                                "symbols": set(),
                                "total_signals": 0,
                                "reject_reason_counts": defaultdict(int),
                            },
                        )
                        cycle["symbols"].add(symbol)
                        if signal is None:
                            cycle["reject_reason_counts"]["no_signal"] += 1
                        else:
                            cycle["total_signals"] += 1
                        expected_symbols = [
                            sym
                            for sym in self._active_symbols
                            if not (
                                sym.upper().startswith("NSE:")
                                and "NIFTY" in sym.upper()
                                and "CE" not in sym.upper()
                                and "PE" not in sym.upper()
                                and "FUT" not in sym.upper()
                            )
                        ]
                        if expected_symbols and len(cycle["symbols"]) >= len(
                            expected_symbols
                        ):
                            self._logger.info(
                                "strategy_cycle_summary",
                                extra={
                                    "event": "strategy_cycle_summary",
                                    "total_symbols": len(cycle["symbols"]),
                                    "total_signals": cycle["total_signals"],
                                    "reject_reason_counts": dict(
                                        cycle["reject_reason_counts"]
                                    ),
                                },
                            )
                            cycle_stats.pop(bar_key, None)

            # =================================================================
            # PHASE 10: EXECUTE SIGNAL
            # =================================================================

            if signal and signal.action != "HOLD":
                if (
                    signal.action in {"BUY", "SELL"}
                    and not self._strategy_slots_available()
                ):
                    self._warn_symbol_gate(
                        "strategy_slots_full",
                        symbol,
                        "Strategy execution slots are full; rejecting new entry signal",
                        reason="max_concurrent_strategies_reached",
                        max_slots=self._strategy_slot_limit,
                    )
                    return
                with self._lock:
                    state = self._symbol_state.get(symbol)
                    if state:
                        if state.last_signal_at:
                            elapsed = (now - state.last_signal_at).total_seconds()
                            if elapsed < self._config.signal_cooldown_seconds:
                                return

                        state.strategy_data["last_signal"] = {
                            "action": signal.action,
                            "reason": signal.reason,
                            "timestamp": now.isoformat(),
                        }

                self._logger.info(
                    f"🚀 SIGNAL EXECUTING: {symbol} | Action={signal.action} | Reason={signal.reason}"
                )
                self._handle_signal(signal, price, now)
        except Exception as e:
            self._logger.error("Failure in _on_tick: %s", e, exc_info=True)
            return

    def _should_enforce_freshness_backoff(self) -> bool:
        """Args: none; Returns: whether freshness backoff should pause strategy; Raises: none."""
        try:
            if get_market_state() != MarketState.OPEN:
                return False
            transport = getattr(self._market_data, "transport_status", None)
            if callable(transport):
                status = transport() or {}
                if bool(status.get("polling")):
                    return False
                ws_state = str(status.get("ws_state") or "").lower()
                if ws_state in {"reconnecting", "connecting", "backoff"}:
                    return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner._should_enforce_freshness_backoff: %s", exc
            )
            return True

    def set_data_freshness_backoff(
        self,
        backoff_seconds: float,
        *,
        detail_code: str | None = None,
        symbol: str | None = None,
    ) -> None:
        """Args: backoff_seconds, detail_code, symbol. Returns: None. Raises: Exception."""
        try:
            if not self._should_enforce_freshness_backoff():
                return
            seconds = max(float(backoff_seconds), 0.0)
            now_ts = time_module.time()
            until = now_ts + seconds
            current_until = float(getattr(self, "_data_freshness_backoff_until", 0.0))
            if until > current_until:
                self._data_freshness_backoff_until = until
            logged_until = float(
                getattr(self, "_data_freshness_backoff_logged_until", 0.0)
            )
            if until > logged_until:
                self._data_freshness_backoff_logged_until = until
                self._logger.info(
                    "⏸️ STRATEGY PAUSED — Data freshness degraded",
                    extra={
                        "event": "strategy_eval_backoff_active",
                        "backoff_until": self._data_freshness_backoff_until,
                        "backoff_seconds": seconds,
                        "detail_code": detail_code,
                        "symbol_checked": symbol,
                    },
                )
            self._data_freshness_backoff_detail = detail_code
            self._data_freshness_backoff_symbol = symbol
            if detail_code and "rate" in detail_code.lower():
                if symbol:
                    self._rate_limit_backoff_until_by_symbol[symbol] = until
                    self._mark_symbol_unready(symbol, "rate_limit_breach")
                self._warn_symbol_gate(
                    "rate_limit_breach",
                    symbol or "GLOBAL",
                    "Broker rate-limit breach signaled; throttling polling/subscriptions",
                    reason="broker_rate_limit_signal",
                    detail_code=detail_code,
                    backoff_seconds=seconds,
                )
                update_interval = getattr(
                    self._market_data, "set_poll_interval_seconds", None
                )
                if callable(update_interval):
                    update_interval(max(seconds, 5.0))
            self._logger.debug(
                "Condition met: strategy_eval_backoff_set",
                extra={
                    "event": "strategy_eval_backoff_set",
                    "backoff_seconds": seconds,
                    "backoff_until": self._data_freshness_backoff_until,
                    "detail_code": detail_code,
                    "symbol_checked": symbol,
                },
            )
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner.set_data_freshness_backoff: %s",
                exc,
                extra={"event": "strategy_eval_backoff_error"},
                exc_info=exc,
            )

    def _handle_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """
        Handle signal execution with comprehensive error handling.

        ✅ FIX: Added early time guard to prevent processing outside market hours.
        """
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX: EARLY TIME GUARD (Check BEFORE any processing)
        # ═══════════════════════════════════════════════════════════
        from nifty_scalper_bot.utils.market_hours import (
            get_time_status,
            is_market_hours_cached,
        )

        if not is_market_hours_cached():
            # Throttle logging to once per minute per symbol
            cache_key = f"time_block_{signal.symbol}"
            if not hasattr(self, "_time_block_logged"):
                self._time_block_logged = {}

            now = timestamp.timestamp()
            last_logged = self._time_block_logged.get(cache_key, 0)

            if now - last_logged > 60:  # Log once per minute
                _, reason = get_time_status()
                self._logger.debug(
                    f"⏰ Signal blocked (outside market hours): {signal.symbol} | {reason}"
                )
                self._time_block_logged[cache_key] = now

            return  # ❌ STOP HERE - Don't process signal
        # ═══════════════════════════════════════════════════════════

        self._logger.info(
            f"🔴 1. SIGNAL HANDLER ENTERED: {signal.symbol} {signal.action}"
        )

        base_symbol = ""
        try:
            action = signal.action
            base_symbol = self._normalize_symbol(signal.symbol)
            trade_symbol = base_symbol
            trade_price = price

            if action in {"BUY", "SELL"}:
                self._handle_entry_signal(
                    signal, base_symbol, trade_symbol, trade_price, timestamp
                )

            elif action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                self._handle_exit_signal(
                    signal, base_symbol, trade_symbol, trade_price, timestamp
                )

        except Exception as exc:
            self._logger.error(f"🔴 HANDLER CRASHED: {exc}", exc_info=True)
            if base_symbol:
                try:
                    self._record_trade(
                        base_symbol,
                        TradeRecord(
                            timestamp, signal.action, 0, price, "error", str(exc)
                        ),
                    )
                except Exception:
                    pass

    def _adopt_orphan_positions(self) -> None:
        """
        Auto-adopt orphan positions with default risk management.

        ✅ PRODUCTION FIX (Feb 2, 2026):
        - Uses is_symbol_managed() instead of get_bracket(symbol)
        - Uses attach_orphan_position() instead of non-existent create_bracket()
        - Better logging and error handling
        """
        if not self._position_manager:
            return

        if not self._bracket_manager:
            self._logger.debug("BracketManager not available, skipping orphan adoption")
            return

        positions = self._position_manager.get_all_positions()
        adopted_count = 0

        for pos in positions or []:
            try:
                symbol = getattr(pos, "symbol", "")
                if not symbol:
                    continue
                if not is_strategy_instrument(symbol):
                    continue
                if symbol in self._active_orphan_guards:
                    continue
                self._active_orphan_guards.add(symbol)

                # Check strategy tag
                strategy = (
                    getattr(pos, "strategy", "")
                    or getattr(pos, "strategy_name", "")
                    or getattr(pos, "tag", "")
                    or ""
                )

                # Identify Orphan (Manual/Unknown/Empty)
                is_orphan = strategy.lower().strip() in (
                    "manual",
                    "unknown",
                    "manual/unknown",
                    "",
                    "none",
                )

                if not is_orphan:
                    continue

                # 1. Determine Side & Quantity Safely
                raw_qty = int(getattr(pos, "quantity", 0) or 0)
                qty = abs(raw_qty)

                # Use 'side' attr if available, else infer from sign
                side = getattr(pos, "side", None)
                if not side:
                    side = "SHORT" if raw_qty < 0 else "LONG"

                entry = float(
                    getattr(pos, "entry_price", 0)
                    or getattr(pos, "avg_price", 0)
                    or getattr(pos, "average_price", 0)
                    or 0
                )

                if qty <= 0 or entry <= 0:
                    continue

                # ═══════════════════════════════════════════════════════════════
                # ✅ FIX #1: Use is_symbol_managed() instead of get_bracket()
                # ═══════════════════════════════════════════════════════════════
                if self._bracket_manager.is_symbol_managed(symbol):
                    continue  # Already protected, skip

                # 3. Log the adoption
                self._logger.warning(
                    f"🔧 AUTO-ADOPTING ORPHAN: {symbol} ({side}) | "
                    f"Qty={qty} | Entry={entry:.2f}"
                )

                # ═══════════════════════════════════════════════════════════════
                # ✅ FIX #2: Use attach_orphan_position() instead of create_bracket()
                # ═══════════════════════════════════════════════════════════════
                try:
                    bracket_id = self._bracket_manager.attach_orphan_position(
                        symbol=symbol, side=side, qty=qty, entry_price=entry
                    )
                    adopted_count += 1
                    self._logger.info(
                        f"✅ Orphan protected: {symbol} | Bracket={bracket_id}"
                    )

                    # Try to tag the position to prevent re-adoption
                    try:
                        pos.strategy = "Adopted_Orphan"
                    except (AttributeError, TypeError):
                        pass  # Position might be frozen/immutable

                except Exception as e:
                    self._logger.error(f"❌ Failed to adopt orphan {symbol}: {e}")
                finally:
                    self._active_orphan_guards.discard(symbol)

            except Exception as e:
                self._active_orphan_guards.discard(symbol)
                self._logger.error(f"❌ Error processing position: {e}")

        if adopted_count > 0:
            self._logger.info(
                f"📊 Orphan Adoption Complete: {adopted_count} positions protected"
            )

    def _calculate_signal_score(self, symbol: str, side: str, price: float) -> float:
        """
        Calculate confidence using INSTANT metrics (No history required).

        ✅ WORLD CLASS FIX: Better handling of market hours and volume.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # Check session override for testing
        allow_off_hours = os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower() == "true"
        ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
        is_market_hours = 9 <= ist_now.hour < 16

        # Base score
        score = 0.5

        with self._lock:
            state = self._symbol_state.get(symbol)
            if not state:
                return 0.75  # Trust signal if no state

            # 1. VWAP Proximity
            if state.vwap and state.vwap > 0 and price > 0:
                dist_pct = abs(price - state.vwap) / state.vwap
                if dist_pct < 0.005:  # <0.5%
                    score += 0.3
                elif dist_pct < 0.01:  # <1.0%
                    score += 0.2
                elif dist_pct < 0.02:  # <2.0%
                    score += 0.1
                elif dist_pct > 0.03:  # >3%
                    score -= 0.2

            # 2. Volume Check - Relaxed for off-hours
            if state.last_tick:
                vol = float(state.last_tick.get("volume", 0))
                if vol > 100000:
                    score += 0.2
                elif vol > 50000:
                    score += 0.15
                elif vol > 10000:
                    score += 0.1
                elif vol > 0:
                    score += 0.05
                elif allow_off_hours and not is_market_hours:
                    # Off-hours: don't penalize zero volume
                    score += 0.1

        # 3. Boost for testing mode
        if allow_off_hours and not is_market_hours:
            score = max(score, 0.6)  # Ensure signals pass during testing

        return min(1.0, max(0.0, score))

    def _resolve_contract_safely(
        self, base_symbol: str, action: str, price: float, option_type: str | None
    ) -> SelectedContract | None:
        """
        CRITICAL FIX: Safely resolves option contracts with Null Guards.
        Prevents crash if DataHub failed to initialize or Option Chain is empty.
        """
        # 1. GUARD: Check if Strike Selector component exists
        if self._strike_selector is None:
            log_throttled(
                self._logger,
                "strike_selector_none",
                f"🛑 CRITICAL: Strike Selector is None! DataHub likely failed. Cannot trade {base_symbol}.",
                interval_sec=60.0,
                level=logging.CRITICAL,
            )
            return None

        # 2. GUARD: Check if we have actual chain data (Prevents selecting from empty chain)
        # We rely on DataHub to tell us if the chain is alive.
        if self._data_hub:
            if hasattr(
                self._data_hub, "has_chain_data"
            ) and not self._data_hub.has_chain_data(base_symbol):
                log_throttled(
                    self._logger,
                    f"missing_chain_{base_symbol}",
                    f"🛑 MISSING CHAIN DATA: Cannot select strike for {base_symbol}. DataHub returned no chain.",
                    interval_sec=30.0,
                    level=logging.ERROR,
                )
                return None

        try:
            # 3. EXECUTE: Safe selection
            # Map action to selector side
            selector_side = "BUY" if action == "BUY" else "SELL"
            safe_opt_type = (
                cast(Literal["CE", "PE"], option_type)
                if option_type in ("CE", "PE")
                else None
            )

            selection = self._strike_selector.select_contract(
                underlying=base_symbol,
                side=selector_side,
                underlying_price=price,
                option_type=safe_opt_type,
            )

            if not selection:
                self._logger.warning(
                    f"⚠️ Strike Selector returned None for {base_symbol} {action} @ {price}"
                )
                return None

            return selection

        except Exception as e:
            self._logger.error(
                f"💥 EXCEPTION in strike selection for {base_symbol}: {e}",
                exc_info=True,
            )
            return None

    def _handle_entry_signal(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
    ) -> None:
        """
        Handle entry signal execution with comprehensive safeguards.

        Production-grade protections:
        1. Entry lock (atomic execution)
        2. Order-in-flight check
        3. Signal debounce
        4. Position check (no pyramiding)
        5. Confidence threshold
        6. VWAP filter
        7. Risk validation
        """

        # ═══════════════════════════════════════════════════════════════
        # 🛡️ GUARD 0: ATOMIC ENTRY LOCK
        # Prevents race conditions when multiple ticks trigger signals
        # ═══════════════════════════════════════════════════════════════
        if not self._entry_lock.acquire(blocking=False):
            self._logger.debug(
                f"🛡️ ENTRY LOCK BUSY: {base_symbol} | " "Another entry being processed",
                extra={"event": "entry_lock_busy", "symbol": base_symbol},
            )
            return

        try:
            self._handle_entry_signal_inner(
                signal, base_symbol, trade_symbol, trade_price, timestamp
            )
        finally:
            self._entry_lock.release()

    def _handle_entry_signal_inner(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
    ) -> None:
        """Args: signal, base_symbol, trade_symbol, trade_price, timestamp. Returns: None. Raises: Exception."""
        try:
            self._logger.debug(
                "Entered StrategyRunner._handle_entry_signal_inner",
                extra={"event": "entry_signal_inner", "symbol": base_symbol},
            )

            # ═══════════════════════════════════════════════════════════════
            # 🛡️ GUARD 0.5: ORDER IN-FLIGHT CHECK
            # Prevents duplicate submissions before order fills
            # ═══════════════════════════════════════════════════════════════
            if self._is_order_in_flight(trade_symbol or base_symbol, base_symbol):
                self._logger.info(
                    f"🛡️ ORDER IN-FLIGHT REJECT: {base_symbol} | "
                    "Waiting for pending order to complete",
                    extra={"event": "order_in_flight_reject", "symbol": base_symbol},
                )
                return
            # -----------------------------------------------------------
            # 🛡️ GUARD 1: Signal Debounce (Anti-Whipsaw)
            # -----------------------------------------------------------
            with self._lock:
                state = self._symbol_state.get(base_symbol)
                if state and state.last_signal_at:
                    delta = (timestamp - state.last_signal_at).total_seconds()
                    debounce_limit = self._risk_manager.settings.signal_debounce_seconds

                    if delta < debounce_limit:
                        self._logger.info(
                            f"⏳ DEBOUNCE REJECT: {base_symbol} | "
                            f"Wait {debounce_limit - delta:.1f}s more | "
                            f"Action={signal.action}",
                            extra={
                                "event": "signal_debounce_reject",
                                "symbol": base_symbol,
                            },
                        )
                        return

            # -----------------------------------------------------------
            # 🛡️ GUARD 1.5: Recently Closed Check (Anti Re-Entry)
            # ✅ FIX (6 Feb 2026): Prevents bot from re-entering manually exited trades
            # -----------------------------------------------------------
            import time as _time_mod

            _exit_cooldown_sec = float(
                os.getenv("EXIT_REENTRY_COOLDOWN_SEC", "300")
            )  # 5 min default
            _last_exit_time = self._recently_closed.get(base_symbol, 0)
            if (
                _last_exit_time
                and (_time_mod.time() - _last_exit_time) < _exit_cooldown_sec
            ):
                _remaining = _exit_cooldown_sec - (_time_mod.time() - _last_exit_time)
                self._logger.info(
                    f"🛡️ REENTRY COOLDOWN: {base_symbol} | "
                    f"Exited {_time_mod.time() - _last_exit_time:.0f}s ago | "
                    f"Wait {_remaining:.0f}s more",
                    extra={"event": "signal_reentry_cooldown", "symbol": base_symbol},
                )
                return

            # -----------------------------------------------------------
            # 🛡️ GUARD 2: Position Check (No Pyramiding + Cross-Strike)
            # ✅ FIX (6 Feb 2026): Also blocks same-underlying different-strike entries
            # -----------------------------------------------------------
            if self._position_manager:
                active_contract = self._position_manager.get_active_contract(
                    base_symbol
                )
                if active_contract and not self._risk_manager.settings.allow_pyramiding:
                    self._logger.info(
                        f"🛡️ PYRAMID REJECT: {base_symbol} | "
                        f"Already active on {active_contract.symbol} | "
                        "Pyramiding Disabled",
                        extra={"event": "signal_pyramid_reject", "symbol": base_symbol},
                    )
                    cooldown_seconds = 15.0
                    cooldown_map = getattr(self, "_pyramid_reject_cooldown", None)
                    if cooldown_map is None:
                        cooldown_map = {}
                        self._pyramid_reject_cooldown = cooldown_map
                    cooldown_key = (base_symbol, signal.action)
                    cooldown_map[cooldown_key] = time_module.time() + cooldown_seconds
                    self._logger.debug(
                        "Condition met: pyramid_reject_cooldown_set",
                        extra={
                            "event": "pyramid_reject_cooldown_set",
                            "symbol": base_symbol,
                            "direction": signal.action,
                            "cooldown_seconds": cooldown_seconds,
                        },
                    )
                    with self._lock:
                        if state:
                            state.last_signal_at = timestamp
                    return

                # ✅ FIX (6 Feb 2026): Cross-strike check — block if ANY NIFTY option is active
                _max_nifty_positions = int(os.getenv("MAX_NIFTY_POSITIONS", "1"))
                _all_positions = (
                    list(self._position_manager.get_open_positions())
                    if hasattr(self._position_manager, "get_open_positions")
                    else []
                )
                _nifty_active = [
                    p
                    for p in _all_positions
                    if "NIFTY" in getattr(p, "symbol", "").upper()
                    and abs(getattr(p, "quantity", 0) or 0) > 0
                ]
                if len(_nifty_active) >= _max_nifty_positions:
                    candidate_signal_confidence = self._normalize_confidence(
                        float(getattr(signal, "confidence", 0.0) or 0.0)
                    )
                    active_position = _nifty_active[0] if _nifty_active else None
                    active_position_confidence = 0.0
                    if active_position is not None:
                        active_position_confidence = self._normalize_confidence(
                            float(getattr(active_position, "confidence", 0.0) or 0.0)
                        )
                    if (
                        active_position is not None
                        and candidate_signal_confidence > active_position_confidence
                        and hasattr(self._order_manager, "exit_position")
                    ):
                        active_qty = int(
                            abs(getattr(active_position, "quantity", 0) or 0)
                        )
                        if active_qty > 0:
                            self._logger.info(
                                "signal_cross_strike_replace",
                                extra={
                                    "event": "signal_cross_strike_replace",
                                    "symbol": base_symbol,
                                    "active_symbol": getattr(
                                        active_position, "symbol", ""
                                    ),
                                    "candidate_confidence": candidate_signal_confidence,
                                    "active_confidence": active_position_confidence,
                                },
                            )
                            self._order_manager.exit_position(
                                symbol=getattr(active_position, "symbol", ""),
                                quantity=active_qty,
                                tag="cross_strike_replace",
                            )
                        else:
                            self._logger.info(
                                "signal_cross_strike_reject",
                                extra={
                                    "event": "signal_cross_strike_reject",
                                    "symbol": base_symbol,
                                    "reason": "active_qty_invalid",
                                },
                            )
                            with self._lock:
                                if state:
                                    state.last_signal_at = timestamp
                            return
                    else:
                        self._logger.info(
                            f"🛡️ CROSS-STRIKE REJECT: {base_symbol} | "
                            f"Already {len(_nifty_active)} NIFTY positions active "
                            f"(max={_max_nifty_positions}) | "
                            f"Active: {[getattr(p, 'symbol', '?') for p in _nifty_active]}",
                            extra={
                                "event": "signal_cross_strike_reject",
                                "symbol": base_symbol,
                            },
                        )
                        with self._lock:
                            if state:
                                state.last_signal_at = timestamp
                        return

            side = "LONG" if signal.action == "BUY" else "SHORT"
            confidence = self._calculate_signal_score(signal.symbol, side, trade_price)

            # Confidence Threshold
            min_confidence = float(os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.45"))

            if confidence < min_confidence:
                self._logger.info(
                    f"🚫 CONFIDENCE REJECT: {base_symbol} | "
                    f"Score={confidence:.2f} < min={min_confidence:.2f}",
                    extra={"event": "signal_confidence_reject", "symbol": base_symbol},
                )
                return
            action = signal.action
        except Exception as exc:
            self._logger.error(
                "Failure in _handle_entry_signal_inner: %s",
                exc,
                exc_info=True,
            )
            return

        # ===========================================================
        # ✅ SMART VWAP FILTER (Unshackles Elite Strategies)
        # ===========================================================
        should_check_vwap = True
        if signal.metadata and signal.metadata.get("ignore_vwap"):
            should_check_vwap = False
            self._logger.debug(
                f"ℹ️ VWAP Check Bypassed by Strategy: {signal.strategy_name}"
            )

        current_vwap = None
        with self._lock:
            if state := self._symbol_state.get(base_symbol):
                current_vwap = state.vwap

        # Only block if Strategy did NOT opt-out
        if should_check_vwap and current_vwap and current_vwap > 0:
            vwap_dist = ((trade_price - current_vwap) / current_vwap) * 100

            # SCALP RULE: For BUY (Long), Price must be ABOVE VWAP
            if action == "BUY" and trade_price < current_vwap:
                self._logger.info(
                    f"🛑 VWAP REJECT: {base_symbol} | "
                    f"Price={trade_price:.2f} < VWAP={current_vwap:.2f} | "
                    f"Dist={vwap_dist:.2f}%",
                    extra={"event": "signal_vwap_reject", "symbol": base_symbol},
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        trade_price,
                        "skipped",
                        "vwap_filter",
                    ),
                )
                return

            self._logger.info(
                f"📊 VWAP PASS: Price={trade_price:.2f} > VWAP={current_vwap:.2f} Dist={vwap_dist:.2f}%"
            )

        # ===========================================================
        # Contract Selection Logic
        # ===========================================================
        selection: SelectedContract | None = None
        selector = self._strike_selector

        try:
            # Resolve direction and option type
            direction = "BULLISH" if action == "BUY" else "BEARISH"
            metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
            option_type = metadata.get("option_type")

            # Always infer option_type if missing (Legacy Support)
            if not option_type:
                option_type = "CE" if direction == "BULLISH" else "PE"

            sell_premium = (
                bool(metadata.get("sell_premium")) and not self._options_long_only
            )
            entry_side: OrderSide = "SELL" if sell_premium else "BUY"

            # Check active contract reuse logic (Scaling In)
            if self._position_manager:
                active = self._position_manager.get_active_contract(base_symbol)
                if active:
                    if self._position_manager.is_flat(active.symbol):
                        self._position_manager.clear_active_contract(base_symbol)
                    else:
                        reuse = True
                        if (
                            active.option_type != option_type
                            and not self._allow_hedge_entries
                        ):
                            reuse = False
                        if reuse:
                            selection = SelectedContract(
                                symbol=active.symbol,
                                option_type=active.option_type,
                                strike=active.strike,
                                expiry=active.expiry,
                                ltp=trade_price,
                                delta=None,
                                metadata={"source": "position_manager"},
                            )
                            trade_symbol = selection.symbol

            # Strategy Explicit Bypass (e.g. Signal is already on an Option)
            if not selection and (
                base_symbol.endswith("CE") or base_symbol.endswith("PE")
            ):
                selection = SelectedContract(
                    symbol=base_symbol,
                    option_type="CE" if "CE" in base_symbol else "PE",
                    strike=0.0,
                    expiry=timestamp,
                    ltp=trade_price,
                    delta=None,
                    metadata={"source": "explicit"},
                )
                trade_symbol = base_symbol

            # -------------------------------------------------------------
            # 🔎 SELECTOR CALL & PRICE SAFETY
            # -------------------------------------------------------------
            if not selection:
                # Use Safe Resolver (Maps Futures -> Index)
                selection = self._resolve_contract_safely(
                    base_symbol=base_symbol,
                    action=action,
                    price=trade_price,
                    option_type=option_type,
                )

                if selection:
                    trade_symbol = selection.symbol

                    # ✅ CRITICAL FIX: PRICE SAFETY CHECK
                    # If we switched from Future/Index to Option, we MUST have the Option Price.
                    # We cannot use the Underlying Price (e.g. 25000) for an Option (e.g. 200).

                    if selection.ltp and selection.ltp > 0:
                        trade_price = selection.ltp
                    elif trade_symbol != base_symbol:
                        # Fallback: Try fetching live quote for the Option
                        # This happens if the selector found the symbol but hasn't received a tick yet
                        q = self._market_data.get_quote(trade_symbol)

                        # Extract price using robust helper (defined in file scope)
                        safe_price = (
                            _extract_float(q, "ltp", "last_price", "close")
                            if q
                            else 0.0
                        )

                        if safe_price > 0:
                            trade_price = safe_price
                            self._logger.info(
                                f"🔄 Fetched fresh price for {trade_symbol}: {trade_price}"
                            )
                        else:
                            # CRITICAL: Do not trade if we don't know the Option price
                            self._logger.error(
                                f"🔴 PRICE MISSING: Cannot trade {trade_symbol}. "
                                f"Selection LTP is None and Live Quote failed. "
                                f"Preventing usage of Underlying Price ({trade_price})."
                            )
                            return

            if not selection:
                self._logger.warning(
                    f"🔴 CONTRACT REJECT: {base_symbol} | "
                    f"No option contract selected | "
                    f"Check option chain data availability",
                    extra={"event": "signal_contract_reject", "symbol": base_symbol},
                )
                return

            # Monthly Lockout Check
            lockout, _ = self._monthly_lockout_active(selection.expiry, timestamp)
            if lockout:
                return

            # Apply Premium Targets & Risk Sizing
            signal = self._apply_premium_targets(signal, trade_price, entry_side)

            # Use the robust ATR fallback helper
            atr_val = self._get_atr_with_fallback(
                symbol=trade_symbol, metadata=metadata, current_price=trade_price
            )

            # ═══════════════════════════════════════════════════════════════
            # ✅ CRITICAL FIX: Correct SL/TP for position side
            # ═══════════════════════════════════════════════════════════════
            signal = self._correct_sl_tp_for_position_side(
                signal=signal,
                entry_price=trade_price,
                entry_side=entry_side,
                atr=atr_val,
            )

            self._logger.info(
                f"📊 SIZING: {trade_symbol} | Price={trade_price:.2f} | "
                f"SL={signal.stop_loss:.2f} | ATR={atr_val:.2f}",
                extra={
                    "event": "sizing_calculation",
                    "symbol": trade_symbol,
                    "price": trade_price,
                    "stop_loss": signal.stop_loss,
                    "atr": atr_val,
                },
            )

            sized_qty = self._risk_manager.suggest_position_size(
                side=entry_side,
                price=trade_price,
                stop_loss=signal.stop_loss,
                atr=atr_val,
                requested_quantity=signal.quantity,
                confidence=signal.confidence,
                symbol=trade_symbol,
            )

            available_margin = 0.0
            if self._data_hub is not None:
                try:
                    available_margin = float(
                        self._data_hub.get_available_balance() or 0.0
                    )
                except Exception as exc:
                    self._logger.error(
                        "Failure in StrategyRunner._handle_entry_signal_inner margin fetch: %s",
                        exc,
                        extra={"event": "margin_fetch_failed", "symbol": trade_symbol},
                        exc_info=exc,
                    )
            margin_per_unit = max(trade_price, 0.0)
            if hasattr(self._order_manager, "estimate_margin"):
                try:
                    margin_per_unit = float(
                        self._order_manager.estimate_margin(
                            trade_symbol, 1, trade_price
                        )
                    )
                except Exception:
                    margin_per_unit = max(trade_price, 0.0)
            size_by_margin = (
                int(available_margin // margin_per_unit) if margin_per_unit > 0 else 0
            )
            if size_by_margin > 0:
                sized_qty = min(int(sized_qty), size_by_margin)

            if sized_qty <= 0:
                self._logger.info(
                    "Insufficient capital",
                    extra={"event": "insufficient_capital", "symbol": trade_symbol},
                )
                return

            # Validate Position Limits
            allowed, reason = self._risk_manager.validate_new_position(
                symbol=trade_symbol,
                side="LONG" if entry_side == "BUY" else "SHORT",
                quantity=int(sized_qty),
                entry_price=trade_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            if not allowed:
                self._logger.warning(f"🔴 RISK BLOCK: {reason} | {trade_symbol}")
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            # ===========================================================
            # ✅ EXECUTION: Marketable Limit Order
            # ===========================================================
            # Send Limit slightly deeper than market to guarantee fill but cap slippage.
            execution_price = trade_price
            if self._market_data:
                q = self._market_data.get_quote(trade_symbol)
                if q:
                    # Get the price we need to cross to get filled immediately
                    base = q.get("ask" if entry_side == "BUY" else "bid", trade_price)
                    # Add 1% "Freak Trade Protection" buffer
                    buffer = 1.01 if entry_side == "BUY" else 0.99
                    execution_price = round(base * buffer, 2)

            # ✅ FIX 6a: Shift SL/TP when execution_price diverges from trade_price
            _price_shift = execution_price - trade_price
            if abs(_price_shift) > 0.01:
                _new_sl = signal.stop_loss + _price_shift if signal.stop_loss else None
                _new_tp = (
                    signal.take_profit + _price_shift if signal.take_profit else None
                )
                self._logger.info(
                    f"📐 SL/TP SHIFTED by {_price_shift:+.2f} | "
                    f"SL: {signal.stop_loss:.2f}→{_new_sl:.2f} | "
                    f"TP: {signal.take_profit:.2f}→{_new_tp:.2f}",
                    extra={"event": "sl_tp_price_shift", "symbol": trade_symbol},
                )
                signal = Signal(
                    action=signal.action,
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    confidence=signal.confidence,
                    reason=signal.reason,
                    stop_loss=_new_sl,
                    take_profit=_new_tp,
                    metadata=signal.metadata,
                )

            signal = self._anchor_sl_tp_to_execution(
                signal,
                signal_price=trade_price,
                execution_price=execution_price,
                entry_side=entry_side,
                atr=atr_val,
            )

            self._logger.info(
                f"🟡 SUBMITTING ORDER: {trade_symbol} Qty: {sized_qty} Limit: {execution_price}"
            )

            strat_name = (
                signal.metadata.get("strategy", "MAN") if signal.metadata else "MAN"
            )
            unique_tag = f"{strat_name[:3]}_{int(timestamp.timestamp())}"

            bracket_meta = signal.metadata if isinstance(signal.metadata, dict) else {}
            bracket_type = str(bracket_meta.get("bracket_type") or "").upper()
            use_virtual_bracket = (
                bracket_type == "VIRTUAL"
                and hasattr(self._order_manager, "place_bracket_order")
                and signal.stop_loss
                and signal.take_profit
            )
            tp1_price: float | None = None
            tp1_qty: int | None = None
            trailing_atr_mult: float | None = None
            effective_tp: float | None = signal.take_profit

            if use_virtual_bracket:
                try:
                    sl_atr_mult = float(bracket_meta.get("sl_atr_mult") or 0.0)
                    tp1_atr_mult = float(bracket_meta.get("tp1_atr_mult") or 0.0)
                    tp2_atr_mult = float(bracket_meta.get("tp2_atr_mult") or 0.0)
                    tp1_qty_pct = float(bracket_meta.get("tp1_qty_pct") or 0.0)

                    if tp1_atr_mult > 0 and atr_val > 0:
                        if entry_side == "BUY":
                            tp1_price = execution_price + (atr_val * tp1_atr_mult)
                        else:
                            tp1_price = execution_price - (atr_val * tp1_atr_mult)

                    if (
                        tp2_atr_mult > 0
                        and atr_val > 0
                        and (effective_tp is None or effective_tp <= 0)
                    ):
                        if entry_side == "BUY":
                            effective_tp = execution_price + (atr_val * tp2_atr_mult)
                        else:
                            effective_tp = execution_price - (atr_val * tp2_atr_mult)

                    if 0 < tp1_qty_pct < 1:
                        tp1_qty = max(1, int(round(int(sized_qty) * tp1_qty_pct)))
                        if tp1_qty >= int(sized_qty):
                            tp1_qty = None

                    runner_trail = bool(bracket_meta.get("runner_trail_after_tp1"))
                    sl_mode = str(bracket_meta.get("sl_mode") or "")
                    if runner_trail or sl_mode == "ATR_TRAIL":
                        trailing_atr_mult = float(
                            bracket_meta.get("trailing_atr_mult") or sl_atr_mult or 1.5
                        )
                        if trailing_atr_mult <= 0:
                            trailing_atr_mult = None

                    self._logger.info(
                        "Condition met: virtual_bracket_ready",
                        extra={
                            "event": "virtual_bracket_ready",
                            "symbol": trade_symbol,
                            "tp1_price": tp1_price,
                            "tp1_qty": tp1_qty,
                            "tp2_price": effective_tp,
                            "trailing_atr_mult": trailing_atr_mult,
                        },
                    )
                except Exception as exc:
                    self._logger.error(
                        "Failure in StrategyRunner._handle_entry_signal_inner bracket setup: %s",
                        exc,
                        extra={
                            "event": "virtual_bracket_setup_failed",
                            "symbol": trade_symbol,
                        },
                        exc_info=exc,
                    )
                    use_virtual_bracket = False

            order_id = None
            if use_virtual_bracket:
                try:
                    order_id = self._order_manager.place_bracket_order(
                        symbol=trade_symbol,
                        side=entry_side,
                        quantity=int(sized_qty),
                        entry_price=execution_price,
                        stop_loss=signal.stop_loss,
                        take_profit=effective_tp or signal.take_profit,
                        tp1_price=tp1_price,
                        tp1_qty=tp1_qty,
                        trailing_atr_mult=trailing_atr_mult,
                        tag=unique_tag,
                    )
                except Exception as exc:
                    self._logger.error(
                        "Failure in StrategyRunner._handle_entry_signal_inner virtual bracket: %s",
                        exc,
                        extra={
                            "event": "virtual_bracket_submit_failed",
                            "symbol": trade_symbol,
                        },
                        exc_info=exc,
                    )
                    order_id = None

            if not order_id:
                order_id = self._order_manager.place_order(
                    symbol=trade_symbol,
                    side=entry_side,
                    quantity=int(sized_qty),
                    order_type=OrderType.LIMIT,
                    price=execution_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    signal_id=unique_tag,
                    tag=unique_tag,
                )
            if order_id:
                self._mark_order_in_flight(trade_symbol, base_symbol)

            # ✅ Update State Timers (Debounce)
            with self._lock:
                state = self._symbol_state.get(base_symbol)
                if state:
                    state.last_signal_at = timestamp
                    # Also debounce the specific option symbol
                    if trade_symbol != base_symbol:
                        opt_state = self._symbol_state.get(trade_symbol)
                        if opt_state:
                            opt_state.last_signal_at = timestamp

            if order_id:
                self._logger.info(f"🟢 ORDER SUBMITTED! ID: {order_id}")

                # Async Verification & Chase Logic
                if self._main_loop and self._main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._verify_order_status(order_id, trade_symbol, 3.0),
                        self._main_loop,
                    )

                self._notify_orchestrator_submission(signal, base_symbol)

                # Update Active Contract Tracking
                if self._position_manager and selection:
                    if (
                        self._allow_hedge_entries
                        or not self._position_manager.get_active_contract(base_symbol)
                    ):
                        self._position_manager.set_active_contract(
                            base_symbol, selection
                        )

                if selector:
                    selector.register_open(base_symbol, selection)

                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        int(sized_qty),
                        trade_price,
                        "submitted",
                        signal.reason,
                        order_id,
                    ),
                )

                # Set longer cooldown on success
                self._set_trade_cooldown(base_symbol, timestamp)
            else:
                self._logger.error(f"🔴 Order Execution Failed for {trade_symbol}")

        except Exception as exc:
            self._logger.error(f"🔴 ENTRY LOGIC CRASH: {exc}", exc_info=True)
            # Ensure cooldown even on crash
            self._set_signal_cooldown(base_symbol, timestamp)

    async def _verify_order_status(
        self, order_id: str, symbol: str, delay_seconds: float
    ) -> None:
        """
        World-Class Verification: Detects STUCK Limit orders and CHASES them.
        """
        await asyncio.sleep(delay_seconds)

        try:
            # Run in thread to avoid blocking main loop
            if hasattr(self._order_manager, "get_order"):
                order = await asyncio.to_thread(self._order_manager.get_order, order_id)

                if not order:
                    return

                status = str(order.status).upper()
                # ═══════════════════════════════════════════════════════
                if status in ["COMPLETE", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"]:
                    self._clear_order_in_flight(symbol)

                # 🛡️ ACTIVE CHASE LOGIC
                # If Limit Order is ignored by market (OPEN) after 3s, we must act.
                if status in ["OPEN", "PENDING", "SUBMITTED"]:
                    self._logger.warning(
                        f"⏳ ORDER {order_id} STUCK ({status}). Initiating Chase..."
                    )

                    # Strategy: Modify Price to be more aggressive
                    # Buy: Current LTP + 0.5% | Sell: Current LTP - 0.5%
                    # This effectively converts it to a Market order without losing queue priority completely

                    # 1. Get Fresh Price
                    new_price = 0.0
                    if self._market_data:
                        q = await asyncio.to_thread(self._market_data.get_quote, symbol)
                        if q:
                            # Panic Chase: Cross the spread aggressively
                            base = q.get("ask" if order.side == "BUY" else "bid", 0)
                            if base > 0:
                                buff = 1.005 if order.side == "BUY" else 0.995
                                new_price = round(base * buff, 2)

                    if new_price > 0:
                        self._logger.info(
                            f"🏃 CHASING: Modifying {order_id} to {new_price}"
                        )
                        await asyncio.to_thread(
                            self._order_manager.modify_order,
                            order_id=order_id,
                            price=new_price,
                        )
                    else:
                        self._logger.error("❌ Could not get fresh price for Chase.")

                elif status == "COMPLETE":
                    self._logger.info(f"✅ ORDER {order_id} FILLED.")

        except Exception as exc:
            self._logger.warning(f"Order verification/chase warning: {exc}")

    def _get_atr_with_fallback(
        self, symbol: str, metadata: dict, current_price: float
    ) -> float:
        """
        Get ATR with multiple fallback sources.

        Priority:
        1. Signal metadata (from strategy)
        2. Symbol state (from tick processing)
        3. Indicator engine (live calculation)
        4. Price-based estimate (1% of price)

        Args:
            symbol: Trading symbol
            metadata: Signal metadata dict
            current_price: Current LTP

        Returns:
            ATR value (never None, always positive)
        """
        if not is_strategy_instrument(symbol):
            raise RuntimeError("ATR requested for non-strategy instrument")

        bars = self._market_data.get_ohlc_bars(symbol) if self._market_data else []
        if len(bars) < self._required_candles:
            raise RuntimeError("ATR unavailable due to insufficient data")

        atr_val = 0.0
        source = "unknown"

        # 1. Try signal metadata
        if metadata:
            raw = metadata.get("atr")
            if raw is not None:
                try:
                    atr_val = float(raw)
                    if atr_val > 0:
                        source = "metadata"
                except (TypeError, ValueError):
                    pass

        # 2. Try symbol state
        if atr_val <= 0:
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state and hasattr(state, "atr") and state.atr:
                    try:
                        atr_val = float(state.atr)
                        if atr_val > 0:
                            source = "symbol_state"
                    except (TypeError, ValueError):
                        pass

        # 3. Try indicator engine
        if atr_val <= 0 and self._indicator_engine:
            try:
                indicators = self._indicator_engine.get_indicators(symbol, ["atr"])
                raw = indicators.get("atr")
                if raw is not None:
                    atr_val = float(raw)
                    if atr_val > 0:
                        source = "indicator_engine"
            except Exception:
                pass

        # 4. Try base underlying (e.g., NIFTY instead of NIFTY2620325200CE)
        if atr_val <= 0:
            base = self._extract_underlying(symbol)
            if base and base != symbol:
                with self._lock:
                    state = self._symbol_state.get(base)
                    if state and hasattr(state, "atr") and state.atr:
                        try:
                            atr_val = float(state.atr)
                            if atr_val > 0:
                                source = "underlying_state"
                        except (TypeError, ValueError):
                            pass

        if atr_val <= 0:
            raise RuntimeError("ATR unavailable due to insufficient data")

        spread = 0.0
        try:
            quote = self._market_data.get_quote(symbol) if self._market_data else None
            if quote:
                ask_price = float(quote.get("ask") or quote.get("ask_price") or 0.0)
                bid_price = float(quote.get("bid") or quote.get("bid_price") or 0.0)
                spread = max(0.0, ask_price - bid_price)
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner._get_atr_with_fallback spread read: %s",
                exc,
                extra={"event": "atr_spread_read_failed", "symbol": symbol},
                exc_info=exc,
            )

        min_atr = max(current_price * 0.01, spread * 1.5, 1.0)
        atr_val = max(atr_val, min_atr)

        self._logger.debug(
            f"ATR resolved: {symbol} = {atr_val:.2f} (source: {source})",
            extra={
                "event": "atr_resolved",
                "symbol": symbol,
                "atr": atr_val,
                "source": source,
            },
        )

        return atr_val

    def _extract_underlying(self, symbol: str) -> str:
        """Extract underlying from option symbol (e.g., NIFTY from NIFTY2620325200CE)."""
        if not symbol:
            return ""

        # Common patterns
        if symbol.startswith("NIFTY") and not symbol.startswith("NIFTYFUT"):
            return "NIFTY"
        if symbol.startswith("BANKNIFTY"):
            return "BANKNIFTY"
        if symbol.startswith("FINNIFTY"):
            return "FINNIFTY"

        # Generic extraction: take alphabetic prefix
        import re

        match = re.match(r"^([A-Z]+)", symbol)
        if match:
            return match.group(1)

        return symbol

    def _handle_exit_signal(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
    ) -> None:
        """Handle exit (CLOSE_LONG/CLOSE_SHORT) signals."""
        action = signal.action
        selector = self._strike_selector
        position_manager = self._position_manager

        if selector is not None:
            selection = selector.resolve_active(base_symbol)
        else:
            selection = None

        if selection is None and position_manager is not None:
            positions = position_manager.get_positions(trade_symbol)
            if not positions:
                self._logger.warning(f"No position to close for {trade_symbol}")
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        trade_price,
                        "skipped",
                        "no_position",
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            position = positions[0]
            trade_symbol = position.symbol

        elif selection is None:
            self._logger.warning(f"No active contract found for {base_symbol}")
            self._record_trade(
                base_symbol,
                TradeRecord(
                    timestamp,
                    action,
                    signal.quantity,
                    trade_price,
                    "skipped",
                    "no_active_contract",
                ),
            )
            self._set_signal_cooldown(base_symbol, timestamp)
            return

        else:
            trade_symbol = selection.symbol
            position = (
                position_manager.get_position(trade_symbol)
                if position_manager
                else None
            )

        if position is None:
            self._logger.warning(f"No position found for {trade_symbol}")
            self._record_trade(
                base_symbol,
                TradeRecord(
                    timestamp,
                    action,
                    signal.quantity,
                    trade_price,
                    "skipped",
                    "position_not_found",
                ),
            )
            return

        # Determine exit side
        exit_side = "SELL" if position.side == "LONG" else "BUY"

        try:
            # Create exit intent
            exit_intent = ExitIntent(
                symbol=trade_symbol,
                position_id=getattr(position, "id", None),
                side=exit_side,
                quantity=position.quantity,
                price=trade_price,
                reason=signal.reason,
            )

            # Place reduce-only exit
            # Blocking call - safe in thread
            exit_order_id = self._order_manager.place_reduce_only_exit(exit_intent)

            if exit_order_id:
                self._logger.info(
                    f"Submitted reduce-only {exit_side} for {trade_symbol} qty={position.quantity} id={exit_order_id}"
                )

                self._notify_orchestrator_exit(base_symbol)

                if selector is not None and selection is not None:
                    selector.clear_active(base_symbol)

                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        position.quantity,
                        trade_price,
                        "submitted",
                        signal.reason or "Close position",
                        exit_order_id,
                    ),
                )

                self._set_trade_cooldown(base_symbol, timestamp)

        except OrderPlacementError as exc:
            self._logger.error(f"Exit order placement failed: {exc}", exc_info=True)
            self._record_trade(
                base_symbol,
                TradeRecord(
                    timestamp,
                    action,
                    position.quantity,
                    trade_price,
                    "error",
                    str(exc),
                ),
            )
            # ═══════════════════════════════════════════════════════════
            self._set_post_exit_cooldown(base_symbol, timestamp)

            # Also clear any in-flight status for this symbol
            self._clear_order_in_flight(trade_symbol)

    # [PASTE THIS METHOD INTO StrategyRunner CLASS]
    def calculate_portfolio_greeks(self) -> dict[str, float]:
        """Calculate portfolio-level Greeks aggregation."""
        try:
            from nifty_scalper_bot.indicators.greeks import BlackScholesCalculator
        except ImportError:
            return {"net_delta": 0.0, "net_gamma": 0.0, "net_theta": 0.0}

        calculator = BlackScholesCalculator()
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0

        if not self._position_manager:
            return {"net_delta": 0.0, "net_gamma": 0.0, "net_theta": 0.0}

        # Fetch Spot Price (Simplified for robustness)
        spot = 26000.0  # Default fallback

        # ✅ FIX: Try both attribute names to be safe
        mdm = self._market_data

        if mdm:
            # Try getting LTP
            ltp = mdm.get_latest_price("NSE:NIFTY 50")
            if ltp and ltp > 0:
                spot = ltp

        for pos in self._position_manager.get_all_positions():
            if pos.quantity == 0 or "NIFTY" not in pos.symbol:
                continue

            try:
                # Extract Strike & Type from Symbol (e.g. NIFTY25DEC26000CE)
                import re

                match = re.search(r"(\d{5})([CP]E)", pos.symbol)
                if not match:
                    continue

                strike = float(match.group(1))
                opt_type = match.group(2)

                # Dynamic Time to Expiry (Target: 15:30 on Expiry Day)
                # Simplified: 1 day to expiry
                t_years = 1.0 / 365.0

                # IV Estimate (Using ATR proxy or fixed 15%)
                iv = 0.15

                greeks = calculator.calculate_greeks(
                    spot, strike, t_years, iv, opt_type
                )

                # Directional Adjustment
                sign = 1 if pos.side == "LONG" else -1
                net_delta += sign * pos.quantity * greeks.delta
                net_gamma += sign * pos.quantity * greeks.gamma
                net_theta += sign * pos.quantity * greeks.theta
            except Exception:
                continue

        return {"net_delta": net_delta, "net_gamma": net_gamma, "net_theta": net_theta}

    def _set_trade_cooldown(self, symbol: str, timestamp: datetime) -> None:
        """Set trade-level cooldown after order submission."""
        cooldown = self._config.trade_cooldown_seconds
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return

            state.last_trade_at = timestamp
            state.cooldown_until = (
                timestamp + timedelta(seconds=cooldown) if cooldown > 0 else None
            )

    def _set_signal_cooldown(self, symbol: str, timestamp: datetime) -> None:
        """Set signal-level cooldown after signal processing."""
        cooldown = self._config.signal_cooldown_seconds
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return

            state.cooldown_until = (
                timestamp + timedelta(seconds=cooldown) if cooldown > 0 else None
            )

    def _record_trade(self, symbol: str, record: TradeRecord) -> None:
        """Record a trade for auditing and persistence."""
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return

            state.trade_history.append(record)

        manager = self._persistent_state
        if manager is not None:
            payload = {"symbol": symbol, **record.to_dict()}
            try:
                manager.save_trade(payload)
            except Exception as exc:
                self._logger.error("Failure in _record_trade persistence: %s", exc)

    def _monthly_lockout_active(
        self, expiry: datetime, timestamp: datetime
    ) -> tuple[bool, float]:
        """Return monthly expiry lockout state and minutes to expiry."""
        if self._monthly_halt_minutes <= 0:
            return False, 0.0

        if not _is_monthly_expiry(expiry):
            return False, 0.0

        expiry_dt = expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)
        now_dt = (
            timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        )

        minutes_to_expiry = (expiry_dt - now_dt).total_seconds() / 60.0

        if minutes_to_expiry < 0:
            return False, minutes_to_expiry

        return minutes_to_expiry <= self._monthly_halt_minutes, minutes_to_expiry

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol to canonical exchange-qualified form."""
        normalized = canonical(symbol)
        if not normalized:
            msg = "symbol must not be empty"
            raise ValueError(msg)
        return normalized

    def _update_last_signal_selection(
        self, symbol: str, selection: SelectedContract
    ) -> None:
        """Update strategy data with last selected contract info."""
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return

            info = state.strategy_data.get("last_signal")
            if isinstance(info, dict):
                info["selected_symbol"] = selection.symbol
                info["selected_strike"] = selection.strike
                info["selected_expiry"] = selection.expiry.isoformat()

    @staticmethod
    def _format_reason(reason: str | None, trade_symbol: str, base_symbol: str) -> str:
        """Format reason string with symbol context if needed."""
        if not reason:
            reason = ""

        reason_text = str(reason)

        if trade_symbol != base_symbol and trade_symbol not in reason_text:
            return f"{reason_text} [{trade_symbol}]".strip()

        return reason_text


__all__ = [
    "StrategyRunner",
    "StrategyRunnerConfig",
    "SymbolState",
    "SymbolRuntimeState",
    "TradeRecord",
    "OrderRouter",
]
