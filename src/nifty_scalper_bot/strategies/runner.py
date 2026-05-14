"""Event-driven strategy runner coordinating trading managers."""

from __future__ import annotations

import asyncio
import calendar
from collections import defaultdict, deque
from contextlib import suppress
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from enum import Enum
import json
import inspect
import logging
import os
from pathlib import Path
import re
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
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.core.event_bus import EventBus
from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.trade_manager import TradeManager
from nifty_scalper_bot.core.universe_controller import UniverseController
from nifty_scalper_bot.data.candle_engine import (
    CandleEngine,
    ensure_valid_data,
    normalize_ohlc_timezone,
    repair_with_backfill,
    sanitize,
    validate_dataframe,
)
from nifty_scalper_bot.data.pipeline import (
    MarketDataPipeline,
    get_pipeline,
    MIN_REQUIRED_CANDLES as PIPELINE_MIN_CANDLES,
)

# Assumes you created the data/constants.py file as advised
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.normalizers import normalize_history_row
from nifty_scalper_bot.data.source import (
    DataIntegrityError,
    ensure_ltp,
    is_symbol_valid,
)
# Signals route directly through OrderManager submit/place APIs; no execution hub layer.
from nifty_scalper_bot.execution.order_manager import OrderType, TradePlan
from nifty_scalper_bot.execution.order_state_machine import (
    ExecutionState,
    OrderStateMachine,
)
from nifty_scalper_bot.execution.position_manager import OrderSide, PositionManager
from nifty_scalper_bot.options.strike_selector import SelectedContract, StrikeSelector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.risk.position_sizing import (
    RiskManager as DeterministicRiskManager,
    RiskSnapshot,
)
from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar, OneMinuteBarBuilder
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.market_regime_engine import (
    MarketRegime,
    MarketRegimeEngine,
)
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.strategies.signal_quality import (
    infer_option_side,
    missing_score_components,
    score_signal_quality,
)
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector
from nifty_scalper_bot.utils import metrics
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import LogThrottle, get_logger, log_throttled
from nifty_scalper_bot.utils.market_hours import (
    MarketState,
    allow_offhours_testing_safe,
    get_market_session_state,
    get_market_state,
    is_market_hours_cached,
    is_market_open_now,
    stale_threshold_for_symbol,
)
from nifty_scalper_bot.utils.metrics import Counter, signals_generated_total
from nifty_scalper_bot.utils.symbols import (
    canonical,
    enforce_canonical,
    is_strategy_instrument,
    normalize_symbol,
)

if TYPE_CHECKING:
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.persistent_state import (
        PersistentStateManager,
        TradeDict,
    )

LOGGER = get_logger(__name__)
RELAX_REGIME_FILTER = (
    os.getenv("RELAX_REGIME_FILTER", "true").lower() != "false"
)  # default True: regime starts with no snapshot
MIN_EVAL_INTERVAL_SECONDS = 5.0
_IST = ZoneInfo("Asia/Kolkata")

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read bool env flag. Args: name/default. Returns: bool. Raises: none."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    LOGGER.warning(
        "INVALID_ENV_BOOL name=%s value=%r default=%s",
        name,
        raw,
        default,
        extra={
            "event": "INVALID_ENV_BOOL",
            "name": name,
            "value": raw,
            "default": bool(default),
        },
    )
    return bool(default)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read env var as bool. Args: name/default. Returns: bool. Raises: none."""
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


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


@dataclass(slots=True)
class DeterministicExecutionPipeline:
    """Data->signal->validation->risk->execution pipeline."""

    trade_manager: TradeManager
    risk_manager: DeterministicRiskManager
    order_executor: Any
    log_throttle: LogThrottle

    def on_new_candle(
        self, symbol: str, signal: Signal | None, ltp: float, risk: RiskSnapshot
    ) -> str:
        """Process one closed-candle decision. Args: symbol/signal/ltp/risk. Returns: status. Raises: Exception."""
        if signal is None:
            return "NO_SIGNAL"
        try:
            _ = ensure_ltp(ltp)
            signal = signal.with_metadata(signal_price=ltp, pipeline="deterministic")
            order_id = self.order_executor.execute(signal)
            if not order_id:
                LOGGER.info(
                    '{"event":"SIGNAL_REJECTED","symbol":"%s","reason":"execution_rejected"}',
                    symbol,
                )
                return "REJECTED_EXECUTION"
            return "ORDER_PLACED"
        except DataIntegrityError:
            LOGGER.info(
                '{"event":"SIGNAL_REJECTED","symbol":"%s","reason":"data_integrity"}',
                symbol,
            )
            raise
        except Exception as e:
            LOGGER.exception(
                '{"event":"ORDER_FAILED","symbol":"%s","error":"%s"}',
                symbol,
                e,
                exc_info=True,
            )
            raise


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
        trace_id: str | None = None,
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
class SignalExecutionResult:
    """Structured signal execution outcome. Args: fields. Returns: result. Raises: none."""

    accepted: bool
    reason: str
    order_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRunnerConfig:
    """Configuration controlling runner level behaviour."""

    # Existing
    min_indicator_bars: int = 20
    max_trade_history: int = 100
    fetch_history_on_startup: bool = False

    # ✅ REQUIRED FIX (missing fields)
    signal_cooldown_seconds: float = 3.0
    trade_cooldown_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.min_indicator_bars < 0:
            raise ValueError("min_indicator_bars must be non-negative")

        if self.max_trade_history <= 0:
            raise ValueError("max_trade_history must be positive")

        # ✅ Validation for new fields
        if self.signal_cooldown_seconds < 0:
            raise ValueError("signal_cooldown_seconds must be >= 0")

        if self.trade_cooldown_seconds < 0:
            raise ValueError("trade_cooldown_seconds must be >= 0")


class RunnerState(Enum):
    """State machine for strategy runner lifecycle."""

    STARTING = 1
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
            "last_signal_at": _format_dt(self.last_signal_at),
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
        payload.get("exchange_timestamp")
        or payload.get("timestamp")
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

    while anchor.weekday() != 1:  # FIX S13: Tuesday = 1 (NIFTY expiry day)
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
        datahub=None,
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
        self._execution_engine = None  # Removed — signals route directly via order_manager
        self._logger = get_logger(__name__)
        self._logger.debug(
            "StrategyRunner using MessageBus id=%s", id(self._message_bus)
        )
        # ✅ FIX 1: Ensure 'data' directory exists to prevent Persistence Crash
        try:
            os.makedirs("data", exist_ok=True)
            self._logger.info("✅ Verified 'data/' directory exists for persistence.")
        except Exception as e:
            self._logger.error(f"❌ Failed to create 'data/' directory: {e}")
        self._data_hub = data_hub
        self.datahub = datahub or data_hub
        self._datahub_registered_symbols: set[str] = set()
        self._strike_selector = strike_selector
        self._bracket_manager = bracket_manager
        self._symbol_source: MarketDataManager | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._legacy_tick_subscription_mode = os.getenv(
            "STRATEGY_RUNNER_LEGACY_SUBSCRIBE", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Time block logging throttle
        self._time_block_logged: Dict[str, float] = {}
        self._allow_eval_without_new_bar = (
            os.getenv("RUNNER_ALLOW_EVAL_WITHOUT_NEW_BAR", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._eval_without_new_bar_seconds = max(
            1.0,
            float(os.getenv("RUNNER_EVAL_WITHOUT_NEW_BAR_SECONDS", "15")),
        )
        self._last_same_bar_eval_ts_by_symbol: dict[str, float] = {}
        self._tick_log_throttle_seconds = max(
            1.0,
            float(os.getenv("RUNNER_TICK_LOG_THROTTLE_SECONDS", "120")),
        )
        self._bar_log_throttle_seconds = max(
            1.0,
            float(os.getenv("RUNNER_BAR_LOG_THROTTLE_SECONDS", "120")),
        )
        self._eval_log_throttle_seconds = max(
            1.0,
            float(os.getenv("RUNNER_EVAL_LOG_THROTTLE_SECONDS", "60")),
        )
        self._cooldown_log_throttle_seconds = max(
            1.0,
            float(os.getenv("RUNNER_COOLDOWN_LOG_THROTTLE_SECONDS", "60")),
        )
        self._index_stale_tick_seconds = max(
            1.0,
            float(os.getenv("RUNNER_INDEX_STALE_TICK_SECONDS", "120")),
        )
        self._future_stale_tick_seconds = max(
            1.0,
            float(os.getenv("RUNNER_FUTURE_STALE_TICK_SECONDS", "120")),
        )
        self._option_stale_tick_seconds = max(
            1.0,
            float(os.getenv("RUNNER_OPTION_STALE_TICK_SECONDS", "900")),
        )
        self._generic_stale_tick_seconds = max(
            1.0,
            float(os.getenv("RUNNER_GENERIC_STALE_TICK_SECONDS", "60")),
        )
        self._no_signal_log_throttle_seconds = max(
            1.0,
            float(os.getenv("RUNNER_NO_SIGNAL_LOG_THROTTLE_SECONDS", "300")),
        )
        self._block_low_volatility = (
            os.getenv("RUNNER_BLOCK_LOW_VOLATILITY", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._log_throttle_state: dict[str, float] = {}
        self._first_tick_logged_symbols: set[str] = set()
        self._first_live_bar_logged_symbols: set[str] = set()
        self._symbol_last_signal_ts: dict[str, float] = {}
        self._underlying_last_signal_ts: dict[str, float] = {}
        self._reason_last_signal_ts: dict[str, float] = {}
        self._premium_squeeze_last_signal_ts: dict[str, float] = {}
        self._signal_reject_cooldown_ts: dict[str, float] = {}
        self._order_attempt_window: Deque[float] = deque()
        self._underlying_signal_cooldown_seconds = max(1.0, float(os.getenv("RUNNER_UNDERLYING_SIGNAL_COOLDOWN_SECONDS", "60") or 60))
        self._reason_signal_cooldown_seconds = max(1.0, float(os.getenv("RUNNER_REASON_SIGNAL_COOLDOWN_SECONDS", "120") or 120))
        self._max_order_attempts_per_minute = max(1, int(os.getenv("RUNNER_MAX_ORDER_ATTEMPTS_PER_MINUTE", "3") or 3))
        self._last_execution_halted_log_ts: float = 0.0
        self._runtime_data_hard_ready = False
        self._runtime_evaluation_ready = False
        self._runtime_live_orders_armed = False
        self._runtime_readiness_reason: str | None = None
        self._runtime_startup_ready = False
        self._startup_gate_last_log_ts = 0.0
        self._active_selected_ce: str | None = None
        self._active_selected_pe: str | None = None
        self._active_atm_strike: int | None = None
        self._active_option_symbols: set[str] = set()

        if self._message_bus is None:
            raise RuntimeError("MessageBus not injected into StrategyRunner")
        
        self._logger.info(
            "StrategyRunner initialized with MessageBus: ticks=MDM-callback signals=MessageBus"
        )

        hedge_env = os.getenv("NSB__ALLOW_HEDGE_ENTRIES", "false").strip().lower()
        self._allow_hedge_entries = hedge_env in {"1", "true", "yes", "on"}
        allow_poll_env = os.getenv("ALLOW_POLLING_FALLBACK", "true").strip().lower()
        self._allow_polling_fallback = allow_poll_env in {"1", "true", "yes", "on"}

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
        self._trade_candidate_selector = TradeCandidateSelector()
        self._trade_counter_by_symbol_candle: Dict[str, dict] = {}
        self._settings = get_settings()
        try:
            settings = self._settings
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
        self._eval_lock = threading.Lock()
        self._trade_counter_lock = threading.Lock()
        self._running = False
        self._trading_paused = False
        self.ready = False
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
        self._orders_in_flight: dict[str, float] = {}
        self._orders_lock = threading.RLock()
        self.orders_in_flight: set[str] = set()
        self.order_lock = asyncio.Lock()
        self._order_timeout_sec = 10
        self._execution_state_lock = threading.RLock()
        self._execution_state_by_symbol: dict[str, OrderStateMachine] = {}
        self._event_bus = EventBus()
        self._entry_lock = threading.Lock()  # Atomic entry lock
        self._last_cumulative_volume: dict[str, int] = {}
        self._last_valid_price: dict[str, float] = {}
        self._last_valid_price_ts: dict[str, datetime] = {}
        # Global risk-halt latch keeps control-plane work quiet once breaker trips.
        # We intentionally keep this sticky so per-symbol loops cannot spam checks/logs.
        self._risk_halt_active = False
        self._risk_halt_logged = False
        warmup_bars = 20
        self._required_candles = max(
            warmup_bars,
            self._config.min_indicator_bars,
            int(os.getenv("REQUIRED_CANDLES", str(warmup_bars))),
        )
        self._context_required_bars = max(50, self._required_candles)
        self._option_required_bars = 5
        self._max_symbol_count: int = int(os.getenv("STRATEGY_MAX_SYMBOL_COUNT", "32"))
        self._universe_controller = UniverseController()
        self._universe_dynamic_mode = bool(
            getattr(get_settings(), "universe_dynamic_mode", True)
        )
        self._history_gate_failed: bool = False
        self._backfill_task_started = False
        self._history_ready_by_symbol: dict[str, bool] = {}
        self._required_symbol_count: int = int(os.getenv("REQUIRED_SYMBOL_COUNT", "1"))
        self._symbol_states: dict[str, SymbolState] = {}
        self._symbol_bar_count: dict[str, int] = {}
        self._last_eval_ts: dict[str, float] = defaultdict(float)
        self._eval_gate_lock = threading.Lock()
        self._last_global_eval_ts: float = time.monotonic()
        self._last_tick_seen_ts: float = time.monotonic()
        self._last_tick_time_by_symbol: dict[str, float] = defaultdict(float)
        self._last_tick: dict[str, dict[str, Any]] = {}
        self._symbol_locks: defaultdict[str, threading.Lock] = defaultdict(
            threading.Lock
        )
        self._last_ws_stale_log_ts_by_symbol: dict[str, float] = defaultdict(float)
        self._last_ws_reconnect_attempt_ts: float = 0.0
        self._last_stall_warn_ts: float = 0.0  # throttle stall warnings to 30s
        self._candle_engines: dict[str, CandleEngine] = {}
        # STEP 1/4: Single deterministic pipeline — ticks flow here → closed candles only
        self._pipeline: MarketDataPipeline = get_pipeline(store_maxlen=1500)
        # STEP 5: counter for "invalid data" drops — never silently discarded
        self._invalid_data_skip_counter: int = 0
        self._symbol_history: dict[str, list[OneMinuteBar]] = {}
        self._recent_history_cache: dict[str, list[OneMinuteBar]] = {}
        self._restored_from_cache_symbols: set[str] = set()
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
        self._ingest_lock = asyncio.Lock()
        self._hydration_complete = False
        self._quarantined_symbols: set[str] = set()
        self._session_gap_count: dict[str, int] = {}
        self._runner_state: RunnerState = RunnerState.STARTING
        self._active_orphan_guards: set[str] = set()
        self._orphan_retry_count: dict[str, int] = {}
        self._orphan_retry_last_attempt: dict[str, float] = {}
        self._signals_last_hour: Deque[float] = deque(maxlen=1000)
        self._last_signal_frequency_check_ts: float = 0.0
        self._last_eval_queue_log_ts: float = 0.0
        self._eval_queue_depth = 0
        self._eval_queue_peak = 0
        self._eval_queue_lock = threading.Lock()
        self._eval_in_progress_symbols: set[str] = set()
        self._eval_counter = 0
        self._signal_counter = 0
        self._regime_block_counter = 0
        self._capital_block_counter = 0
        self._last_candle_eval: dict[str, float] = {}
        self._regime_skip_log_ts: dict[str, float] = {}
        self._qty_zero_log_ts: dict[str, float] = {}
        self._last_spot_warn_ts = 0.0
        self._spot_stale_flag = False
        self._last_summary_log = time.monotonic()
        self._last_system_heartbeat_log = time.monotonic()
        self._last_strategy_status_log = time.monotonic()
        self._strategy_window_symbols: set[str] = set()
        self._strategy_window_signals = 0
        self._strategy_window_trailing_updates = 0
        self._candle_versions: dict[str, int] = defaultdict(int)
        self._last_strategy_versions: dict[str, int] = defaultdict(int)
        self._gap_repair_inflight: set[str] = set()
        self._history_refresh_interval_seconds: float = 60.0
        self._last_history_refresh_by_symbol: dict[str, float] = {}
        # BUG W1 FIX: track symbols that have received at least one LIVE (non-backfill)
        # completed bar.  Used by the PHASE-9 stale-bar gate instead of
        # _symbol_history, which is populated by hydration bars (hours old).
        # Without this, has_live_bars=True from day 1 → bar_age=18h >> threshold
        # → stale-bar gate fires on the FIRST tick → PHASE-7 same-bar-skip then
        # blocks all subsequent ticks → zero strategy evaluations until first live
        # bar closes (up to 60 s after startup).
        self._live_bar_seen: set[str] = set()
        self._data_phase: dict[str, str] = {}
        # BUG W2 FIX: emit a one-shot INFO log when indicator warmup first clears
        # for each symbol so Railway logs confirm the moment strategies become active.
        self._warmup_complete_logged: set[str] = set()
        self._max_trades_per_symbol_per_candle = 1
        self._min_trade_interval_seconds = 60.0
        self._session_allow_out_of_hours = allow_offhours_testing_safe()
        self._force_signal_enabled = os.getenv("FORCE_SIGNAL", "").lower() == "true"
        self._disable_early_forced_signals = (
            os.getenv("FEATURE_DISABLE_EARLY_FORCED_SIGNALS", "").lower() == "true"
        )
        self._vwap_crossover_enabled = (
            os.getenv("ENABLE_VWAP_CROSSOVER", "false").lower() == "true"
        )
        self._vwap_sl_pct = float(os.getenv("VWAP_SL_PCT", "1.5"))
        self._vwap_tp_pct = float(os.getenv("VWAP_TP_PCT", "2.0"))
        self._global_min_signal_confidence = float(
            os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.45")
        )
        self._max_nifty_positions = int(os.getenv("MAX_NIFTY_POSITIONS", "1"))
        self._market_regime_engine = MarketRegimeEngine()
        self._last_regime_by_symbol: dict[str, MarketRegime] = {}
        self._last_regime_inputs_by_symbol: dict[str, dict[str, Any]] = {}

        # FIX S10-2: wire bracket-exit callback so direction lock clears on SL/TP
        if self._bracket_manager is not None and hasattr(
            self._bracket_manager, "attach_on_exit_complete"
        ):
            self._bracket_manager.attach_on_exit_complete(
                self._on_bracket_exit_complete
            )

    def _on_bracket_exit_complete(self, symbol: str, *args: Any, **kwargs: Any) -> None:
        """Callback from BracketManager after virtual bracket exit completes; clears runner state only."""
        del args, kwargs
        try:
            if not symbol:
                return

            logger = getattr(self, "_logger", LOGGER)
            logger.info("BRACKET_EXIT_COMPLETE symbol=%s", symbol)

            for attr in ("active_trades", "_active_trades", "open_trades", "_open_trades"):
                store = getattr(self, attr, None)
                if isinstance(store, dict):
                    store.pop(symbol, None)

            for attr in ("_execution_locks", "execution_locks", "_inflight_orders", "inflight_orders"):
                store = getattr(self, attr, None)
                if isinstance(store, dict):
                    store.pop(symbol, None)
                elif isinstance(store, set):
                    store.discard(symbol)

            if getattr(self, "current_symbol", None) == symbol:
                self.current_symbol = None

            try:
                base = self._normalize_symbol(symbol)
            except Exception:
                base = symbol
            self._notify_orchestrator_exit(base)
            self._clear_order_in_flight(symbol)
            self._reset_execution_state(base)

        except Exception:
            logger = getattr(self, "_logger", LOGGER)
            logger.exception("BRACKET_EXIT_CALLBACK_FAILED symbol=%s", symbol)

    # ==================== LIFECYCLE MANAGEMENT ====================



    async def _process_token(self, token, candles, indicators):
        """Process token candles into signals. Args: token, candles, indicators. Returns: None. Raises: none."""
        del indicators
        symbol = None
        if self._market_data and hasattr(self._market_data, "_symbol_by_token"):
            symbol = self._market_data._symbol_by_token.get(token)

        if not symbol:
            return

        try:
            if candles.empty:
                return
            latest_candle = candles.iloc[-1]
            price = float(latest_candle['close'])
            trace_id = f"{symbol}-{int(time.time() * 1000)}"

            signal = self._strategy_manager.generate_signal(symbol, price)
            if signal:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                prepared_signal, prepare_reason = await self._prepare_signal_for_handling(
                    signal,
                    price,
                    trace_id,
                )
                if prepared_signal is None:
                    self._emit_runner_eval_decision(
                        symbol=symbol,
                        stage="token_process",
                        reason=str(prepare_reason or "signal_prepare_failed"),
                        allowed=False,
                        trace_id=trace_id,
                    )
                    return
                self._handle_signal(prepared_signal, price, now, trace_id=trace_id)

            self._logger.debug("STRATEGY_TRIGGERED token=%s symbol=%s", token, symbol)
        except Exception as e:
            self._logger.error(f"Error in _process_token for {symbol}: {e}")

    def start(self) -> None:
        """Start processing market data events."""
        symbols: list[str] = []
        # Runner start follows readiness gates; execution is enabled by mark_ready.
        try:
            with self._lock:
                if self._running:
                    return
                self._running = True
            self._trading_paused = False
            if not isinstance(self._active_symbols, set):
                raise RuntimeError("Invalid active symbols container type")
            symbols = list(self._active_symbols)
            self._frozen_universe = set(symbols)
            self._universe_controller.update(symbols)
            self._history_gate_failed = False
            # ✅ CRITICAL FIX: Only set HISTORICAL_READY if mark_ready() has NOT already
            # promoted the state to EXECUTION_ENABLED.  The startup sequence calls
            # mark_ready() → EXECUTION_ENABLED, then calls start() seconds later.
            if not self._active_symbols:
                log_throttled(
                    self._logger,
                    "runner_start_deferred_no_active_symbols",
                    f"STRATEGY_RUNNER_START_DEFERRED reason=no_active_symbols state={self._runner_state}",
                    interval_sec=60.0,
                    level=logging.INFO,
                    extra={
                        "event": "STRATEGY_RUNNER_START_DEFERRED",
                        "reason": "no_active_symbols",
                        "state": str(self._runner_state),
                    },
                )
                if self._runner_state != RunnerState.EXECUTION_ENABLED:
                    self._runner_state = RunnerState.STARTING
                self._running = False
                return
            if self._runner_state != RunnerState.EXECUTION_ENABLED:
                self._runner_state = RunnerState.HISTORICAL_READY
            else:
                self._logger.info(
                    "STRATEGY_RUNNER_STATE_PRESERVED state=%s reason=start_after_mark_ready",
                    self._runner_state,
                    extra={
                        "event": "STRATEGY_RUNNER_STATE_PRESERVED",
                        "state": str(self._runner_state),
                        "reason": "start_after_mark_ready",
                    },
                )
            if self._order_manager and hasattr(self._order_manager, "get_kill_switch_status"):
                try:
                    ks = self._order_manager.get_kill_switch_status()
                    self._logger.info(
                        "ORDER_KILL_SWITCH_STATE active=%s reason=%s failures=%s engaged_at=%s",
                        ks.get("active"),
                        ks.get("kill_reason"),
                        ks.get("consecutive_failures"),
                        ks.get("engaged_at"),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("kill_switch_state_unavailable: %s", exc)
            for symbol in symbols:
                self._symbol_states.setdefault(symbol, SymbolState.DISCOVERED)
                self._data_phase.setdefault(symbol, "HYDRATION")
                self._rate_limit_backoff_until_by_symbol = {}
            # BUG W3 FIX: Do NOT wipe warmup accumulators when mark_ready() has
            # already promoted runner state to EXECUTION_ENABLED.  The call order
            # in startup_sequence is: hydrate → mark_ready() → start().  Wiping
            # these dicts unconditionally discards the VWAP / streak / bar-count
            # state computed during hydration.  The downgrade-protection in
            # _set_symbol_hydration_state() saves symbol states from regressing,
            # but _history_ready_by_symbol is wiped here explicitly and only
            # recovered on the first tick — making the debug log misleading.
            preserve_execution_state = self._runner_state == RunnerState.EXECUTION_ENABLED
            if not preserve_execution_state:
                self._vwap_state = {}
                self._symbol_bar_count = {}
                self._hydration_ready_streak = {}
                self._history_ready_by_symbol = {symbol: False for symbol in symbols}
            else:
                self._logger.info(
                    "STRATEGY_WARMUP_STATE_PRESERVED symbols=%d",
                    len(symbols),
                    extra={
                        "event": "STRATEGY_WARMUP_STATE_PRESERVED",
                        "symbol_count": len(symbols),
                    },
                )
            # Always reset per-session rate limits (independent of warmup state).

        # Capture the loop if called from async context (optional safety)
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        # self._market_data.start()
        # worker = threading.Thread(target=self._strategy_worker, daemon=True)
        # worker.start()

            if self._data_hub is not None:
                reset = getattr(self._data_hub, "reset_warmup", None)
                if callable(reset):
                    reset()

            for symbol in symbols:
                self._subscribe_symbol(symbol)

            self._logger.info("Strategy runner started with symbols: %s", symbols)

            # ✅ FIX: Launch Backfill Task (EMERGENCY FALLBACK ONLY)
            # BUG W4 FIX: _backfill_history() was scheduled immediately in runner.start(),
            # which races with core/app.py EngineWarmupTask. We only need the backfill task
            # as an emergency fallback if EngineWarmupTask fails.
            emergency_backfill_enabled = _env_bool(
                "RUNNER_ENABLE_EMERGENCY_BACKFILL",
                default=False,
            )
            if (
                self._config.fetch_history_on_startup
                and emergency_backfill_enabled
                and self._main_loop
                and not self._backfill_task_started
            ):
                self._backfill_task_started = True

                async def _deferred_backfill() -> None:
                    # Fix: delay the fallback task by 60s so app.py startup_sequence always finishes
                    await asyncio.sleep(60.0)
                    await self._backfill_history()

                self._main_loop.create_task(_deferred_backfill())
            with self._lock:
                self._runner_state = RunnerState.EXECUTION_ENABLED
                self._running = True
        except Exception as e:
            with self._lock:
                self._running = False
                self._runner_state = RunnerState.ERROR
            self._logger.error("Failure in StrategyRunner.start: %s", e)
            raise

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
        dynamic_option_add_enabled = bool(
            _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", True)
        )
        is_dynamic_option_symbol = bool(
            dynamic_option_add_enabled
            and normalized.startswith("NFO:")
            and ("CE" in normalized or "PE" in normalized)
        )
        with self._lock:
            self._candle_engines.setdefault(normalized, CandleEngine())
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
                if is_dynamic_option_symbol:
                    self._frozen_universe.add(normalized)
                    self._logger.info(
                        "Dynamic option symbol admitted to frozen universe",
                        extra={
                            "event": "symbol_add_dynamic_option_admitted",
                            "symbol": normalized,
                        },
                    )
                else:
                    self._logger.info(
                        "Symbol add deferred until next session boundary",
                        extra={"event": "symbol_add_deferred", "symbol": normalized},
                    )
                    return
            self._active_symbols.add(normalized)
            self._tracked_symbols.add(normalized)
            self._data_phase[normalized] = "HYDRATION"
            self._symbol_states.setdefault(normalized, SymbolState.DISCOVERED)
            self._set_symbol_hydration_state(normalized, SymbolState.HYDRATING)
            # FIX (2026-02-27): Initialize _last_bar_ts for dynamically-added symbols.
            # Options arrive via add_symbol() not mark_ready(), so _last_bar_ts is never
            # set.  PHASE 9 sees None → "bar_not_finalized" → _mark_symbol_unready →
            # HYDRATING → PHASE 7 blocks all future ticks → permanent lockout for options.
            if normalized not in self._last_bar_ts:
                self._last_bar_ts[normalized] = datetime.now(timezone.utc)
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

        cached = self._recent_history_cache.get(normalized) or []
        if cached:
            self._symbol_history[normalized] = list(cached)
            self._restored_from_cache_symbols.add(normalized)
            self._logger.info("RUNNER_HISTORY_CACHE_RESTORED symbol=%s bars=%d", normalized, len(cached))
        self._hydrate_from_mdm_cache(normalized)

        self._logger.info("Tracking symbol %s", normalized)

    def _required_bars_for_symbol(self, symbol: str) -> int:
        """Args: symbol. Returns: required bar count by symbol role. Raises: None."""
        return (
            self._option_required_bars
            if self._is_tradable_option_symbol(symbol)
            else self._context_required_bars
        )

    def _prehydrate_symbol_history(self, symbol: str) -> None:
        """Hydrate startup candles from cache only. Args: symbol. Returns: None. Raises: None."""
        self._hydrate_from_mdm_cache(symbol)

    def _get_mdm_bars(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        """Fetch cached bars from MDM/DataHub only. Args: symbol, limit. Returns: rows. Raises: None."""
        for source in (self._market_data, self._data_hub):
            if source is None:
                continue
            for name in ("get_ohlc_bars", "get_ohlc", "get_recent_bars"):
                fn = getattr(source, name, None)
                if not callable(fn):
                    continue
                try:
                    try:
                        bars = fn(symbol, limit=limit)
                    except TypeError:
                        bars = fn(symbol)
                    if bars:
                        return [dict(row) for row in list(bars)[-limit:]]
                except Exception:
                    continue
        return []

    def _request_mdm_hydration(self, symbol: str, min_bars: int) -> None:
        """Request async hydration from owner service. Args: symbol/min_bars. Returns: None. Raises: None."""
        for source in (self._market_data, self._data_hub):
            fn = getattr(source, "request_hydration", None)
            if callable(fn):
                try:
                    fn(symbol, min_bars=min_bars, reason="runner_missing_bars")
                    return
                except Exception:
                    pass
        log_throttled(
            self._logger,
            f"runner_waiting_for_mdm_hydration:{symbol}",
            f"RUNNER_WAITING_FOR_MDM_HYDRATION symbol={symbol} min_bars={min_bars}",
            interval_sec=60.0,
            level=logging.INFO,
            extra={"event": "RUNNER_WAITING_FOR_MDM_HYDRATION", "symbol": symbol, "min_bars": min_bars},
        )

    def _hydrate_from_mdm_cache(self, symbol: str) -> int:
        """Hydrate symbol from cached MDM bars. Args: symbol. Returns: count. Raises: None."""
        target = self._required_bars_for_symbol(symbol)
        rows = self._get_mdm_bars(symbol, target)
        ingested = 0
        for row in rows:
            payload = dict(row)
            payload["symbol"] = symbol
            self.ingest_historical_bar(payload)
            ingested += 1
        if ingested >= target:
            self._set_symbol_hydration_state(symbol, SymbolState.READY)
        else:
            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
            self._request_mdm_hydration(symbol, target)
        log_throttled(
            self._logger,
            f"runner_mdm_cache_hydration:{symbol}",
            f"RUNNER_MDM_CACHE_HYDRATION symbol={symbol} ingested={ingested} target={target}",
            interval_sec=60.0,
            level=logging.INFO,
            extra={"event": "RUNNER_MDM_CACHE_HYDRATION", "symbol": symbol, "ingested": ingested, "target": target},
        )
        return ingested

    def _should_log_throttled(self, key: str, interval_s: float = 30.0) -> bool:
        """Decide whether a throttled log should emit. Args: key/interval_s. Returns: bool. Raises: None."""
        now = time.monotonic()
        last = float(self._log_throttle_state.get(key, 0.0))
        if now - last >= max(0.0, float(interval_s)):
            self._log_throttle_state[key] = now
            return True
        return False

    def _symbol_has_valid_data(self, symbol: str) -> bool:
        """Validate symbol candle data integrity from the indicator engine."""
        try:
            history = self._symbol_history.get(symbol, [])
            required = self._required_bars_for_symbol(symbol)
            if len(history) < required:
                if self._indicator_engine and self._indicator_engine.has_min_bars(
                    symbol, required
                ):
                    return True
                return False

            bar_rows = [
                (bar.__dict__ if hasattr(bar, "__dict__") else bar.as_mapping())
                for bar in history
            ]
            frame = pd.DataFrame(bar_rows)
            required_columns = {"open", "high", "low", "close", "volume"}
            if not required_columns.issubset(frame.columns):
                return False
            if len(frame) < required:
                return False
            if frame[list(required_columns)].isna().any().any():
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner._symbol_has_valid_data: %s",
                exc,
                extra={"event": "symbol_data_validation_error", "symbol": symbol},
                exc_info=exc,
            )
            return False

    def _seed_pipeline_store(self, symbol: str) -> None:
        """Seed pipeline CandleStore from IndicatorEngine PriceHistory.

        CandleStore.seed() is a no-op if already populated (STEP 6 guard).
        Called inside self._lock — PriceHistory getters hold their own internal lock.
        """
        try:
            if self._indicator_engine is None:
                return
            ph = self._indicator_engine._histories.get(symbol)
            if ph is None or len(ph) == 0:
                return
            opens = ph.get_opens()
            highs = ph.get_highs()
            lows = ph.get_lows()
            closes = ph.get_closes()
            volumes = ph.get_volumes()
            timestamps = ph.get_timestamps()
            if not closes:
                return
            bars: list[dict[str, Any]] = []
            for i, ts in enumerate(timestamps):
                bars.append({
                    "timestamp": ts,
                    "open":   opens[i]   if i < len(opens)   else closes[i],
                    "high":   highs[i]   if i < len(highs)   else closes[i],
                    "low":    lows[i]    if i < len(lows)    else closes[i],
                    "close":  closes[i],
                    "volume": volumes[i] if i < len(volumes) else 0.0,
                })
            self._pipeline.store.seed(symbol, bars)
        except Exception as exc:
            self._logger.warning(
                "Failed to seed pipeline store for %s: %s", symbol, exc,
                extra={"event": "pipeline_store_seed_failed", "symbol": symbol},
            )

    def _seed_candle_engine_from_history(self, symbol: str) -> None:
        """Seed legacy CandleEngine.df from IndicatorEngine PriceHistory.

        Ensures ensure_valid_data() passes on first tick without broker API calls.
        Called inside self._lock.
        """
        try:
            if self._indicator_engine is None:
                return
            ph = self._indicator_engine._histories.get(symbol)
            if ph is None or len(ph) == 0:
                return
            opens = ph.get_opens()
            highs = ph.get_highs()
            lows = ph.get_lows()
            closes = ph.get_closes()
            volumes = ph.get_volumes()
            timestamps = ph.get_timestamps()
            if not closes:
                return
            rows = []
            for i, ts in enumerate(timestamps):
                rows.append({
                    "timestamp": pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None
                                 else pd.Timestamp(ts),
                    "open":   float(opens[i])   if i < len(opens)   else float(closes[i]),
                    "high":   float(highs[i])   if i < len(highs)   else float(closes[i]),
                    "low":    float(lows[i])    if i < len(lows)    else float(closes[i]),
                    "close":  float(closes[i]),
                    "volume": float(volumes[i]) if i < len(volumes) else 0.0,
                })
            df = pd.DataFrame(rows)
            df = df.drop_duplicates(subset="timestamp", keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)
            engine = self._candle_engines.setdefault(symbol, CandleEngine())
            engine.df = df.tail(engine.max_bars).reset_index(drop=True)
            self._logger.debug(
                "candle_engine_seeded",
                extra={"event": "candle_engine_seeded", "symbol": symbol,
                       "bars": len(engine.df)},
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to seed CandleEngine for %s: %s", symbol, exc,
                extra={"event": "candle_engine_seed_failed", "symbol": symbol},
            )

    def quarantine_symbol(self, symbol: str, reason: str, **context: Any) -> None:
        """Move symbol to quarantine set while keeping runner online. Args: symbol, reason. Returns: None. Raises: None."""
        try:
            normalized = enforce_canonical(normalize_symbol(symbol))
            with self._lock:
                self._quarantined_symbols.add(normalized)
                self._set_symbol_hydration_state(
                    normalized, SymbolState.DEGRADED, allow_downgrade=True
                )
            self._logger.critical(
                "symbol_quarantined",
                extra={
                    "event": "symbol_quarantined",
                    "symbol": normalized,
                    "reason": reason,
                    **context,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner.quarantine_symbol: %s",
                exc,
                extra={
                    "event": "quarantine_symbol_error",
                    "symbol": symbol,
                    "reason": reason,
                },
                exc_info=exc,
            )

    def remove_symbol(self, symbol: str) -> None:
        """Stop tracking a symbol."""
        normalized = self._normalize_symbol(symbol)
        removed_from_watchdog = False
        pending_backfill_cancelled = False
        with self._lock:
            state = self._symbol_state.get(normalized)
            if state is None:
                return

            state.active = False
            history = list(self._symbol_history.get(normalized) or [])
            if history:
                keep = max(self._context_required_bars, self._option_required_bars)
                self._recent_history_cache[normalized] = history[-keep:]
                self._logger.info("RUNNER_HISTORY_CACHE_PRESERVED symbol=%s bars=%d", normalized, len(self._recent_history_cache[normalized]))
            self._active_symbols.discard(normalized)
            self._tracked_symbols.discard(normalized)
            self._live_symbols.discard(normalized)
            self._frozen_universe.discard(normalized)
            self._symbol_states.pop(normalized, None)
            self._symbol_bar_count.pop(normalized, None)
            self._hydration_ready_streak.pop(normalized, None)
            self._vwap_state.pop(normalized, None)
            self._last_tick_time_by_symbol.pop(normalized, None)
            self._last_ws_stale_log_ts_by_symbol.pop(normalized, None)
            self._last_same_bar_eval_ts_by_symbol.pop(normalized, None)
            self._last_strategy_versions.pop(normalized, None)
            self._candle_versions.pop(normalized, None)
            self._live_bar_seen.discard(normalized)
            self._warmup_complete_logged.discard(normalized)
            self._symbol_last_signal_ts.pop(normalized, None)
            self._first_tick_logged_symbols.discard(normalized)
            self._first_live_bar_logged_symbols.discard(normalized)
            removed_from_watchdog = True
            pending_backfill_cancelled = True
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
        stale_throttle_keys = [
            key for key in list(self._log_throttle_state) if normalized in key
        ]
        for key in stale_throttle_keys:
            self._log_throttle_state.pop(key, None)

        self._logger.info(
            "SYMBOL_REMOVAL_CLEANUP symbol=%s removed_from_runner=%s removed_from_watchdog=%s pending_backfill_cancelled=%s",
            normalized,
            True,
            removed_from_watchdog,
            pending_backfill_cancelled,
            extra={
                "event": "SYMBOL_REMOVAL_CLEANUP",
                "symbol": normalized,
                "removed_from_runner": True,
                "removed_from_watchdog": removed_from_watchdog,
                "pending_backfill_cancelled": pending_backfill_cancelled,
            },
        )

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
            data_source = self._data_hub or self._market_data
            quote = data_source.get_quote(base_symbol)
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

    def build_candidate_snapshots(
        self,
        underlying: str = "NIFTY",
        direction_bias: Literal["CE", "PE"] = "CE",
        atm_strike: int | None = None,
        window_each_side: int = 2,
    ) -> list[dict[str, Any]]:
        """Build candidate snapshots for sync/offline usage only. Args: context fields. Returns: snapshots. Raises: RuntimeError."""
        try:
            asyncio.get_running_loop()
            self._logger.warning(
                "CANDIDATE_SNAPSHOT_BUILD_FAILED reason=async_required",
                extra={"event": "CANDIDATE_SNAPSHOT_BUILD_FAILED", "reason": "async_required"},
            )
            return []
        except RuntimeError:
            snapshots, _ = asyncio.run(
                self.build_candidate_snapshots_async(
                    underlying=underlying,
                    direction_bias=direction_bias,
                    atm_strike=atm_strike,
                    window_each_side=window_each_side,
                )
            )
            return snapshots

    async def build_candidate_snapshots_async(
        self,
        underlying: str = "NIFTY",
        direction_bias: Literal["CE", "PE"] = "CE",
        atm_strike: int | None = None,
        window_each_side: int = 2,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Args: option context. Returns: snapshots and refresh_pending flag. Raises: none."""
        try:
            if self._market_data is None or not hasattr(self._market_data, "get_symbol_snapshot"):
                return [], True
            spot_snapshot = self._market_data.get_symbol_snapshot(underlying)
            spot_ltp = getattr(spot_snapshot, "ltp", None)
            spot_canonical = str(getattr(spot_snapshot, "canonical_symbol", "") or "").upper()
            if spot_ltp is None or float(spot_ltp) <= 0:
                self._logger.warning(
                    "CANDIDATE_SNAPSHOT_BUILD_FAILED reason=spot_missing underlying=%s spot_canonical=%s spot_ltp=%s",
                    underlying,
                    spot_canonical,
                    spot_ltp,
                    extra={"event": "CANDIDATE_SNAPSHOT_BUILD_FAILED", "reason": "spot_missing", "underlying": underlying, "spot_canonical": spot_canonical, "spot_ltp": spot_ltp},
                )
                return [], True
            atm = int(atm_strike or round(float(spot_ltp) / 50.0) * 50)
            side = str(direction_bias).upper()
            if side not in {"CE", "PE"}:
                side = "CE"
            target_strikes = {
                atm + 50 * offset
                for offset in range(-max(1, int(window_each_side)), max(1, int(window_each_side)) + 1)
            }
            selected = self._resolve_candidate_contracts(side=side, target_strikes=target_strikes)
            selected = sorted(set(selected), key=lambda item: abs(item[1] - atm))
            any_refresh_pending = False
            candidates: list[dict[str, Any]] = []
            for sym, strike in selected[: max(1, 2 * window_each_side + 1)]:
                symbol_refresh_pending = False
                try:
                    self._market_data.request_symbol_subscription(sym)
                except Exception:
                    pass
                ensure_tick_fn = getattr(self._market_data, "ensure_fresh_tick", None)
                if callable(ensure_tick_fn):
                    self._logger.debug(
                        "CANDIDATE_REFRESH_REQUESTED symbol=%s", sym, extra={"event": "CANDIDATE_REFRESH_REQUESTED", "symbol": sym}
                    )
                    refresh_result = ensure_tick_fn(sym)
                    if inspect.isawaitable(refresh_result):
                        try:
                            await asyncio.wait_for(refresh_result, timeout=2.0)
                            self._logger.debug(
                                "CANDIDATE_REFRESH_COMPLETE symbol=%s",
                                sym,
                                extra={"event": "CANDIDATE_REFRESH_COMPLETE", "symbol": sym},
                            )
                        except asyncio.TimeoutError:
                            symbol_refresh_pending = True
                            any_refresh_pending = True
                            self._logger.warning(
                                "CANDIDATE_REFRESH_TIMEOUT symbol=%s",
                                sym,
                                extra={"event": "CANDIDATE_REFRESH_TIMEOUT", "symbol": sym},
                            )
                snap = self._market_data.get_symbol_snapshot(sym)
                if symbol_refresh_pending and (snap.ltp is None or not snap.tradable_quote):
                    self._logger.warning(
                        "CANDIDATE_SNAPSHOT_PENDING_REFRESH symbol=%s",
                        sym,
                        extra={"event": "CANDIDATE_SNAPSHOT_PENDING_REFRESH", "symbol": sym},
                    )
                spread_pct = None
                if snap.bid and snap.ask and snap.bid > 0 and snap.ask > 0:
                    mid = (snap.bid + snap.ask) / 2.0
                    if mid > 0:
                        spread_pct = (snap.ask - snap.bid) / mid
                candidates.append(
                    {
                        "symbol": snap.canonical_symbol,
                        "side": side,
                        "option_type": side,
                        "direction_bias": side,
                        "atm_strike": atm,
                        "strike": strike,
                        "ltp": snap.ltp,
                        "bid": snap.bid,
                        "ask": snap.ask,
                        "mid": snap.mid,
                        "spread_pct": spread_pct,
                        "tick_age_s": snap.tick_age_s,
                        "source": snap.source,
                        "real_ticks_last_60s": snap.real_ticks_last_60s,
                        "latest_candle_provisional": snap.latest_candle_provisional,
                        "latest_candle_synthetic": snap.latest_candle_synthetic,
                        "latest_candle_volume": float(getattr(snap, "latest_candle_volume", 0.0) or 0.0),
                        "ohlc_valid": snap.ohlc_valid,
                        "atm_distance": int(abs(strike - atm) / 50),
                        "bid_missing": snap.bid_missing,
                        "ask_missing": snap.ask_missing,
                        "bid_ask_source": snap.bid_ask_source,
                        "tradable_quote": snap.tradable_quote,
                        "refresh_pending": symbol_refresh_pending,
                        "atr_option": float(getattr(snap, "atr_option", 0.0) or 0.0),
                        "history_bars": int(getattr(snap, "history_bars", 0) or 0),
                        "data_quality_score": float(getattr(snap, "data_quality_score", 0.0) or 0.0),
                        "quote_quality": "bid_ask" if bool(snap.tradable_quote) else "ltp_only",
                        "ltp_only_fallback": bool(
                            snap.ltp is not None
                            and float(snap.ltp) > 0
                            and not bool(snap.tradable_quote)
                            and not symbol_refresh_pending
                        ),
                        "effective_bars": int(getattr(snap, "effective_bars", 0) or 0),
                    }
                )
            no_valid_candidates = not any(
                bool(cand.get("tradable_quote"))
                and not bool(cand.get("refresh_pending"))
                and cand.get("ltp") is not None
                for cand in candidates
            )
            return candidates, bool(any_refresh_pending and no_valid_candidates)
        except Exception as exc:
            self._logger.error(
                "CANDIDATE_SNAPSHOT_BUILD_FAILED reason=%s underlying=%s direction_bias=%s atm_strike=%s",
                exc,
                underlying,
                direction_bias,
                atm_strike,
                extra={
                    "event": "CANDIDATE_SNAPSHOT_BUILD_FAILED",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "underlying": underlying,
                    "direction_bias": direction_bias,
                    "atm_strike": atm_strike,
                },
                exc_info=True,
            )
            return [], True

    def _schedule_signal_preparation(
        self,
        signal: Signal,
        price: float,
        now: datetime,
        trace_id: str,
    ) -> tuple[bool, str | None]:
        """Schedule signal preparation from sync paths. Args: signal/price/now/trace_id. Returns: scheduled state and reason. Raises: none."""
        async def _job() -> None:
            prepared_signal, prepare_reason = await self._prepare_signal_for_handling(
                signal, price, trace_id
            )
            if prepared_signal is None:
                self._emit_runner_eval_decision(
                    symbol=signal.symbol,
                    stage="phase10_execute",
                    reason=str(prepare_reason or "signal_prepare_failed"),
                    allowed=False,
                    trace_id=trace_id,
                )
                self._logger.info(
                    "SIGNAL_EXECUTION_RESULT symbol=%s accepted=%s reason=%s order_id=%s trace_id=%s",
                    signal.symbol, False, prepare_reason, None, trace_id,
                    extra={"event": "SIGNAL_EXECUTION_RESULT", "symbol": signal.symbol, "accepted": False, "reason": prepare_reason, "order_id": None, "trace_id": trace_id},
                )
                return
            result = self._handle_signal(prepared_signal, price, now, trace_id=trace_id)
            self._logger.info(
                "SIGNAL_EXECUTION_RESULT symbol=%s accepted=%s reason=%s order_id=%s trace_id=%s",
                signal.symbol,
                result.accepted,
                result.reason,
                result.order_id,
                trace_id,
                extra={
                    "event": "SIGNAL_EXECUTION_RESULT",
                    "symbol": signal.symbol,
                    "accepted": result.accepted,
                    "reason": result.reason,
                    "order_id": result.order_id,
                    "trace_id": trace_id,
                },
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            prepared_signal, prepare_reason = asyncio.run(
                self._prepare_signal_for_handling(signal, price, trace_id)
            )
            if prepared_signal is None:
                return False, prepare_reason
            result = self._handle_signal(prepared_signal, price, now, trace_id=trace_id)
            self._logger.info(
                "SIGNAL_EXECUTION_RESULT symbol=%s accepted=%s reason=%s order_id=%s trace_id=%s",
                signal.symbol,
                result.accepted,
                result.reason,
                result.order_id,
                trace_id,
                extra={
                    "event": "SIGNAL_EXECUTION_RESULT",
                    "symbol": signal.symbol,
                    "accepted": result.accepted,
                    "reason": result.reason,
                    "order_id": result.order_id,
                    "trace_id": trace_id,
                },
            )
            return True, None
        task = loop.create_task(_job(), name=f"signal_prepare:{signal.symbol}:{trace_id}")
        def _on_done(done_task: asyncio.Task[None]) -> None:
            """Handle async task completion. Args: done_task. Returns: None. Raises: None."""
            try:
                done_task.result()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "SIGNAL_PREPARATION_TASK_FAILED symbol=%s trace_id=%s error=%s",
                    signal.symbol,
                    trace_id,
                    exc,
                    extra={"event": "SIGNAL_PREPARATION_TASK_FAILED", "symbol": signal.symbol, "trace_id": trace_id, "error": str(exc)},
                    exc_info=exc,
                )
        task.add_done_callback(_on_done)
        return True, "signal_preparation_scheduled"

    def set_active_option_context(
        self,
        *,
        selected_ce: str | None = None,
        selected_pe: str | None = None,
        atm_strike: int | float | str | None = None,
        option_symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        """Set active option context. Args: selected_ce/selected_pe/atm_strike/option_symbols. Returns: none. Raises: none."""
        selected_ce_norm = normalize_symbol(str(selected_ce)) if selected_ce else None
        selected_pe_norm = normalize_symbol(str(selected_pe)) if selected_pe else None
        atm_value = None
        if atm_strike not in (None, ""):
            try:
                atm_value = int(float(atm_strike))
            except (TypeError, ValueError):
                atm_value = None
        if selected_ce_norm:
            self._active_selected_ce = selected_ce_norm
        if selected_pe_norm:
            self._active_selected_pe = selected_pe_norm
        self._active_atm_strike = atm_value
        self._active_option_symbols = {normalize_symbol(str(sym)) for sym in (option_symbols or []) if sym}
        self._logger.info(
            "RUNNER_ACTIVE_OPTION_CONTEXT selected_ce=%s selected_pe=%s atm_strike=%s option_count=%d",
            self._active_selected_ce,
            self._active_selected_pe,
            self._active_atm_strike,
            len(self._active_option_symbols),
            extra={"event": "RUNNER_ACTIVE_OPTION_CONTEXT", "selected_ce": self._active_selected_ce, "selected_pe": self._active_selected_pe, "atm_strike": self._active_atm_strike, "option_count": len(self._active_option_symbols)},
        )

    def set_active_trading_universe(self, basket: Mapping[str, Any]) -> None:
        """Set active trading universe snapshot. Args: basket. Returns: none. Raises: none."""
        option_symbols = basket.get("option_symbols") or basket.get("symbols") or []
        self.set_active_option_context(
            selected_ce=cast(str | None, basket.get("selected_ce") or basket.get("atm_ce")),
            selected_pe=cast(str | None, basket.get("selected_pe") or basket.get("atm_pe")),
            atm_strike=cast(int | float | str | None, basket.get("atm_strike")),
            option_symbols=cast(list[str] | tuple[str, ...] | set[str], option_symbols),
        )
        self._logger.info(
            "RUNNER_ACTIVE_BASKET_UPDATED selected_ce=%s selected_pe=%s option_count=%d",
            self._active_selected_ce,
            self._active_selected_pe,
            len(self._active_option_symbols),
        )

    def set_runtime_readiness(
        self,
        *,
        data_hard_ready: bool,
        evaluation_ready: bool,
        live_orders_armed: bool,
        reason: str | None = None,
        selected_ce: str | None = None,
        selected_pe: str | None = None,
        atm_strike: int | float | str | None = None,
        option_symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        """Set app-level runtime readiness flags. Args: flags/reason. Returns: none. Raises: none."""
        self._runtime_data_hard_ready = bool(data_hard_ready)
        self._runtime_evaluation_ready = bool(evaluation_ready)
        self._runtime_live_orders_armed = bool(live_orders_armed)
        self._runtime_readiness_reason = reason
        self._runtime_startup_ready = bool(
            self._runtime_data_hard_ready and self._runtime_evaluation_ready
        )
        if any(value is not None for value in (selected_ce, selected_pe, atm_strike, option_symbols)):
            self.set_active_option_context(selected_ce=selected_ce, selected_pe=selected_pe, atm_strike=atm_strike, option_symbols=option_symbols)
        self._logger.info(
            "RUNNER_STARTUP_READINESS_UPDATE startup_ready=%s data_hard_ready=%s evaluation_ready=%s live_orders_armed=%s reason=%s selected_ce=%s selected_pe=%s option_count=%s",
            self._runtime_startup_ready,
            self._runtime_data_hard_ready,
            self._runtime_evaluation_ready,
            self._runtime_live_orders_armed,
            self._runtime_readiness_reason,
            self._active_selected_ce,
            self._active_selected_pe,
            len(self._active_option_symbols),
        )

    def get_runtime_readiness_snapshot(self) -> dict[str, object]:
        """Return runtime readiness snapshot. Args: none. Returns: readiness map. Raises: none."""
        return {
            "startup_ready": self._runtime_startup_ready,
            "data_hard_ready": self._runtime_data_hard_ready,
            "evaluation_ready": self._runtime_evaluation_ready,
            "live_orders_armed": self._runtime_live_orders_armed,
            "reason": self._runtime_readiness_reason,
            "selected_ce": self._active_selected_ce,
            "selected_pe": self._active_selected_pe,
            "atm_strike": self._active_atm_strike,
            "option_count": len(self._active_option_symbols),
            "runner_state": str(self._runner_state),
        }

    async def _prepare_signal_for_handling(
        self,
        signal: Signal,
        price: float,
        trace_id: str | None,
    ) -> tuple[Signal | None, str | None]:
        """Prepare signal metadata pre-sync handler. Args: signal, price, trace_id. Returns: prepared signal + block reason. Raises: none."""
        del price
        mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
        is_live_mode = mode == "LIVE" or (
            str(os.getenv("ENABLE_LIVE", "false")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if not is_live_mode:
            return signal, None
        runtime_ready = bool(getattr(self, "_runtime_data_hard_ready", False))
        if is_live_mode and not runtime_ready:
            return None, str(
                getattr(self, "_runtime_readiness_reason", None)
                or "startup_pipeline_not_ready"
            )
        metadata = dict(signal.metadata or {})
        option_side = infer_option_side(signal.symbol, metadata)
        is_directional_option = option_side in {"CE", "PE"}
        if not is_directional_option:
            return dataclasses.replace(signal, metadata=metadata), None
        candidate_snapshots_obj = metadata.get("candidate_snapshots")
        if not isinstance(candidate_snapshots_obj, list):
            fallback_candidate = self._build_single_candidate_from_signal(
                signal=signal,
                metadata=metadata,
                option_side=cast(Literal["CE", "PE"], option_side),
            )
            if fallback_candidate is not None:
                metadata["candidate_snapshots"] = [fallback_candidate]
                metadata.setdefault("atm_strike", int(fallback_candidate.get("atm_strike") or 0))
                self._logger.info(
                    "CANDIDATE_FALLBACK_FROM_SIGNAL_USED symbol=%s",
                    signal.symbol,
                    extra={"event": "CANDIDATE_FALLBACK_FROM_SIGNAL_USED", "symbol": signal.symbol},
                )
                return dataclasses.replace(signal, metadata=metadata), None
            underlying = self._extract_underlying(signal.symbol) or "NIFTY"
            if underlying in {"NFO", "NSE", ""}:
                underlying = "NIFTY"
            atm_strike = int(metadata.get("atm_strike") or 0)
            built, refresh_pending = await self.build_candidate_snapshots_async(
                direction_bias=cast(Literal["CE", "PE"], option_side),
                atm_strike=atm_strike,
                underlying=underlying,
            )
            if refresh_pending or not built:
                fallback_candidate = self._build_single_candidate_from_signal(
                    signal=signal,
                    metadata=metadata,
                    option_side=cast(Literal["CE", "PE"], option_side),
                )
                if fallback_candidate is not None:
                    metadata["candidate_snapshots"] = [fallback_candidate]
                    metadata.setdefault("atm_strike", int(fallback_candidate.get("atm_strike") or 0))
                    self._logger.info(
                        "CANDIDATE_FALLBACK_FROM_SIGNAL_USED_AFTER_REFRESH_PENDING symbol=%s",
                        signal.symbol,
                        extra={"event": "CANDIDATE_FALLBACK_FROM_SIGNAL_USED_AFTER_REFRESH_PENDING", "symbol": signal.symbol},
                    )
                    return dataclasses.replace(signal, metadata=metadata), None
            if refresh_pending:
                return None, "candidate_refresh_pending"
            if not built:
                return None, "missing_candidate_snapshots"
            metadata["candidate_snapshots"] = built
            if not metadata.get("atm_strike"):
                for snap in built:
                    if isinstance(snap, dict) and snap.get("atm_strike"):
                        metadata["atm_strike"] = int(snap["atm_strike"])
                        break
        if not metadata.get("candidate_snapshots"):
            return None, "missing_candidate_snapshots"
        return dataclasses.replace(signal, metadata=metadata), None

    def _build_single_candidate_from_signal(
        self,
        *,
        signal: Signal,
        metadata: Mapping[str, Any],
        option_side: Literal["CE", "PE"],
    ) -> dict[str, Any] | None:
        """Build a single candidate snapshot from signal and MDM data. Args: signal/metadata/option_side. Returns: candidate or none. Raises: none."""
        try:
            if self._market_data is None:
                return None
            get_snapshot = getattr(self._market_data, "get_symbol_snapshot", None)
            if not callable(get_snapshot):
                return None
            snapshot = get_snapshot(signal.symbol)
            ltp = float(snapshot.ltp) if snapshot.ltp is not None and float(snapshot.ltp) > 0 else None
            if ltp is None:
                return None
            bid = float(snapshot.bid) if snapshot.bid is not None and float(snapshot.bid) > 0 else 0.0
            ask = float(snapshot.ask) if snapshot.ask is not None and float(snapshot.ask) > 0 else 0.0
            has_bid_ask = bid > 0 and ask > 0
            strike_match = re.search(r"(\d{5})(CE|PE)$", str(signal.symbol).upper())
            parsed_strike = int(strike_match.group(1)) if strike_match is not None else 0
            strike = int(metadata.get("strike") or parsed_strike or metadata.get("atm_strike") or 0)
            atm_strike = int(metadata.get("atm_strike") or strike)
            spread_pct = ((ask - bid) / ltp * 100.0) if has_bid_ask and ltp > 0 else None
            tick_age_raw = getattr(snapshot, "tick_age_s", None)
            tick_age_s = float(tick_age_raw) if tick_age_raw is not None else 0.0
            tick_age_s = max(0.0, tick_age_s)
            real_ticks_raw = getattr(snapshot, "real_ticks_last_60s", None)
            real_ticks_last_60s = int(real_ticks_raw) if real_ticks_raw is not None else 0
            if ltp > 0 and real_ticks_last_60s < 1:
                real_ticks_last_60s = 1
            return {
                "symbol": signal.symbol,
                "side": option_side,
                "option_type": option_side,
                "strike": strike,
                "atm_strike": atm_strike,
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "spread_pct": spread_pct,
                "tick_age_s": tick_age_s,
                "real_ticks_last_60s": real_ticks_last_60s,
                "tradable_quote": bool(snapshot.tradable_quote and has_bid_ask),
                "ltp_only_fallback": not has_bid_ask,
                "quote_quality": "bid_ask" if has_bid_ask else "ltp_only",
                "source": str(snapshot.source or "signal_snapshot"),
                "atr_option": float(metadata.get("atr", 0.0) or 0.0),
                "history_bars": int(metadata.get("history_bars", 0) or 0),
                "data_quality_score": float(
                    metadata.get("data_quality_score")
                    if metadata.get("data_quality_score") is not None
                    else metadata.get("data_score", 0.0)
                    or 0.0
                ),
                "candidate_selected": bool(metadata.get("candidate_selected") or metadata.get("is_selected_option")),
                "quote_usable_for_order_plan": bool(snapshot.tradable_quote and has_bid_ask),
                "tradable_quote": bool(snapshot.tradable_quote and has_bid_ask),
                "effective_bars": int(metadata.get("effective_bars", metadata.get("history_bars", 0)) or 0),
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _build_single_candidate_from_signal: %s", exc, exc_info=exc)
            return None

    @staticmethod
    def _call_contract_resolver(
        fetch: Callable[..., Any], *, side: str, target_strikes: set[int]
    ) -> list[Any]:
        """Invoke resolver with compatible kwargs. Args: fetch/side/strikes. Returns: rows. Raises: none."""
        attempts = [
            {
                "underlying": "NIFTY",
                "side": side,
                "option_type": side,
                "strikes": sorted(target_strikes),
                "expiry": "nearest_weekly",
            },
            {
                "underlying": "NIFTY",
                "option_type": side,
                "strikes": sorted(target_strikes),
            },
            {"side": side, "strikes": sorted(target_strikes)},
            {"option_type": side},
            {},
        ]
        for kwargs in attempts:
            try:
                result = fetch(**kwargs)
                return list(result or [])
            except TypeError:
                continue
            except Exception:
                continue
        return []

    def _reject_signal_execution(
        self,
        *,
        symbol: str,
        trace_id: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> SignalExecutionResult:
        """Log and return rejection. Args: symbol/trace_id/reason/details. Returns: result. Raises: none."""
        payload = dict(details or {})
        self._logger.info(
            "SIGNAL_EXECUTION_RESULT accepted=False reason=%s symbol=%s trace_id=%s",
            reason,
            symbol,
            trace_id,
            extra={
                "event": "SIGNAL_EXECUTION_RESULT",
                "accepted": False,
                "reason": reason,
                "symbol": symbol,
                "trace_id": trace_id,
                **payload,
            },
        )
        return SignalExecutionResult(False, reason, details=payload)

    def _resolve_candidate_contracts(
        self, *, side: str, target_strikes: set[int]
    ) -> list[tuple[str, int]]:
        """Args: side and strikes. Returns: symbol/strike list. Raises: none."""
        sources = [
            ("OptionsContractStore", getattr(self, "_options_contract_store", None) or getattr(self, "_contract_store", None)),
            ("InstrumentManager", getattr(self, "_instrument_manager", None)),
            ("ContractSelector", getattr(self, "_contract_selector", None)),
        ]
        methods = ("get_atm_window", "get_contracts", "select_atm_contracts", "get_nearest_weekly_options")
        for source_name, source in sources:
            if source is None:
                continue
            for method in methods:
                fetch = getattr(source, method, None)
                if not callable(fetch):
                    continue
                rows = self._call_contract_resolver(
                    fetch, side=side, target_strikes=target_strikes
                )
                if not rows:
                    continue
                selected: list[tuple[str, int]] = []
                for row in rows:
                    row_mapping: Mapping[str, Any]
                    if isinstance(row, Mapping):
                        row_mapping = row
                    elif hasattr(row, "__dict__"):
                        row_mapping = cast(Mapping[str, Any], vars(row))
                    else:
                        continue
                    option_type = str(row_mapping.get("option_type") or "").upper()
                    tradingsymbol = str(row_mapping.get("tradingsymbol") or "").upper()
                    if option_type not in {"CE", "PE"}:
                        if tradingsymbol.endswith("CE"):
                            option_type = "CE"
                        elif tradingsymbol.endswith("PE"):
                            option_type = "PE"
                    if option_type not in {"CE", "PE"}:
                        continue
                    try:
                        strike = int(float(row_mapping.get("strike") or 0))
                    except (TypeError, ValueError):
                        continue
                    exchange = str(row_mapping.get("exchange") or "NFO").upper()
                    expiry = str(row_mapping.get("expiry") or "").upper()
                    if (
                        option_type != side
                        or strike not in target_strikes
                        or not tradingsymbol
                        or not expiry
                        or re.search(r"\d{1,2}[A-Z]{3}", tradingsymbol) is None
                    ):
                        continue
                    selected.append((f"{exchange}:{tradingsymbol}", strike))
                if selected:
                    self._logger.info(
                        "CANDIDATE_RESOLVER_USED source=%s count=%s",
                        source_name,
                        len(selected),
                        extra={"event": "CANDIDATE_RESOLVER_USED", "source": source_name, "count": len(selected)},
                    )
                    return selected
        selected = []
        tracked = []
        tracked_fn = getattr(self._market_data, "tracked_snapshot", None)
        if callable(tracked_fn):
            tracked = [str(sym) for sym in tracked_fn()]
        for sym in tracked:
            norm = enforce_canonical(normalize_symbol(sym))
            if not norm.startswith("NFO:NIFTY") or not norm.endswith(side):
                continue
            match = re.search(r"^NFO:NIFTY\d{1,2}[A-Z]{3}(\d{5})(CE|PE)$", norm)
            if match is None:
                continue
            strike = int(match.group(1))
            if strike in target_strikes:
                selected.append((norm, strike))
        self._logger.info(
            "CANDIDATE_RESOLVER_FALLBACK source=tracked_symbols reason=resolver_unavailable count=%s",
            len(selected),
            extra={"event": "CANDIDATE_RESOLVER_FALLBACK", "source": "tracked_symbols", "reason": "resolver_unavailable", "count": len(selected)},
        )
        if not selected:
            self._logger.warning(
                "CANDIDATE_SNAPSHOT_BUILD_FAILED reason=resolver_empty",
                extra={
                    "event": "CANDIDATE_SNAPSHOT_BUILD_FAILED",
                    "reason": "resolver_empty",
                },
            )
        return selected

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
                "runner_state": str(self._runner_state),
                "active_symbols": sorted(self._active_symbols),
                "symbols": symbols,
                "signal_count": getattr(self, "_signal_counter", 0),
                "tick_count": getattr(self, "_eval_counter", 0),
                "last_tick_age_sec": round(
                    time.monotonic() - self._last_tick_seen_ts, 1
                ) if getattr(self, "_last_tick_seen_ts", 0) > 0 else None,
                "last_eval_age_sec": round(
                    time.monotonic() - self._last_global_eval_ts, 1
                ) if getattr(self, "_last_global_eval_ts", 0) > 0 else None,
            }

        # ── Pipeline health (non-blocking, best-effort) ──────────────────────
        try:
            from nifty_scalper_bot.data.pipeline import (  # noqa: PLC0415
                get_pipeline,
                get_dropped_ticks,
                get_dropped_candles,
            )
            _pl = get_pipeline()
            _pl_syms = _pl.store.symbols()
            status["pipeline"] = {
                "dropped_ticks": get_dropped_ticks(),
                "dropped_candles": get_dropped_candles(),
                "candle_counts": {
                    sym: len(_pl.store.get(sym)) for sym in _pl_syms
                },
                "symbols_tracked": len(_pl_syms),
                "ready_symbols": sum(
                    1 for sym in _pl_syms if _pl.candles_ready(sym, 50)
                ),
            }
        except Exception as _exc:
            status["pipeline"] = {"error": str(_exc)}

        return status

    # ==================== INTERNAL HELPERS ====================

    def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to tick updates for a symbol."""
        normalized_symbol = normalize_symbol(symbol)
        if self._data_hub is not None:
            if normalized_symbol not in self._datahub_registered_symbols:
                self._data_hub.subscribe_ticks(symbol, self.on_datahub_tick)
                self._datahub_registered_symbols.add(normalized_symbol)
            return
        callback = self._callbacks.get(symbol)
        callback_already_registered = callback is not None
        if callback is None:

            def _callback(tick: Mapping[str, Any], sym: str = symbol) -> None:
                try:
                    # 1. Guard against empty or malformed ticks (prevents dict() TypeError)
                    if not tick:
                        return

                    # 2. Defensive copy to a mutable dictionary
                    payload = dict(tick)
                    payload["symbol"] = sym

                    # Lightweight per-tick trace id so we can follow a tick
                    # from callback → event bus → runner → evaluation → order.
                    try:
                        _trace_id = f"{sym}-{time_module.monotonic_ns()}"
                    except Exception:  # noqa: BLE001
                        _trace_id = f"{sym}-{int(time.time()*1e9)}"
                    payload.setdefault("trace_id", _trace_id)

                    # Observability: runner received tick from DataHub/MDM
                    try:
                        _price = (
                            tick.get("ltp")
                            or tick.get("last_price")
                            or tick.get("price")
                        )
                        self._logger.debug(
                            "RUNNER_CALLBACK_TICK symbol=%s trace_id=%s price=%s",
                            sym,
                            _trace_id,
                            _price,
                            extra={
                                "event": "RUNNER_CALLBACK_TICK",
                                "symbol": sym,
                                "trace_id": _trace_id,
                                "price": _price,
                                "payload_keys": sorted(list(payload.keys())),
                                "published_to_event_bus": False,
                            },
                        )
                    except Exception:  # pragma: no cover - defensive
                        pass

                    # Dispatch directly to evaluation path; event-bus fanout is
                    # kept best-effort for observability compatibility.
                    self._on_tick_safe(payload)
                    publish_scheduled = False
                    try:
                        self._event_bus.publish(payload)
                        publish_scheduled = True
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "Runner local event bus publish failed for %s: %s",
                            sym,
                            exc,
                        )
                    self._logger.debug(
                        "RUNNER_CALLBACK_DISPATCH symbol=%s mode=%s success=%s",
                        sym,
                        "direct",
                        True,
                        extra={
                            "event": "RUNNER_CALLBACK_DISPATCH",
                            "symbol": sym,
                            "trace_id": _trace_id,
                            "dispatch_mode": "direct",
                            "publish_scheduled": publish_scheduled,
                            "success": True,
                            "reason": "direct_tick_consume",
                        },
                    )
                except Exception:
                    # Capture full stack trace AND the raw tick data for senior-level debugging
                    self._logger.exception(
                        "CRITICAL: Failure in StrategyRunner._subscribe_symbol callback for %s. Raw tick: %s",
                        sym,
                        tick,
                    )
                    self._logger.debug(
                        "RUNNER_CALLBACK_DISPATCH symbol=%s mode=%s success=%s",
                        sym,
                        "direct",
                        False,
                        extra={
                            "event": "RUNNER_CALLBACK_DISPATCH",
                            "symbol": sym,
                            "dispatch_mode": "direct",
                            "publish_scheduled": False,
                            "success": False,
                            "reason": "callback_exception",
                        },
                    )

            callback = _callback
            self._callbacks[symbol] = callback

        self._safe_subscribe(
            symbol, callback, callback_already_registered=callback_already_registered
        )

    def has_datahub_subscription(self, symbol: str, token: int | None = None) -> bool:
        """Check whether DataHub callback is already registered. Args: symbol/token. Returns: bool. Raises: none."""
        _ = token
        return normalize_symbol(symbol) in self._datahub_registered_symbols

    def _safe_subscribe(
        self,
        symbol: str,
        callback: Callable[[Mapping[str, Any]], None],
        *,
        callback_already_registered: bool = False,
    ) -> None:
        """Subscribe symbol safely. Args: symbol/callback. Returns: None. Raises: None."""
        try:
            _trace_id = f"{symbol}-{time_module.monotonic_ns()}"
        except Exception:  # noqa: BLE001
            _trace_id = f"{symbol}-{int(time.time()*1e9)}"
        using_data_hub = self._data_hub is not None
        try:
            if self._data_hub is not None:
                self._data_hub.subscribe_ticks(symbol, callback)
                self._logger.info(
                    "RUNNER_SUBSCRIBE_REQUEST symbol=%s via=data_hub success=true",
                    symbol,
                    extra={
                        "event": "RUNNER_SUBSCRIBE_REQUEST",
                        "symbol": symbol,
                        "trace_id": _trace_id,
                        "using_data_hub": True,
                        "callback_registered": True,
                        "callback_already_registered": callback_already_registered,
                        "success": True,
                        "error": None,
                    },
                )
                return
            if self._legacy_tick_subscription_mode:
                self._market_data.subscribe(symbol, callback)
                self._logger.info(
                    "RUNNER_SUBSCRIBE_REQUEST symbol=%s via=market_data success=true",
                    symbol,
                    extra={
                        "event": "RUNNER_SUBSCRIBE_REQUEST",
                        "symbol": symbol,
                        "trace_id": _trace_id,
                        "using_data_hub": False,
                        "callback_registered": True,
                        "callback_already_registered": callback_already_registered,
                        "success": True,
                        "error": None,
                    },
                )
                return
            raise RuntimeError("DataHub unavailable and legacy subscription disabled")
        except Exception as exc:
            self._logger.error("SUBSCRIBE_FAIL %s: %s", symbol, exc)
            self._logger.error(
                "RUNNER_SUBSCRIBE_REQUEST symbol=%s success=false error=%s",
                symbol,
                exc,
                extra={
                    "event": "RUNNER_SUBSCRIBE_REQUEST",
                    "symbol": symbol,
                    "trace_id": _trace_id,
                    "using_data_hub": using_data_hub,
                    "callback_registered": False,
                    "callback_already_registered": callback_already_registered,
                    "success": False,
                    "error": str(exc),
                },
            )

    def _ingest_historical_bar_unlocked(self, data: dict[str, Any]) -> None:
        """Ingest one historical candle into runner state. Args: data. Returns: None. Raises: None."""
        try:
            # 1. Extract + normalise timestamp
            ts = data["timestamp"]

            # BUG-ε FIX: defensive string → datetime conversion.
            # If upstream parsing silently failed (bare except: pass) ts may
            # still be a string.  ts + timedelta would then raise TypeError.
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif not isinstance(ts, datetime):
                try:
                    ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                except Exception as _ts_err:
                    self._logger.error(
                        f"❌ Hydration Ingest: unparseable timestamp for "
                        f"{data.get('symbol')}: {ts!r} — {_ts_err}"
                    )
                    return

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

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
                self._data_phase.setdefault(symbol, "HYDRATION")
                if symbol not in self._symbol_state:
                    self._symbol_state[symbol] = SymbolRuntimeState(
                        symbol=symbol, history_limit=2000
                    )
                self._symbol_states.setdefault(symbol, SymbolState.DISCOVERED)

        except Exception as exc:
            self._logger.error(
                f"❌ Hydration Ingest Failed for {data.get('symbol')}: {exc}"
            )

    def ingest_historical_bar(self, data: dict) -> None:
        """Public API for startup hydration ingestion. Args: data. Returns: None. Raises: None."""
        self._startup_hydrated = True
        symbol = (
            self._normalize_symbol(data.get("symbol", ""))
            if isinstance(data, dict)
            else ""
        )
        normalized = (
            normalize_history_row(
                symbol,
                data,
                source=(
                    data.get("source", "historical")
                    if isinstance(data, dict)
                    else "historical"
                ),
            )
            if symbol
            else None
        )
        if normalized is None:
            return
        try:
            self._ingest_historical_bar_unlocked(normalized)
            indicator_count = 0
            if hasattr(self._indicator_engine, "history_count"):
                indicator_count = int(self._indicator_engine.history_count(symbol))
            self._logger.debug(
                "RUNNER_BAR_INGESTED symbol=%s source=%s runner_bars=%d indicator_bars=%d",
                symbol,
                normalized.get("source"),
                len(self._symbol_history.get(symbol, [])),
                indicator_count,
            )
        except Exception as e:
            self._logger.error(
                "RUNNER_BAR_INGEST_FAILED symbol=%s error=%s",
                symbol,
                e,
                extra={
                    "event": "RUNNER_BAR_INGEST_FAILED",
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            raise

    def reseed_history_from_bars(
        self,
        symbol: str,
        bars: Iterable[Mapping[str, Any]],
        source: str = "mdm_reseed",
        min_bars: int = 1,
    ) -> int:
        """Args: symbol/bars/source/min_bars. Returns: runner history count. Raises: Exception."""
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            if not normalized_symbol:
                return 0
            normalized_rows: dict[datetime, dict[str, Any]] = {}
            for row in bars or ():
                normalized = normalize_history_row(normalized_symbol, dict(row), source=source)
                if normalized is None:
                    continue
                ts_value = normalized.get("timestamp")
                if not isinstance(ts_value, datetime):
                    continue
                timestamp = ts_value.astimezone(timezone.utc).replace(microsecond=0)
                normalized_rows[timestamp] = {
                    "timestamp": timestamp,
                    "open": float(normalized["open"]),
                    "high": float(normalized["high"]),
                    "low": float(normalized["low"]),
                    "close": float(normalized["close"]),
                    "volume": int(normalized.get("volume", 0) or 0),
                }
            selected_rows = sorted(normalized_rows.values(), key=lambda item: item["timestamp"])
            one_minute_bars: list[OneMinuteBar] = []
            for row in selected_rows:
                start_ts = row["timestamp"]
                one_minute_bars.append(
                    OneMinuteBar(
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                        start=start_ts,
                        end=start_ts + timedelta(minutes=1),
                    )
                )
            with self._lock:
                self._symbol_history[normalized_symbol] = list(one_minute_bars[-2000:])
                if one_minute_bars:
                    self._last_bar_ts[normalized_symbol] = one_minute_bars[-1].start
                self._active_symbols.add(normalized_symbol)
                self._tracked_symbols.add(normalized_symbol)
                self._data_phase.setdefault(normalized_symbol, "HYDRATION")
            indicator_count = self._indicator_engine.replace_history(
                normalized_symbol,
                selected_rows,
                source=source,
                min_bars=min_bars,
            )
            target_min_bars = max(1, int(min_bars or 1))
            if indicator_count >= target_min_bars:
                self._set_symbol_hydration_state(normalized_symbol, SymbolState.READY)
                self._seed_pipeline_store(normalized_symbol)
                self._seed_candle_engine_from_history(normalized_symbol)
            else:
                self._set_symbol_hydration_state(normalized_symbol, SymbolState.DEGRADED)
            runner_count = len(self._symbol_history.get(normalized_symbol, []))
            self._logger.info(
                "RUNNER_HISTORY_RESEEDED symbol=%s runner_bars=%d indicator_bars=%d min_bars=%d source=%s",
                normalized_symbol,
                runner_count,
                indicator_count,
                target_min_bars,
                source,
            )
            return min(runner_count, int(indicator_count or 0))
        except Exception as e:
            self._logger.exception(
                "RUNNER_HISTORY_RESEED_FAILED symbol=%s source=%s error=%s",
                symbol,
                source,
                e,
            )
            raise

    async def safe_ingest(self, data: dict[str, Any]) -> None:
        """Synchronize hydration ingestion on event loop. Args: data. Returns: None. Raises: None."""
        self._startup_hydrated = True
        async with self._ingest_lock:
            self._ingest_historical_bar_unlocked(data)

    def mark_ready(self, symbols: list[str]) -> bool:
        """
        Public API to finalize startup hydration.
        Explicitly registers symbols and sets readiness flags.
        """
        valid_symbols: list[str] = []
        self._hydration_complete = True
        with self._lock:
            for sym in symbols:
                normalized = enforce_canonical(normalize_symbol(sym))
                # 1. Register Active (Critical for main loop)
                self._active_symbols.add(normalized)
                self._data_phase.setdefault(normalized, "HYDRATION")
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

                # FIX: _symbol_has_valid_data checked MDM._ohlc_builder (live ticks only —
                # always empty at startup) → valid_symbols=[] → EXECUTION_ENABLED never set
                # → every tick returned at Phase-6 gate → zero trades.
                # Correct gate: use IndicatorEngine.has_min_bars (reflects hydrated bars).
                # Require ≥1 bar so any hydrated symbol qualifies; the 20/50-bar minimum
                # is enforced at strategy evaluation time by indicator_engine.has_min_bars().
                has_history = (
                    self._indicator_engine is not None
                    and self._indicator_engine.has_min_bars(normalized, 1)
                )
                if has_history:
                    valid_symbols.append(normalized)
                    # FIX (STEP 2): Seed pipeline CandleStore from IndicatorEngine
                    # PriceHistory so pipeline.candles_ready() passes immediately on
                    # the first tick — no broker API calls needed for warmup.
                    # STEP 6: CandleStore.seed() is a no-op if already seeded.
                    self._seed_pipeline_store(normalized)
                    # FIX (legacy path): Also seed CandleEngine.df so ensure_valid_data
                    # passes on first tick (avoids _hydrate_missing_bars broker calls).
                    self._seed_candle_engine_from_history(normalized)
                else:
                    self._logger.info(
                        "mark_ready: no hydrated bars for symbol — leaving unready",
                        extra={"event": "mark_ready_no_bars", "symbol": normalized},
                    )

        # 5. Keep _startup_hydrated=True
        self._startup_hydrated = True
        self.ready = len(valid_symbols) >= self._required_symbol_count
        if self.ready:
            self._runner_state = RunnerState.EXECUTION_ENABLED
            self._logger.info("🚀 StrategyRunner execution enabled")
            self._logger.info(
                "execution_enabled",
                extra={
                    "event": "execution_enabled",
                    "valid_symbols": sorted(valid_symbols),
                    "required_symbol_count": self._required_symbol_count,
                },
            )
            self._logger.info(
                f"✅ StrategyRunner marked READY with {len(valid_symbols)} active symbols"
            )
        else:
            self._runner_state = (
                RunnerState.WARMING_UP
                if len(valid_symbols) == 0
                else RunnerState.HISTORICAL_READY
            )
            self._logger.info(
                "mark_ready: HISTORICAL_READY "
                f"(valid={len(valid_symbols)}/{len(symbols)}, "
                f"required={self._required_symbol_count})",
                extra={
                    "event": "mark_ready_historical_ready",
                    "valid_symbols": len(valid_symbols),
                    "total_symbols": len(symbols),
                    "required": self._required_symbol_count,
                },
            )

        # BUG W1 FIX: Removed deadlocking tick-wait that caused 21s startup delay.
        # mark_ready() is called from startup_sequence() (event-loop thread).
        # asyncio.run_coroutine_threadsafe(...).result(timeout=10.5) schedules a
        # coroutine on the SAME loop that is currently blocked on .result() →
        # wait_for_live_tick() can never yield → both calls always time out → bot
        # spends 21 extra seconds blocked before becoming EXECUTION_ENABLED.
        # Tick availability is already handled per-evaluation (spot_stale=True path).

        # REMOVE the hardcoded logger at the bottom of the function:
        # self._logger.info(f"✅ StrategyRunner marked READY with {len(symbols)} symbols")

        # BUG W2 FIX: Log per-symbol bar-count summary at mark_ready() so Railway logs
        # confirm exactly how many indicator bars each symbol has at the moment strategies
        # are unlocked.  This bridges the gap between "Indicators hydrated" (app.py) and
        # the per-symbol "WARMUP COMPLETE" log emitted on the first tick.
        for _sym in symbols:
            norm_sym = enforce_canonical(normalize_symbol(_sym))
            try:
                _bc = len(self._indicator_engine.get_history(norm_sym) or [])
                _ok = self._indicator_engine.has_min_bars(
                    norm_sym, self._required_candles
                )
            except Exception:
                _bc, _ok = 0, False
            self._logger.info(
                f"📊 WARMUP SUMMARY: {norm_sym} | bars={_bc} | "
                f"min_required={self._required_candles} | ready={_ok}",
                extra={
                    "event": "warmup_summary",
                    "symbol": norm_sym,
                    "bar_count": _bc,
                    "required": self._required_candles,
                    "ready": _ok,
                },
            )
        return self.ready

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

    def _repair_candle_gap(
        self,
        symbol: str,
        previous_bar: OneMinuteBar,
        incoming_bar: OneMinuteBar,
    ) -> list[OneMinuteBar]:
        """Recover missing candles. Args: symbol, previous_bar, incoming_bar; Returns: recovered bars; Raises: none."""
        repaired: list[OneMinuteBar] = []
        gap_seconds = max(
            0.0,
            (incoming_bar.timestamp - previous_bar.timestamp).total_seconds(),
        )
        upper_symbol = symbol.upper()
        # Sparse option contracts can legitimately pause for a few minutes between prints.
        if ("CE" in upper_symbol or "PE" in upper_symbol) and gap_seconds < 180.0:
            return repaired
        if gap_seconds <= 180.0:
            return repaired
        expected = previous_bar.timestamp + timedelta(minutes=1)
        history_cache = self._load_history_cache(symbol)
        cache_by_minute: dict[datetime, dict[str, Any]] = {}
        for row in history_cache:
            row_ts = row.get("timestamp")
            if not isinstance(row_ts, datetime):
                continue
            ts_utc = row_ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
            cache_by_minute[ts_utc] = row

        is_option_symbol = upper_symbol.endswith("CE") or upper_symbol.endswith("PE")
        prev_close = float(previous_bar.close)
        while expected < incoming_bar.timestamp:
            row = cache_by_minute.get(expected)
            if row:
                close = float(row.get("close") or prev_close)
                repaired_bar = OneMinuteBar(
                    open=float(row.get("open") or prev_close),
                    high=float(row.get("high") or close),
                    low=float(row.get("low") or close),
                    close=close,
                    volume=max(int(float(row.get("volume") or 0)), 0),
                    start=expected,
                    end=expected + timedelta(seconds=59),
                    synthetic=False,
                )
            elif not is_option_symbol:
                repaired_bar = OneMinuteBar(
                    open=prev_close,
                    high=prev_close,
                    low=prev_close,
                    close=prev_close,
                    volume=0,
                    start=expected,
                    end=expected + timedelta(seconds=59),
                    synthetic=True,
                )
            else:
                expected += timedelta(minutes=1)
                continue
            repaired.append(repaired_bar)
            prev_close = repaired_bar.close
            expected += timedelta(minutes=1)

        if repaired:
            if self._should_log_throttled(f"candle_gap_repaired:{symbol}", 300.0):
                self._logger.info(
                    "candle_gap_repaired",
                    extra={
                        "event": "candle_gap_repaired",
                        "symbol": symbol,
                        "count": len(repaired),
                    },
                )
            if (
                self._main_loop
                and self._main_loop.is_running()
                and symbol not in self._gap_repair_inflight
            ):
                self._gap_repair_inflight.add(symbol)
                self._request_mdm_hydration(symbol, self._required_bars_for_symbol(symbol))
        return repaired

    async def _refresh_gap_history_async(self, symbol: str) -> None:
        """Refresh repaired symbol history from MDM cache only. Args: symbol. Returns: None. Raises: None."""
        target = self._required_bars_for_symbol(symbol)
        try:
            rows = self._get_mdm_bars(symbol, target)
            if rows:
                self._write_history_cache(symbol, rows)
                for row in rows:
                    with suppress(Exception):
                        self.ingest_historical_bar(row)
            else:
                self._request_mdm_hydration(symbol, target)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("gap_history_refresh_failed for %s: %s", symbol, exc)
        finally:
            self._gap_repair_inflight.discard(symbol)

    def _refresh_history_if_due(self, symbol: str) -> None:
        """Trigger periodic historical refresh. Args: symbol. Returns: None. Raises: None."""
        if not symbol or self._main_loop is None or not self._main_loop.is_running():
            return
        now_mono = time.monotonic()
        last_refresh = self._last_history_refresh_by_symbol.get(symbol, 0.0)
        if now_mono - last_refresh < self._history_refresh_interval_seconds:
            return
        self._last_history_refresh_by_symbol[symbol] = now_mono
        self._logger.info(
            "Condition met: historical_refresh_triggered",
            extra={"event": "historical_refresh_triggered", "symbol": symbol},
        )
        self._request_mdm_hydration(symbol, self._required_bars_for_symbol(symbol))

    def _emit_composite_reports(self) -> None:
        """Emit periodic system/strategy aggregate logs. Args: none; Returns: none; Raises: none."""
        now = time.monotonic()
        if now - self._last_system_heartbeat_log >= 120.0:
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
                stale_symbols = [
                    k for k in self._symbol_history if not self._symbol_history.get(k)
                ]
                for key in stale_symbols:
                    self._symbol_history.pop(key, None)
                with self._orders_lock:
                    stale_inflight = [
                        k
                        for k, ts in self._orders_in_flight.items()
                        if (time.time() - ts) > 3600.0
                    ]
                    for key in stale_inflight:
                        self._orders_in_flight.pop(key, None)
                with self._trade_counter_lock:
                    stale_candles = [
                        key
                        for key in self._trade_counter_by_symbol_candle
                        if key[1] < cutoff
                    ]
                    for key in stale_candles:
                        self._trade_counter_by_symbol_candle.pop(key, None)
            except Exception as exc:
                self._logger.error(
                    "Failure in StrategyRunner._emit_composite_reports: %s", exc
                )
            open_positions = 0
            if hasattr(self._position_manager, "get_all_positions"):
                try:
                    open_positions = len(
                        self._position_manager.get_all_positions() or []
                    )
                except Exception:
                    open_positions = 0
            self._logger.info(
                "SYSTEM_HEARTBEAT",
                extra={
                    "event": "system_heartbeat",
                    "active_symbols": len(self._active_symbols),
                    "tick_flow": (
                        "ACTIVE" if (now - self._last_tick_seen_ts) <= 5.0 else "STALE"
                    ),
                    "last_tick_age_s": round(
                        max(0.0, now - self._last_tick_seen_ts), 2
                    ),
                    "candle_builder": "HEALTHY",
                    "indicator_engine": (
                        "HEALTHY"
                        if self._indicator_engine is not None
                        else "UNAVAILABLE"
                    ),
                    "strategy_runner": (
                        "ACTIVE"
                        if self._running and not self._trading_paused
                        else "PAUSED"
                    ),
                    "open_positions": open_positions,
                    "data_integrity": "OK",
                },
            )
            self._last_system_heartbeat_log = now

        if now - self._last_strategy_status_log >= 150.0:
            self._logger.info(
                "STRATEGY_STATUS_REPORT",
                extra={
                    "event": "strategy_status_report",
                    "symbols_evaluated": len(self._strategy_window_symbols),
                    "signals_generated": int(self._strategy_window_signals),
                    "trailing_updates": int(self._strategy_window_trailing_updates),
                    "positions_active": len(
                        getattr(
                            self._position_manager, "get_all_positions", lambda: []
                        )()
                        or []
                    ),
                },
            )
            self._strategy_window_symbols.clear()
            self._strategy_window_signals = 0
            self._strategy_window_trailing_updates = 0
            self._last_strategy_status_log = now

    def _has_session_candle_gaps(self, symbol: str) -> bool:
        """Return True when RECENT live session history has timestamp gaps.

        Only the last 90 minutes of today's session bars are examined.
        Historical hydration bars span multiple days and always leave a
        gap at the hydration-to-live boundary (typically 5-30 min at
        market open).  Including that boundary in the gap check would
        permanently degrade every symbol at startup, blocking all signal
        evaluation.  Restricting the window to 90 minutes means:
          • The hydration gap is ignored on startup.
          • Any real mid-session data outage (>2 min gap) is still caught.
          • Symbols recover automatically once continuous live bars arrive.
        """
        history = self._symbol_history.get(symbol, [])
        if len(history) < 2:
            self._session_gap_count[symbol] = 0
            return False
        now_utc = datetime.now(timezone.utc)
        session_date = now_utc.date()
        cutoff = now_utc - timedelta(minutes=90)
        # Only inspect today's bars that fall within the 90-minute window.
        session_bars = [
            bar
            for bar in history
            if bar.timestamp.date() == session_date and bar.timestamp >= cutoff
        ]
        if len(session_bars) < 2:
            # Not enough recent bars to assess gaps — treat as gap-free.
            self._session_gap_count[symbol] = 0
            return False
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
            vwap_state = dict(self._vwap_state.get(symbol, {}))
        indicators = {
            symbol: {
                "vwap": vwap_value,
                "cum_volume": float(vwap_state.get("cum_vol", 0.0)),
            }
        }
        return self.update_symbol_hydration(symbol, bars, indicators)

    def _update_symbol_readiness(self, symbol: str) -> SymbolState:
        """Update lifecycle state from bar history and cumulative VWAP health.

        Uses _symbol_history for volume/VWAP when live bars exist.
        When _symbol_history is empty (startup, before first live minute bar),
        falls back to indicator_engine bar count (which includes the 1125 hydrated
        bars loaded at startup) so the bar-count gate does not falsely return HYDRATING.
        """
        with self._lock:
            bars = list(self._symbol_history.get(symbol, []))
            state = self._symbol_state.get(symbol)
            vwap_snapshot = dict(self._vwap_state.get(symbol, {}))
        vol_sum = sum(float(getattr(bar, "volume", 0)) for bar in bars)
        vwap_val = float(state.vwap) if state and state.vwap else 0.0

        # When no live bars exist yet, try to supplement VWAP/volume from the
        # tick-based accumulator so the hydration check has real numbers to work with.
        if not bars:
            vwap_state = vwap_snapshot
            cum_vol = float(vwap_state.get("cum_vol", 0.0))
            cum_pv = float(vwap_state.get("cum_pv", 0.0))
            vol_sum = cum_vol
            if vwap_val == 0.0 and cum_vol > 0.0:
                vwap_val = cum_pv / cum_vol

        indicators = {symbol: {"vwap": vwap_val, "cum_volume": vol_sum}}

        # ── Bar-count: prefer live bars; fall back to indicator_engine ──────────
        # update_symbol_hydration uses len(bars) as the bar-count gate.  Passing
        # the raw (empty) _symbol_history list at startup gives bar_count=0, which
        # is always < required_candles → HYDRATING — overriding the READY state
        # that mark_ready() already set.  If the indicator engine holds enough bars
        # (it does after startup hydration), synthesise a dummy list of that length.
        # update_symbol_hydration never iterates bar items — it only calls len().
        effective_bars: list = bars
        if len(bars) < self._required_candles and self._indicator_engine is not None:
            try:
                ind_history = self._indicator_engine.get_history(symbol) or []
                if len(ind_history) >= self._required_candles:
                    # Provide a list of the correct length; individual items unused.
                    effective_bars = [None] * len(ind_history)  # type: ignore[list-item]
            except Exception as e:
                LOGGER.exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise  # fall through to actual bars — hydration will be HYDRATING

        return self.update_symbol_hydration(symbol, effective_bars, indicators)

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
            is_option = symbol.startswith("NFO:") and symbol.endswith(("CE", "PE"))
            last_tick_ts = float(self._last_tick_time_by_symbol.get(symbol, 0.0) or 0.0)
            recent_tick = last_tick_ts > 0 and (time.time() - last_tick_ts) <= 120.0
            tick_age_s = round(time.time() - last_tick_ts, 2) if recent_tick else None
            if gap_count > 1:
                reason = "repeated_missing_candles" if recent_tick else "no_recent_tick_for_gap_assessment"
                log_level = logging.INFO if is_option and recent_tick else logging.WARNING
                log_throttled(
                    self._logger,
                    f"soft_data_issue:{symbol}",
                    f"SOFT_DATA_ISSUE symbol={symbol} reason={reason} source=candle_gap_detector age_s={tick_age_s}",
                    interval_sec=60.0,
                    level=log_level,
                    extra={"event": "SOFT_DATA_ISSUE", "symbol": symbol, "reason": reason, "source": "candle_gap_detector", "age_s": tick_age_s, "details": {"gaps": gap_count}, "gaps": gap_count},
                )
                if gap_count > 1 and recent_tick:
                    return self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)
            else:
                reason = "single_missing_candle" if recent_tick else "no_recent_tick_for_gap_assessment"
                log_throttled(
                    self._logger,
                    f"soft_data_issue:{symbol}",
                    f"SOFT_DATA_ISSUE symbol={symbol} reason={reason} source=candle_gap_detector age_s={tick_age_s}",
                    interval_sec=60.0,
                    level=logging.INFO,
                    extra={"event": "SOFT_DATA_ISSUE", "symbol": symbol, "reason": reason, "source": "candle_gap_detector", "age_s": tick_age_s, "details": {"gaps": gap_count}},
                )
                if gap_count > 1 and recent_tick:
                    return self._set_symbol_hydration_state(symbol, SymbolState.DEGRADED)

        if valid_vwap and valid_volume:
            self._hydration_ready_streak[symbol] = int(self._hydration_ready_streak.get(symbol, 0)) + 1
        else:
            self._hydration_ready_streak[symbol] = max(1, int(self._hydration_ready_streak.get(symbol, 0)))
        return self._set_symbol_hydration_state(symbol, SymbolState.READY)

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
            # Reset the VWAP-readiness streak so a new day's VWAP must be validated
            # from scratch before being considered "ready".
            self._hydration_ready_streak[symbol] = 0
            # ── FIX (2026-02-26): DO NOT set hydration state to HYDRATING here.
            # _ensure_symbol_vwap_state is responsible for VWAP session data only.
            # Resetting hydration state here with allow_downgrade=True was forcibly
            # overriding the READY state set by mark_ready() on the first tick of
            # every trading day, causing all strategy evaluation to be blocked until
            # the bar-count check slowly promoted back.  Hydration state transitions
            # are the responsibility of _update_symbol_readiness, not the VWAP layer.
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
            self._logger.warning(
                "Dropping out-of-order candle",
                extra={"symbol": symbol, "bar_ts": bar.timestamp, "last_ts": last_ts},
            )
            return

        # 2. STATE: Update High-Water Mark
        history = self._symbol_history.setdefault(symbol, [])
        if not is_backfill and last_ts and history:
            expected_next = last_ts + timedelta(minutes=1)
            if bar.timestamp > expected_next:
                repaired = self._repair_candle_gap(symbol, history[-1], bar)
                for synthetic_bar in repaired:
                    self._ingest_bar(symbol, synthetic_bar, is_backfill=True)

        if not last_ts or bar.timestamp > last_ts:
            self._last_bar_ts[symbol] = bar.timestamp
        history.append(bar)
        if not is_backfill:
            self._candle_versions[symbol] += 1
        if len(history) > 400:
            del history[:-400]

        try:
            # 3. INDICATORS: Feed the Engine
            # [FIX] Use update_price() instead of update_bar().
            if hasattr(self._indicator_engine, "update_bar"):
                self._indicator_engine.update_bar(symbol, bar)
            else:
                # FIX S12-3: was float(bar.close) — scalar input causes _normalize_price
                # to use close for ALL of open/high/low/close, destroying ATR/EMA accuracy.
                # Pass bar.as_mapping() so the full OHLC is stored correctly.
                synthetic_gap_fill = bool(getattr(bar, "synthetic", False))
                indicator_volume = 0 if synthetic_gap_fill else bar.volume
                self._indicator_engine.update_price(
                    symbol,
                    bar.as_mapping(),
                    volume=indicator_volume,
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
                # BUG W3 FIX: state.vwap was never seeded during historical ingest because
                # the full VWAP accumulator block below requires is_backfill=False. After
                # warmup, _update_symbol_readiness read state.vwap=None → valid_vwap=False
                # → symbol permanently DEGRADED with valid_vwap=False until 2+ live bars.
                # In strategy_manager.generate_signal(), vwap=None from indicators +
                # exchange_vwap=0 from a not-yet-traded option → invalid_reason="vwap_zero"
                # → symbol suspended after 10 consecutive evaluations. Fix: after feeding
                # the bar to indicator_engine, read the computed VWAP back and seed state.vwap
                # once for the LAST historical bar so readiness check has a valid VWAP to use.
                with self._lock:
                    state = self._symbol_state.get(symbol)
                    if state and state.vwap is None:
                        # Use indicator_engine VWAP (computed over historical close*volume).
                        # Falls back to bar.close if all historical bars had zero volume
                        # (new weekly-expiry option with no prior-day trades).
                        ie_vwap = None
                        try:
                            ie_vwap = self._indicator_engine.get_vwap(symbol)
                        except Exception as e:
                            LOGGER.exception(
                                "[CRITICAL] unhandled exception", exc_info=True
                            )
                            raise
                        state.vwap = (
                            ie_vwap if ie_vwap and ie_vwap > 0 else float(bar.close)
                        )
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
                    # ✅ FIX: Removed _last_cumulative_volume overwrite. cum_vol is
                    # the VWAP accumulator (sum of per-bar deltas), NOT the exchange raw
                    # cumulative volume. Overwriting here corrupts future delta calculations.
                    pass

            # 🔥 THE TRIGGER: Run Strategy Logic
            # [FIX] Removed .on_bar() call as StrategyManager is signal-driven (via ticks), not bar-driven.
            # BUG W1 FIX: Mark symbol as having seen a live bar.  This flag is used by
            # the PHASE-9 stale-bar gate instead of _symbol_history (which contains old
            # hydration bars) so the gate only activates after the FIRST real minute bar
            # completes — not immediately at startup.
            if bar and symbol not in self._live_bar_seen:
                self._mark_live(symbol)
            return

        except Exception as exc:
            self._logger.error(
                f"Failure in _ingest_bar: {exc}",
                exc_info=True,
                extra={"event": "ingest_bar_failed", "symbol": symbol},
            )

    def _mark_live(self, symbol: str) -> None:
        """Mark symbol live phase. Args: symbol. Returns: None. Raises: None."""
        try:
            if self._data_phase.get(symbol) != "LIVE":
                self._data_phase[symbol] = "LIVE"
                self._live_bar_seen.add(symbol)
                self._logger.info(
                    "LIVE_MODE_ENABLED symbol=%s",
                    symbol,
                    extra={"event": "LIVE_MODE_ENABLED", "symbol": symbol},
                )
        except Exception as e:
            self._logger.error("Failure in StrategyRunner._mark_live: %s", e)

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

    def _validate_long_option_geometry(
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
                    # TRUE conflict: opposing directional signals (BUY vs SELL, or
                    # CLOSE_LONG vs CLOSE_SHORT). Drop these — strategies genuinely disagree.
                    directional = {"BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT"}
                    active_directions = actions & directional
                    # Opposing pairs that cannot be reconciled:
                    is_true_conflict = (
                        "BUY" in active_directions and "SELL" in active_directions
                    ) or (
                        "CLOSE_LONG" in active_directions
                        and "CLOSE_SHORT" in active_directions
                    )
                    if is_true_conflict:
                        self._logger.error(
                            "Conflicting signals detected for %s: %s — dropping",
                            symbol,
                            sorted_actions,
                            extra={
                                "event": "signal_conflict",
                                "symbol": symbol,
                                "actions": sorted_actions,
                            },
                        )
                        continue
                    # Non-conflict mix (e.g. BUY + HOLD): keep highest-confidence
                    # directional signal rather than discarding the whole batch.
                    directional_signals = [
                        s for s in symbol_signals if s.action in directional
                    ]
                    if directional_signals:
                        best = max(
                            directional_signals,
                            key=lambda s: self._normalize_confidence(s.confidence),
                        )
                        self._logger.info(
                            "Mixed actions for %s (%s) — using highest-confidence: %s",
                            symbol,
                            sorted_actions,
                            best.action,
                            extra={"event": "signal_mixed_resolved", "symbol": symbol},
                        )
                        aggregated[symbol] = best
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

        except Exception as e:
            LOGGER.exception(
                "[CRITICAL] unhandled exception", exc_info=True
            )
            raise

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

    def _build_option_score_config(self, side: Literal["BUY", "SELL"] ) -> Mapping[str, Any]:
        """Return strike selector score configuration for the supplied side."""
        return {"weights": dict(self._option_score_weights), "delta_target": float(self._option_delta_target), "max_iv_rank": float(self._option_max_iv_rank), "min_liquidity": float(self._option_min_liquidity), "side": side, }

    def _get_spot_tick(self) -> dict[str, Any] | None:
        """Resilient spot tick fetcher with aliases and snapshot fallback."""
        aliases = ("NSE:NIFTY", "NIFTY", "NIFTY50", "NSE:NIFTY 50")
        for alias in aliases:
            if self._market_data:
                tick = self._market_data.get_latest_tick(alias)
                if tick:
                    return tick
            if self._data_hub:
                tick = getattr(self._data_hub, "get_latest_tick", lambda *_: None)(alias)
                if tick:
                    return tick
                tick = getattr(self._data_hub, "get_quote", lambda *_: None)(alias)
                if tick:
                    return tick
        if self._market_data and hasattr(self._market_data, "get_symbol_snapshot"):
            snap = self._market_data.get_symbol_snapshot("NSE:NIFTY")
            if snap and getattr(snap, "ltp", None):
                tick_age = float(getattr(snap, "tick_age_s", 0.0) or 0.0)
                return {
                    "symbol": "NSE:NIFTY",
                    "ltp": float(getattr(snap, "ltp")),
                    "received_at": time.time() - tick_age,
                    "source": getattr(snap, "source", "market_data_snapshot"),
                }
            if isinstance(snap, Mapping):
                ltp = snap.get("ltp") or snap.get("close")
                if ltp:
                    return {
                        "symbol": "NSE:NIFTY",
                        "ltp": float(ltp),
                        "received_at": snap.get("received_at") or snap.get("timestamp") or time.time(),
                        "source": snap.get("source", "market_data_snapshot"),
                    }
        if self._market_data and hasattr(self._market_data, "get_cached_ltp"):
            ltp = self._market_data.get_cached_ltp("NSE:NIFTY", max_age_seconds=300, require_ws=False)
            if ltp:
                return {"symbol": "NSE:NIFTY", "ltp": ltp, "received_at": time.time(), "source": "market_data_cached_ltp"}
        return None

    def _get_spot_price(self) -> float:
        """Get cached-only spot. Args: none. Returns: price. Raises: none."""
        for source in (self._data_hub, self._market_data):
            if source is None:
                continue
            cached_fn = getattr(source, "get_cached_ltp", None)
            if callable(cached_fn):
                try:
                    price = cached_fn(
                        "NSE:NIFTY", max_age_seconds=300.0, require_ws=False
                    )
                    if price and price > 0:
                        return float(price)
                except Exception:
                    pass
            get_latest_price = getattr(source, "get_latest_price", None)
            if callable(get_latest_price):
                try:
                    price = get_latest_price("NSE:NIFTY", allow_pull=False)
                    if price and price > 0:
                        return float(price)
                except TypeError:
                    pass
                except Exception:
                    pass
        return 0.0

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
        if self._order_manager is not None and hasattr(self._order_manager, "resolve_lot_size"):
            try:
                lot_size = max(1, int(self._order_manager.resolve_lot_size(symbol)))
            except Exception:
                lot_size = 1
        elif metadata is not None:
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
        spot_tick = self._get_spot_tick()
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
            quote = tick or {}
            bid_price = float(
                quote.get("best_bid")
                or quote.get("bid")
                or quote.get("best_bid_price")
                or 0.0
            )
            ask_price = float(
                quote.get("best_ask")
                or quote.get("ask")
                or quote.get("best_ask_price")
                or 0.0
            )
            bid_qty = int(float(quote.get("bid_quantity") or quote.get("bid_qty") or 0))
            ask_qty = int(float(quote.get("ask_quantity") or quote.get("ask_qty") or 0))
            if bid_price > 0 and ask_price > 0:
                spread = max(0.0, ask_price - bid_price)
                spread_pct = (spread / max(price, 1e-6)) * 100.0
                if spread_pct > 1.5:
                    raise RuntimeError("Execution blocked due to option spread guard")
            if (bid_qty + ask_qty) < 300:
                raise RuntimeError("Execution blocked due to liquidity guard")
            tick_size = 0.05
            order_price = price
            if side == "BUY" and ask_price > 0:
                order_price = ask_price + tick_size
            elif side == "SELL" and bid_price > 0:
                order_price = max(tick_size, bid_price - tick_size)
            order_id = ""
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    plan = TradePlan(symbol=symbol, side=side, quantity=normalized_qty, entry_price=order_price, stop_loss=stop_loss, take_profit=take_profit, strategy_name="runner", tag=f"runner_{side.lower()}", allow_market_entry=False)
                    order_id = self._order_manager.submit_trade_plan(plan)
                    break
                except Exception as exc:
                    if attempt >= (max_attempts - 1):
                        raise
                    adjust = tick_size * float(attempt + 1)
                    order_price = (
                        order_price + adjust
                        if side == "BUY"
                        else max(tick_size, order_price - adjust)
                    )
                    self._logger.warning(
                        "order_retry_adjusted",
                        extra={
                            "event": "order_retry_adjusted",
                            "symbol": symbol,
                            "side": side,
                            "attempt": attempt + 2,
                            "error": str(exc),
                            "adjusted_price": order_price,
                        },
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

            except Exception as e:
                LOGGER.exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise

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

            except Exception as e:
                LOGGER.exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise

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

    def _is_order_in_flight(self, trade_symbol: str, base_symbol: str) -> bool:
        """Return whether a fresh order is currently in flight for the symbol group."""
        key = trade_symbol or base_symbol
        now = time.time()

        with self._orders_lock:
            for symbol in (key, base_symbol):
                ts = self._orders_in_flight.get(symbol)
                if ts is None:
                    continue
                if now - ts > self._order_timeout_sec:
                    del self._orders_in_flight[symbol]
                    continue
                return True
        return False

    def _acquire_order_in_flight(
        self, symbol: str, underlying: str | None = None
    ) -> bool:
        """Atomically acquire in-flight guards; Args: symbol/underlying. Returns: bool. Raises: None."""
        now = time.time()
        with self._orders_lock:
            for key in (symbol, underlying):
                if not key:
                    continue
                ts = self._orders_in_flight.get(key)
                if ts is not None and now - ts <= self._order_timeout_sec:
                    return False
            self._orders_in_flight[symbol] = now
            self.orders_in_flight.add(symbol)
            if underlying and underlying != symbol:
                self._orders_in_flight[underlying] = now
                self.orders_in_flight.add(underlying)
        return True

    def _mark_order_in_flight(self, symbol: str, underlying: str | None = None) -> None:
        """Record a fresh in-flight order timestamp for symbol and optional underlying."""
        now = time.time()
        with self._orders_lock:
            self._orders_in_flight[symbol] = now
            if underlying and underlying != symbol:
                self._orders_in_flight[underlying] = now

    def _clear_order_in_flight(self, symbol: str) -> None:
        """Clear order in-flight marker for symbol and derived underlying key."""
        with self._orders_lock:
            self._orders_in_flight.pop(symbol, None)
            self.orders_in_flight.discard(symbol)
            try:
                underlying = self._normalize_symbol(symbol)
                if underlying and underlying != symbol:
                    self._orders_in_flight.pop(underlying, None)
                    self.orders_in_flight.discard(underlying)
            except Exception as e:
                LOGGER.exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise

    def _get_execution_state_machine(self, symbol: str) -> OrderStateMachine:
        """Return symbol state machine. Args: symbol. Returns: OrderStateMachine. Raises: none."""
        with self._execution_state_lock:
            if symbol not in self._execution_state_by_symbol:
                self._execution_state_by_symbol[symbol] = OrderStateMachine()
            return self._execution_state_by_symbol[symbol]

    def _transition_execution_state(
        self, symbol: str, new_state: ExecutionState
    ) -> bool:
        """Apply state transition. Args: symbol, new_state. Returns: bool. Raises: none."""
        machine = self._get_execution_state_machine(symbol)
        return machine.transition(new_state)

    def _reset_execution_state(self, symbol: str) -> None:
        """Reset execution state to IDLE. Args: symbol. Returns: none. Raises: none."""
        machine = self._get_execution_state_machine(symbol)
        machine.force_idle()

    # Cooldown logic moved to ExecutionEngine

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
        now_ts = time.time()
        tick_timestamp = _extract_timestamp(message.data, datetime.now(timezone.utc))
        tick_age_ms = max(
            0.0,
            (now_ts - tick_timestamp.astimezone(timezone.utc).timestamp()) * 1000.0,
        )
        
        # CRITICAL FIX 1: Relax stale data guard to 15 seconds.
        # Options do not tick as fast as the spot index. 3 seconds is too tight
        # and will cause perfectly valid entry signals to be dropped.
        if tick_age_ms > 15000.0:  
            log_throttled(
                self._logger,
                "stale_reconnect_tick_drop",
                (
                    "Condition met: stale_reconnect_tick_drop "
                    f"age_ms={tick_age_ms:.1f}"
                ),
                interval_sec=15.0,
                level=logging.DEBUG,
            )
            return

        symbol_value = tick.get("symbol")
        symbol = (
            enforce_canonical(normalize_symbol(str(symbol_value)))
            if symbol_value
            else "UNKNOWN"
        )
        self._logger.debug("RUNNER_RECEIVED_TICK %s", symbol)
        
        # CRITICAL FIX 2: Throttling to prevent ThreadPool Exhaustion and 90s deadlocks
        if not hasattr(self, "_last_eval_time"):
            self._last_eval_time = {}
        
        last_eval = self._last_eval_time.get(symbol, 0.0)
        
        # Limit processing to 1 tick per symbol per second to unblock the event loop
        # The DataHub still records the tick price, but we skip the heavy strategy math
        if now_ts - last_eval < 1.0:
            return
            
        self._last_eval_time[symbol] = now_ts
        
        try:
            # Offload heavy synchronous processing (and blocking broker calls) to a thread
            # We are now safely spawning a maximum of 1 thread per symbol, per second.
            await asyncio.to_thread(self._on_tick_safe, tick)
        except Exception as exc:
            # Fixed variable reference: Ensure we use self._logger instead of undefined LOGGER
            self._logger.error(f"Error in async tick processing: {exc}", exc_info=True)

    def _strategy_worker(self) -> None:
        """Consume events and run tick evaluation. Args: none. Returns: none. Raises: none."""
        for event in self._event_bus.consume():
            if not self._running:
                continue
            try:
                self._on_tick_safe(cast(Mapping[str, Any], event))
            except Exception as e:
                self._logger.error("Failure in StrategyRunner._strategy_worker: %s", e)

    def _is_market_open(self, now: datetime) -> bool:
        """Return True only when market state is OPEN."""
        try:
            now_ist = now.astimezone(_IST) if now.tzinfo else now.replace(tzinfo=_IST)
            within_session = dt_time(9, 15) <= now_ist.time() <= dt_time(15, 30)
            settings = get_settings()
            env_name = str(getattr(settings, "environment", "")).lower()
            if env_name == "production":
                return within_session and get_market_state() == MarketState.OPEN
            if bool(getattr(settings, "allow_offmarket_trading", False)):
                return True
            return within_session and get_market_state() == MarketState.OPEN
        except Exception as e:
            self._logger.warning(
                f"Market time check failed: {e}. Defaulting to CLOSED."
            )
            return False

    def _on_tick_from_bus(self, tick: Mapping[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol_value = tick.get("symbol")
            trace_id = tick.get("trace_id")
            if not symbol_value:
                self._logger.debug(
                    "RUNNER_TICK_CONSUMED no_symbol",
                    extra={
                        "event": "RUNNER_TICK_CONSUMED",
                        "symbol": None,
                        "trace_id": trace_id,
                        "active_symbol": False,
                        "state_active": False,
                        "phase": None,
                        "reason_if_skipped": "missing_symbol",
                    },
                )
                return
            symbol = enforce_canonical(normalize_symbol(str(symbol_value)))
            active = symbol in self._active_symbols
            if symbol not in self._tracked_symbols:
                if active:
                    self._tracked_symbols.add(symbol)
                else:
                    self._logger.debug(
                        "RUNNER_TICK_CONSUMED symbol=%s skipped=untracked",
                        symbol,
                        extra={
                            "event": "RUNNER_TICK_CONSUMED",
                            "symbol": symbol,
                            "trace_id": trace_id,
                            "active_symbol": False,
                            "state_active": False,
                            "phase": self._data_phase.get(symbol),
                            "reason_if_skipped": "symbol_not_in_active_universe",
                        },
                    )
                    return
            price = tick.get("last_price") or tick.get("ltp")
            if not isinstance(price, (int, float)):
                self._logger.debug(
                    "RUNNER_TICK_CONSUMED symbol=%s skipped=no_price",
                    symbol,
                    extra={
                        "event": "RUNNER_TICK_CONSUMED",
                        "symbol": symbol,
                        "trace_id": trace_id,
                        "active_symbol": active,
                        "state_active": symbol in self._symbol_state,
                        "phase": self._data_phase.get(symbol),
                        "reason_if_skipped": "missing_or_invalid_price",
                    },
                )
                return
            self._logger.debug(
                "RUNNER_TICK_CONSUMED symbol=%s trace_id=%s price=%s",
                symbol,
                trace_id,
                price,
                extra={
                    "event": "RUNNER_TICK_CONSUMED",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "price": float(price),
                    "active_symbol": active,
                    "state_active": symbol in self._symbol_state,
                    "phase": self._data_phase.get(symbol),
                    "reason_if_skipped": None,
                },
            )
            self._on_tick_safe(
                {**dict(tick), "symbol": symbol, "last_price": float(price)}
            )
        except Exception as e:
            self._logger.error("Failure in StrategyRunner._on_tick_from_bus: %s", e)

    async def on_data(self, message: "Message") -> None:
        payload = dict(message.data or {})
        payload.setdefault("trace_id", f"bus-{time_module.monotonic_ns()}")
        if not payload.get("symbol"):
            token = payload.get("token") or payload.get("instrument_token")
            if token is not None and self._market_data is not None:
                symbol_map = getattr(self._market_data, "_symbol_by_token", {}) or {}
                resolved = symbol_map.get(int(token)) if str(token).isdigit() else None
                if resolved:
                    payload["symbol"] = resolved
        self.on_tick_event(payload)

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol = str(tick.get("symbol") or "")
            if not symbol:
                self._logger.info(
                    "RUNNER_EVAL_DECISION",
                    extra={
                        "event": "RUNNER_EVAL_DECISION",
                        "symbol": "UNKNOWN",
                        "trace_id": tick.get("trace_id"),
                        "stage": "message_ingress",
                        "allowed": False,
                        "reason": "missing_symbol",
                    },
                )
                return
            price = tick.get("last_price") or tick.get("ltp")
            if not isinstance(price, (int, float)):
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage="message_ingress",
                    reason="missing_price",
                    allowed=False,
                    trace_id=str(tick.get("trace_id") or ""),
                )
                return
            self._on_tick_safe({**tick, "symbol": symbol, "last_price": float(price)})
        except Exception as e:
            self._logger.error("Failure in StrategyRunner.on_tick_event: %s", e)

    def on_datahub_tick(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        self.on_tick_event(tick)

    def _on_tick_safe(self, tick: Mapping[str, Any]) -> None:
        """Safe wrapper for _on_tick to handle exceptions."""
        symbol = tick.get("symbol")
        if not symbol:
            return

        try:
            normalized_symbol = enforce_canonical(normalize_symbol(str(symbol)))
            if normalized_symbol.count(":") != 1:
                raise RuntimeError(f"Malformed canonical symbol: {normalized_symbol}")
            # Reuse an upstream trace_id if the tick payload already carries one
            # (set by RunnerCallback or by an outer caller); otherwise mint here.
            trace_id = tick.get("trace_id") or f"{normalized_symbol}-{time_module.monotonic_ns()}"
            # RUNNER_TICK_CONSUMED: the tick has been pulled off the event bus
            # and accepted by the runner's evaluation entry point.  This is the
            # canonical observability breakpoint between publication and
            # evaluation, independent of which caller invoked _on_tick_safe.
            try:
                _price_consumed = (
                    tick.get("last_price") or tick.get("ltp") or tick.get("price")
                )
                _phase = self._data_phase.get(normalized_symbol)
                _state_active = False
                _state_obj = self._symbol_state.get(normalized_symbol)
                if _state_obj is not None:
                    _state_active = bool(getattr(_state_obj, "active", False))
                self._logger.debug(
                    "RUNNER_TICK_CONSUMED symbol=%s trace_id=%s price=%s phase=%s",
                    normalized_symbol,
                    trace_id,
                    _price_consumed,
                    _phase,
                    extra={
                        "event": "RUNNER_TICK_CONSUMED",
                        "symbol": normalized_symbol,
                        "trace_id": trace_id,
                        "price": _price_consumed,
                        "active_symbol": normalized_symbol in self._active_symbols,
                        "state_active": _state_active,
                        "phase": _phase,
                        "reason_if_skipped": None,
                    },
                )
            except Exception:  # pragma: no cover - defensive
                pass
            self._last_tick_time_by_symbol[normalized_symbol] = time.time()
            try:
                self._last_tick[normalized_symbol] = dict(tick)
            except Exception:
                self._last_tick[normalized_symbol] = {
                    "symbol": normalized_symbol,
                    "ltp": tick.get("ltp") or tick.get("last_price") or tick.get("price"),
                    "timestamp": tick.get("timestamp"),
                }
            if not hasattr(self, "_first_tick_ingested_symbols"):
                self._first_tick_ingested_symbols = set()
            if normalized_symbol not in self._first_tick_ingested_symbols:
                self._first_tick_ingested_symbols.add(normalized_symbol)
                history_count = len(self._indicator_engine.get_history(normalized_symbol) or [])
                self._logger.info(
                    "RUNNER_TICK_INGESTED symbol=%s history_count=%d",
                    normalized_symbol,
                    history_count,
                    extra={"event": "RUNNER_TICK_INGESTED", "symbol": normalized_symbol, "history_count": history_count},
                )
            engine = self._candle_engines.setdefault(normalized_symbol, CandleEngine())
            with self._symbol_locks[normalized_symbol]:
                engine.on_tick(tick)
            now_mono = time.monotonic()
            self._last_tick_seen_ts = now_mono

            # ── PIPELINE FEED + OBSERVABILITY ────────────────────────────────
            # Feed deterministic pipeline so candles_ready() stays current.
            # Also log TICK_RECEIVED (debug) and CANDLE_FORMED (info) events.
            try:
                _pl_candle = self._pipeline.on_tick(dict(tick))
                self._logger.debug(
                    "TICK_RECEIVED symbol=%s ltp=%s",
                    normalized_symbol,
                    tick.get("ltp") or tick.get("last_price", "?"),
                )
                if _pl_candle is not None:
                    _pl_count = len(self._pipeline.store.get(normalized_symbol))
                    self._logger.debug(
                        "CANDLE_FORMED symbol=%s ts=%s close=%.2f candles=%d ready=%s",
                        normalized_symbol,
                        _pl_candle.timestamp,
                        _pl_candle.close,
                        _pl_count,
                        _pl_count >= 50,
                    )
            except Exception as _pl_exc:
                self._logger.debug("Runner pipeline feed failed: %s", _pl_exc)
            # ✅ FIX: Throttle stall warning to 30s — same-bar-skip causes expected
            # gaps between _last_global_eval_ts updates (one eval per bar ≈ 60s cycle).
            # ✅ FIX D: Raise stall threshold to 120s. Options tick once per ~13min;
            # the 5s threshold fired constantly between normal tick batches.
            # A genuine stall = no strategy evaluation for > 2 full minutes with active ticks.
            if (
                self.ready
                and now_mono - self._last_global_eval_ts > 120.0
                and now_mono - self._last_stall_warn_ts > 120.0
            ):
                self._last_stall_warn_ts = now_mono
                if not is_market_open_now():
                    log_throttled(
                        self._logger,
                        "strategy_stall_check_skipped_market_closed",
                        "STALL_CHECK_SKIPPED reason=market_closed",
                        interval_sec=300.0,
                        level=logging.DEBUG,
                        extra={
                            "event": "STALL_CHECK_SKIPPED",
                            "reason": "market_closed",
                        },
                    )
                else:
                    self._logger.warning(
                        "Strategy evaluation stalled >120s (once per 120s)",
                        extra={
                            "event": "strategy_eval_stall",
                            "stall_sec": round(now_mono - self._last_global_eval_ts, 1),
                        },
                    )
            self._health_watchdog()
            self._logger.debug(
                "PIPELINE_OK",
                extra={"symbol": normalized_symbol, "state": str(self._runner_state)},
            )
            self._eval_counter += 1
            with self._eval_gate_lock:
                last_eval = self._last_eval_ts[normalized_symbol]
                if now_mono - last_eval < 0.05:
                    return
                self._last_eval_ts[normalized_symbol] = now_mono
            self._logger.debug(
                "STRATEGY_RECEIVED_TICK",
                extra={"event": "strategy_received_tick", "symbol": normalized_symbol},
            )
            if normalized_symbol not in self._first_tick_logged_symbols:
                self._first_tick_logged_symbols.add(normalized_symbol)
                self._logger.info(
                    "RUNNER_TICK_ACCEPTED symbol=%s trace_id=%s tick_price=%s first_tick=%s",
                    normalized_symbol,
                    trace_id,
                    tick.get("last_price") or tick.get("ltp"),
                    True,
                    extra={
                        "event": "RUNNER_TICK_ACCEPTED",
                        "symbol": normalized_symbol,
                        "trace_id": trace_id,
                        "tick_price": tick.get("last_price") or tick.get("ltp"),
                        "first_tick": True,
                    },
                )
            elif self._should_log_throttled(
                f"runner_tick_accepted_debug:{normalized_symbol}",
                self._tick_log_throttle_seconds,
            ):
                self._logger.debug(
                    "RUNNER_TICK_ACCEPTED symbol=%s trace_id=%s tick_price=%s first_tick=%s",
                    normalized_symbol,
                    trace_id,
                    tick.get("last_price") or tick.get("ltp"),
                    False,
                )
            
            # 🚨 FIX: Legacy ensure_valid_data() block completely removed.
            # We no longer pause live tick processing to attempt blocking historical 
            # backfills. Ticks will now flow directly into Phase 0/1/4 so the 
            # OneMinuteBarBuilder can construct live candles autonomously.
            
            with self._eval_gate_lock:
                if normalized_symbol in self._eval_in_progress_symbols:
                    return
                self._eval_in_progress_symbols.add(normalized_symbol)
            with self._eval_queue_lock:
                self._eval_queue_depth += 1
                self._eval_queue_peak = max(
                    self._eval_queue_peak,
                    self._eval_queue_depth,
                )
                eval_queue_depth = self._eval_queue_depth
                eval_queue_peak = self._eval_queue_peak
            now_queue_log = time.monotonic()
            if now_queue_log - self._last_eval_queue_log_ts > 10.0:
                self._logger.debug(
                    "EvalQueue depth=%d peak=%d",
                    eval_queue_depth,
                    eval_queue_peak,
                    extra={"event": "eval_queue_stats"},
                )
                self._last_eval_queue_log_ts = now_queue_log
            try:
                self._on_tick(normalized_symbol, {**dict(tick), "trace_id": trace_id})
            finally:
                with self._eval_queue_lock:
                    self._eval_queue_depth = max(0, self._eval_queue_depth - 1)
                with self._eval_gate_lock:
                    self._eval_in_progress_symbols.discard(normalized_symbol)
        except Exception as exc:
            LOGGER.error(
                "Critical error in _on_tick for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )
        finally:
            now = time.monotonic()
            self._emit_composite_reports()
            if now - self._last_summary_log >= 60.0:
                self._logger.info(
                    "ENGINE_SUMMARY",
                    extra={
                        "evals": self._eval_counter,
                        "signals": self._signal_counter,
                        "regime_blocks": self._regime_block_counter,
                        "capital_blocks": self._capital_block_counter,
                        "runner_state": str(self._runner_state),
                    },
                )
                self._eval_counter = 0
                self._signal_counter = 0
                self._regime_block_counter = 0
                self._capital_block_counter = 0
                self._last_summary_log = now

    def _health_watchdog(self) -> None:
        """Args: none; Returns: none; Raises: none."""
        now = time.monotonic()
        market_open = is_market_open_now()
        tick_flowing = (now - self._last_tick_seen_ts) <= 5.0
        eval_stalled = (now - self._last_global_eval_ts) > 5.0

        # ✅ FIX: Throttle — same-bar-skip keeps eval_stalled=True for the whole bar.
        # Only warn if stall > 90s (longer than one full bar cycle) to avoid spam.
        genuine_stall = (now - self._last_global_eval_ts) > 90.0
        if self.ready and tick_flowing and eval_stalled and genuine_stall:
            if not market_open:
                log_throttled(
                    self._logger,
                    "strategy_stall_check_skipped_market_closed",
                    "STALL_CHECK_SKIPPED reason=market_closed",
                    interval_sec=300.0,
                    level=logging.DEBUG,
                    extra={
                        "event": "STALL_CHECK_SKIPPED",
                        "reason": "market_closed",
                    },
                )
            else:
                log_throttled(
                    self._logger,
                    "health_watchdog_genuine_stall",
                    "Strategy eval genuinely stalled while ticks flowing (>90s)",
                    interval_sec=120.0,
                    level=logging.WARNING,
                    extra={
                        "event": "strategy_eval_stall",
                        "stall_sec": round(now - self._last_global_eval_ts, 1),
                    },
                )

        now_wall = time.time()
        stale_count = 0
        stale_symbols: list[str] = []

        for symbol, engine in self._candle_engines.items():
            if symbol not in self._active_symbols:
                last_tick_ts = float(self._last_tick_time_by_symbol.get(symbol, 0.0) or 0.0)
                has_recent_tick = last_tick_ts > 0 and (now_wall - last_tick_ts) <= 120.0

                if has_recent_tick and symbol in self._tracked_symbols:
                    self._active_symbols.add(symbol)
                    self._logger.info(
                        "SYMBOL_REACTIVATED_FROM_LIVE_TICK symbol=%s",
                        symbol,
                        extra={"event": "SYMBOL_REACTIVATED_FROM_LIVE_TICK", "symbol": symbol},
                    )
                else:
                    if self._should_log_throttled(f"backfill_skipped_removed:{symbol}", 300.0):
                        self._logger.debug(
                            "BACKFILL_SKIPPED_REMOVED_SYMBOL symbol=%s",
                            symbol,
                            extra={"event": "BACKFILL_SKIPPED_REMOVED_SYMBOL", "symbol": symbol},
                        )
                    continue
            # 1. Use .get() to prevent KeyError on newly subscribed symbols
            stale_for = now_wall - self._last_tick_time_by_symbol.get(symbol, now_wall)

            # 2. Use the centralised, market-session-aware threshold so that
            # off-market option/index tick gaps are not treated as faults.
            symbol_stale_threshold = stale_threshold_for_symbol(symbol, market_open)
            if stale_for > symbol_stale_threshold:
                stale_count += 1
                stale_symbols.append(symbol)

                # Off-market: never trigger a per-symbol WS-stale backfill or
                # zombie restart. Just emit one DEBUG/throttled trace.
                if not market_open:
                    continue

                last_log_ts = self._last_ws_stale_log_ts_by_symbol.get(symbol, 0.0)
                # 3. GATE THE ENTIRE BACKFILL PROCESS, not just the logger
                if now_wall - last_log_ts >= 60.0:
                    self._logger.warning(
                        "⚠️ %s: WS stale (%.1fs) → triggering backfill",
                        symbol,
                        stale_for,
                    )
                    self._last_ws_stale_log_ts_by_symbol[symbol] = now_wall

                    try:
                        repair_input = pd.DataFrame(
                            self._hydrate_missing_bars(
                                symbol, max(self._required_candles, 50)
                            )
                        )
                        if repair_input.empty:
                            continue

                        with self._symbol_locks[symbol]:
                            repaired = repair_with_backfill(
                                symbol,
                                sanitize(engine.get_df()),
                                fetch_recent_rest=lambda _sym: repair_input,
                                max_bars=engine.max_bars,
                            )
                            if not repaired.empty:
                                engine.df = repaired
                    except Exception:
                        # 4. Use .exception to capture traceback and stop silent failures
                        self._logger.exception(
                            "CRITICAL: Failure in StrategyRunner._health_watchdog backfill for %s",
                            symbol,
                        )

        # 5. Move WS Reconnect OUTSIDE the symbol loop.
        # We only want to evaluate a WS restart once per cycle, not once per symbol.
        if stale_count > 0:
            if not market_open:
                log_throttled(
                    self._logger,
                    "zombie_ws_restart_skipped_market_closed",
                    "WS_RESTART_SKIPPED reason=market_closed stale_symbols=%d"
                    % stale_count,
                    interval_sec=300.0,
                    level=logging.DEBUG,
                    extra={
                        "event": "WS_RESTART_SKIPPED",
                        "reason": "market_closed",
                        "stale_symbols": stale_symbols[:32],
                        "stale_count": stale_count,
                    },
                )
                return
            last_reconnect_ts = getattr(self, "_last_ws_reconnect_attempt_ts", 0.0)
            # Increase WS restart throttle to 30s to prevent bouncing the connection
            if now_wall - last_reconnect_ts >= 30.0:
                self._last_ws_reconnect_attempt_ts = now_wall
                try:
                    reconnect = getattr(
                        self._market_data, "_trigger_zombie_ws_restart", None
                    )
                    if callable(reconnect):
                        self._logger.warning(
                            "🔄 Watchdog triggering zombie WS restart (%d symbols stale)",
                            stale_count,
                        )
                        reconnect()
                except Exception:
                    self._logger.exception(
                        "CRITICAL: Failure in StrategyRunner._health_watchdog.reconnect"
                    )

    # ✅ FIX: New Method to Prime Indicators
    async def _backfill_history(self) -> None:
        """
        Download historical data to warm up indicators.
        Skips if startup hydration was already performed by App.py.
        """
        total_bars = 0

        try:
            # FIX 1: Verify actual indicator arrays are warmed up
            is_fully_warmed_up = True
            targets = tuple(self._active_symbols)

            for symbol in targets:
                bars = self._indicator_engine.get_history(symbol)
                if not bars or len(bars) < self._required_candles:
                    is_fully_warmed_up = False
                    break

            if is_fully_warmed_up:
                self._logger.info(
                    "⏭️ Skipping historical backfill (indicators fully warmed up)"
                )
                return

            # 2. FALLBACK: Only runs if App.py failed
            self._logger.warning(
                "⚠️ StrategyRunner memory is empty! Triggering fallback backfill..."
            )

            with self._lock:
                targets = list(self._active_symbols)

            if not targets:
                self._logger.warning("⚠️ Backfill skipped: No active symbols found.")
                return

            for symbol in targets:
                try:
                    target = self._required_bars_for_symbol(symbol)
                    rows = self._get_mdm_bars(symbol, target)
                    if rows:
                        for bar_data in rows:
                            self.ingest_historical_bar(bar_data)
                            total_bars += 1
                        if len(rows) >= target:
                            self._set_symbol_hydration_state(symbol, SymbolState.READY)
                        else:
                            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
                            self._request_mdm_hydration(symbol, target)
                    else:
                        self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
                        self._request_mdm_hydration(symbol, target)

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
        normalized = self._get_mdm_bars(symbol, min_bars)
        for row in normalized:
            payload = dict(row)
            payload["symbol"] = symbol
            self.ingest_historical_bar(payload)
        if len(normalized) < min_bars:
            self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)
            self._request_mdm_hydration(symbol, min_bars)
            return []
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

    def _compute_regime_snapshot(self, symbol: str) -> MarketRegime:
        """Compute market regime for symbol. Args: symbol; Returns: MarketRegime; Raises: none."""
        try:
            indicators = self._indicator_engine.get_indicators(symbol)
            history = self._indicator_engine.get_history(symbol)
            atr_avg = 0.0
            if history:
                tail = history[-20:]
                mean_price = sum(float(v) for v in tail) / max(len(tail), 1)
                atr_avg = mean_price * 0.002
            latest = self._indicator_engine.get_latest(symbol)
            volume = float((latest or {}).get("volume") or 0.0)
            avg_volume = float((latest or {}).get("avg_volume") or 0.0)
            volume_expansion = (volume / avg_volume) if avg_volume > 0 else 1.0
            current_vwap = float(indicators.get("vwap") or 0.0)
            history_tail = history[-3:] if history else []
            reference = sum(float(v) for v in history_tail) / max(len(history_tail), 1)
            vwap_slope = (current_vwap - reference) if reference > 0 else 0.0
            snapshot = self._market_regime_engine.classify(
                {
                    "adx": indicators.get("adx"),
                    "atr": indicators.get("atr"),
                    "atr_average": atr_avg,
                    "vwap_slope": vwap_slope,
                    "volume_expansion": volume_expansion,
                }
            )
            self._last_regime_inputs_by_symbol[symbol] = {
                "adx": indicators.get("adx"),
                "atr": indicators.get("atr"),
                "atr_average": atr_avg,
                "vwap_slope": vwap_slope,
                "volume": volume,
                "avg_volume": avg_volume,
                "volume_expansion": volume_expansion,
                "vwap": current_vwap,
            }
            self._last_regime_by_symbol[symbol] = snapshot.regime
            return snapshot.regime
        except Exception as exc:
            self._logger.error(
                "Failure in StrategyRunner._compute_regime_snapshot: %s", exc
            )
            return self._last_regime_by_symbol.get(symbol, MarketRegime.LOW_ACTIVITY)

    def detect_market_regime(self, symbol: str) -> str:
        """Args: symbol. Returns: coarse regime label. Raises: None."""
        try:
            atr_raw = (
                self._indicator_engine.get_atr(symbol)
                if self._indicator_engine
                else None
            )
            atr = float(atr_raw) if atr_raw is not None else None
            if atr is None:
                return "unknown"
            if atr < 10.0:
                return "low_volatility"
            if atr > 40.0:
                return "high_volatility"
            return "normal"
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner.detect_market_regime: %s", exc
            )
            return "unknown"

    def _strategy_allowed_for_regime(self, strategy: str, regime: MarketRegime) -> bool:
        """Validate regime gate for strategy. Args: strategy, regime; Returns: bool; Raises: none."""
        def _canonical_regime_name(value: str) -> str:
            """Normalize regime aliases. Args: value. Returns: canonical name. Raises: none."""
            normalized_value = str(value or "").strip().upper()
            aliases = {
                "VOLATILE": "HIGH_VOLATILITY",
                "HIGHVOL": "HIGH_VOLATILITY",
                "HIGH_VOL": "HIGH_VOLATILITY",
                "TRENDING": "TREND",
                "RANGING": "RANGE",
            }
            return aliases.get(normalized_value, normalized_value)

        if not _env_bool("RUNNER_ENABLE_REGIME_GATE", True):
            self._logger.debug(
                "REGIME_GATE_BYPASSED strategy=%s regime=%s reason=disabled",
                strategy or "unknown",
                regime.value,
                extra={"event": "REGIME_GATE_BYPASSED"},
            )
            return True
        normalized = (strategy or "").strip().lower()
        strategy_env_map = {
            "vwap_pro": "RUNNER_VWAP_ALLOWED_REGIMES",
            "vwappro": "RUNNER_VWAP_ALLOWED_REGIMES",
            "premium_momentum": "RUNNER_PREMIUM_SQUEEZE_ALLOWED_REGIMES",
            "premium_momentum_squeeze": "RUNNER_PREMIUM_SQUEEZE_ALLOWED_REGIMES",
            "orb_pro": "RUNNER_ORB_ALLOWED_REGIMES",
            "orbpro": "RUNNER_ORB_ALLOWED_REGIMES",
        }
        env_name = strategy_env_map.get(normalized)
        default_allowed = "TREND,NORMAL,HIGH_VOLATILITY"
        if env_name == "RUNNER_VWAP_ALLOWED_REGIMES":
            default_allowed = "TREND,NORMAL"
        allowed_csv = os.getenv(env_name or "", default_allowed) if env_name else default_allowed
        allowed = {
            _canonical_regime_name(item)
            for item in allowed_csv.split(",")
            if item.strip()
        }
        regime_name = _canonical_regime_name(regime.value)
        allowed_for_regime = regime_name in allowed
        self._logger.debug(
            "REGIME_GATE_DECISION strategy=%s regime=%s allowed=%s allowed_regimes=%s env=%s",
            strategy or "unknown",
            regime.value,
            allowed_for_regime,
            sorted(allowed),
            env_name or "default",
            extra={"event": "REGIME_GATE_DECISION"},
        )
        return allowed_for_regime
    def _strategy_regime_decision(
        self, *, strategy: str, regime: MarketRegime, symbol: str, metadata: Mapping[str, Any] | None = None
    ) -> tuple[bool, str]:
        """Return strategy-regime compatibility decision. Args: strategy/regime/symbol/metadata; Returns: tuple[bool,str]; Raises: none."""
        del symbol
        normalized = (strategy or "").strip().lower()
        regime_name = str(regime.value or "").upper()
        canonical = {"VOLATILE": "HIGH_VOLATILITY", "HIGHVOL": "HIGH_VOLATILITY", "HIGH_VOL": "HIGH_VOLATILITY", "TRENDING": "TREND", "RANGING": "RANGE"}.get(regime_name, regime_name)
        meta = dict(metadata or {})
        selected = bool(
            meta.get("candidate_selected")
            or meta.get("is_selected_option")
            or meta.get("selected_ok")
        )
        try:
            spread_pct = float(
                meta.get("candidate_spread_pct")
                if meta.get("candidate_spread_pct") is not None
                else meta.get("spread_pct")
                if meta.get("spread_pct") is not None
                else 999.0
            )
        except (TypeError, ValueError):
            spread_pct = 999.0
        try:
            rr = float(
                meta.get("candidate_rr")
                if meta.get("candidate_rr") is not None
                else meta.get("rr")
                if meta.get("rr") is not None
                else 0.0
            )
        except (TypeError, ValueError):
            rr = 0.0
        if self._strategy_allowed_for_regime(strategy, regime):
            return True, "regime_in_allowed_list"
        if normalized in {"vwap_pro", "vwappro"} and canonical == "HIGH_VOLATILITY":
            max_spread = float(os.getenv("VWAP_HIGH_VOL_MAX_SPREAD_PCT", "0.75") or "0.75")
            min_rr = float(os.getenv("VWAP_HIGH_VOL_MIN_RR", "1.6") or "1.6")
            if selected and spread_pct <= max_spread and rr >= min_rr:
                return True, "vwap_high_vol_execution_quality_soft_allow"
            return False, "vwap_high_vol_execution_quality_failed"
        return False, "regime_not_allowed"

    def _strategy_slots_available(self) -> bool:
        """Return True when active strategy slots are available for new entries."""
        try:
            active_positions = len(self._position_manager.get_open_positions())
        except Exception:
            active_positions = 0
        with self._orders_lock:
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

    def _emit_runner_eval_decision(
        self,
        *,
        symbol: str,
        stage: str,
        reason: str,
        allowed: bool,
        trace_id: str | None = None,
        **context: Any,
    ) -> None:
        """Emit a single structured RUNNER_EVAL_DECISION log.

        Captures the full per-symbol evaluation-path decision so every early
        return is traceable. ``stage`` is one of ``phase7``, ``phase9``,
        ``signal_forward``; ``reason`` names the specific gate.
        """
        try:
            state_obj = self._symbol_state.get(symbol)
            state_active = bool(getattr(state_obj, "active", False)) if state_obj else False
            sym_state = self._symbol_states.get(symbol)
            sym_state_val = getattr(sym_state, "value", str(sym_state)) if sym_state else None
            try:
                candle_count = len(self._indicator_engine.get_history(symbol) or [])
            except Exception:  # pragma: no cover - defensive
                candle_count = None
            current_version = int(self._candle_versions.get(symbol, 0))
            last_version = int(self._last_strategy_versions.get(symbol, 0))
            last_bar_ts = self._last_bar_ts.get(symbol)
            last_bar_ts_iso = last_bar_ts.isoformat() if last_bar_ts else None
            last_eval_bar_ts = None
            if state_obj is not None:
                _leb = getattr(state_obj, "_last_eval_bar_ts", None)
                if _leb is not None:
                    try:
                        last_eval_bar_ts = _leb.isoformat()
                    except Exception:
                        last_eval_bar_ts = str(_leb)
            mdm_last_tick_age = None
            tick_age_ms = None
            try:
                mdm_last_map = getattr(self._market_data, "_last_tick_time", {}) or {}
                last_tick_wall = mdm_last_map.get(symbol)
                if isinstance(last_tick_wall, (int, float)) and last_tick_wall > 0:
                    mdm_last_tick_age = round(time.time() - float(last_tick_wall), 3)
                    tick_age_ms = int(max(0.0, mdm_last_tick_age * 1000.0))
            except Exception:  # pragma: no cover - defensive
                pass
            has_live_bars = symbol in getattr(self, "_live_bar_seen", set())
            payload = {
                "event": "RUNNER_EVAL_DECISION",
                "symbol": symbol,
                "trace_id": trace_id,
                "allowed": allowed,
                "stage": stage,
                "reason": reason,
                "active_symbol": symbol in self._active_symbols,
                "symbol_state": sym_state_val,
                "state_active": state_active,
                "candle_count": candle_count,
                "current_version": current_version,
                "last_version": last_version,
                "last_bar_ts": last_bar_ts_iso,
                "last_eval_bar_ts": last_eval_bar_ts,
                "mdm_last_tick_age_s": mdm_last_tick_age,
                "tick_age_ms": tick_age_ms,
                "has_live_bars": has_live_bars,
                "data_phase": self._data_phase.get(symbol),
            }
            if context:
                payload.update(context)
            self._logger.info(
                "RUNNER_EVAL_DECISION symbol=%s allowed=%s stage=%s reason=%s candle_count=%s current_version=%s last_version=%s tick_age_ms=%s data_phase=%s",
                symbol,
                allowed,
                stage,
                reason,
                candle_count,
                current_version,
                last_version,
                tick_age_ms,
                payload.get("data_phase"),
                extra=payload,
            )
        except Exception as exc:  # noqa: BLE001
            # Observability must never raise; log and continue.
            try:
                self._logger.error(
                    "runner_eval_decision_emit_failed: %s", exc,
                    extra={"event": "runner_eval_decision_emit_error", "symbol": symbol},
                )
            except Exception:  # pragma: no cover - defensive
                pass

    def _is_tradable_symbol(self, symbol: str) -> bool:
        """Return True when symbol is a tradable NIFTY option. Args: symbol. Returns: bool. Raises: none."""
        value = str(symbol or "").upper()
        return value.startswith("NFO:NIFTY") and (value.endswith("CE") or value.endswith("PE"))

    def _is_tradable_option_symbol(self, symbol: str) -> bool:
        """Compatibility alias for option-tradability checks."""
        return self._is_tradable_symbol(symbol)

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Return freshest normalized quote available from DataHub/MDM."""
        normalized = normalize_symbol(symbol)
        for source in (self._data_hub, self._market_data):
            if source is None:
                continue
            for name in ("get_quote", "get_symbol_snapshot", "get_latest_tick"):
                fn = getattr(source, name, None)
                if not callable(fn):
                    continue
                try:
                    raw = fn(normalized)
                except Exception:
                    continue
                if raw is None:
                    continue
                if isinstance(raw, Mapping):
                    payload = dict(raw)
                else:
                    payload = {
                        "ltp": getattr(raw, "ltp", None) or getattr(raw, "last_price", None) or getattr(raw, "price", None),
                        "bid": getattr(raw, "bid", None) or getattr(raw, "best_bid", None),
                        "ask": getattr(raw, "ask", None) or getattr(raw, "best_ask", None),
                        "tick_age_s": getattr(raw, "tick_age_s", None),
                        "ts_ns": getattr(raw, "ts_ns", None),
                    }
                ltp = _extract_float(payload, "ltp", "last_price", "price", "close")
                if ltp is not None and ltp > 0:
                    return payload
        return None

    def _is_option_symbol_tick_fresh(self, symbol: str, *, max_age_s: float | None = None) -> bool:
        """Freshness guard for selected option soft-pass and live execution readiness."""
        if not self._is_tradable_symbol(symbol):
            return False
        limit = float(max_age_s or os.getenv("OPTION_TICK_FRESH_MAX_AGE_S", "60") or 60.0)
        quote = self.get_quote(symbol)
        if isinstance(quote, Mapping):
            age = _extract_float(quote, "tick_age_s", "age_s")
            if age is not None:
                return age <= limit
        for source in (self._market_data, self._data_hub):
            if source is None:
                continue
            fn = getattr(source, "time_since_last_tick", None)
            if callable(fn):
                try:
                    age = fn(symbol)
                except Exception:
                    continue
                if age is not None:
                    return float(age) <= limit
        return False

    def _is_context_symbol(self, symbol: str) -> bool:
        """Return True when symbol is a context-only spot/futures instrument. Args: symbol. Returns: bool. Raises: none."""
        value = str(symbol or "").upper()
        return value == "NSE:NIFTY" or (value.startswith("NFO:NIFTY") and value.endswith("FUT"))

    def _required_bars_for_symbol(self, symbol: str) -> int:
        """Return readiness bars by role. Args: symbol. Returns: int. Raises: none."""
        return self._context_required_bars if self._is_context_symbol(symbol) else self._option_required_bars

    def _sync_indicator_history_if_needed(self, symbol: str) -> None:
        """Ensure indicator engine has runner bars for symbol. Args: symbol. Returns: none. Raises: none."""
        runner_hist = self._symbol_history.get(symbol) or []
        indicator_count = len(self._indicator_engine.get_history(symbol) or [])
        if runner_hist and indicator_count == 0:
            limit = max(self._context_required_bars, self._option_required_bars)
            for bar in runner_hist[-limit:]:
                if hasattr(self._indicator_engine, "update_bar"):
                    self._indicator_engine.update_bar(symbol, bar)
                else:
                    self._indicator_engine.update_price(
                        symbol,
                        bar.as_mapping(),
                        volume=bar.volume,
                        timestamp=bar.timestamp,
                        is_complete=True,
                    )
            self._logger.warning(
                "INDICATOR_HISTORY_AUTO_SYNC symbol=%s synced=%d",
                symbol,
                len(runner_hist[-limit:]),
            )

    def _strategy_evaluation_allowed(
        self, symbol: str, trace_id: str | None = None
    ) -> bool:
        """Args: symbol + trace_id. Returns: bool gate verdict. Raises: none."""
        try:
            if symbol not in self._active_symbols:
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage='phase9',
                    reason='symbol_not_active',
                    allowed=False,
                    trace_id=trace_id,
                )
                return False
            required_bars = self._required_bars_for_symbol(symbol)
            option_execution_min_bars = int(os.getenv("OPTION_EXECUTION_MIN_BARS", "5") or "5")
            required_bars = max(required_bars, option_execution_min_bars)
            history_count = len(self._indicator_engine.get_history(symbol) or [])
            restored_from_cache = symbol in self._restored_from_cache_symbols
            if restored_from_cache and history_count >= required_bars:
                return True
            if not self._indicator_engine.has_min_bars(symbol, required_bars):
                if history_count < required_bars:
                    if self._is_context_symbol(symbol):
                        if symbol == "NSE:NIFTY" and self._should_log_throttled(f"spot_cold:{symbol}", 120.0):
                            self._logger.warning("CONTEXT_SPOT_HISTORY_COLD symbol=NSE:NIFTY bars=%d required=%d", history_count, required_bars)
                        elif self._should_log_throttled(f"fut_cold:{symbol}", 120.0):
                            self._logger.warning("CONTEXT_FUTURES_HISTORY_COLD symbol=%s bars=%d required=%d", symbol, history_count, required_bars)
                    elif self._is_tradable_symbol(symbol):
                        if self._should_log_throttled(f"opt_cold:{symbol}", 120.0):
                            self._logger.warning("OPTION_HISTORY_COLD symbol=%s bars=%d required=%d", symbol, history_count, required_bars)
                        spot_bars = len(self._indicator_engine.get_history("NSE:NIFTY") or [])
                        fut_symbol = next((sym for sym in self._active_symbols if self._is_context_symbol(sym) and sym != "NSE:NIFTY"), "")
                        fut_bars = len(self._indicator_engine.get_history(fut_symbol) or []) if fut_symbol else 0
                        if (spot_bars < self._context_required_bars or fut_bars < self._context_required_bars) and self._should_log_throttled(f"opt_ctx_cold:{symbol}", 120.0):
                            self._logger.warning("OPTION_EVAL_BLOCKED_CONTEXT_COLD option=%s spot_bars=%d futures_bars=%d required=%d", symbol, spot_bars, fut_bars, self._context_required_bars)
                    strict_for_all = _env_bool("OPTION_STRICT_HISTORY_FOR_ALL_STRATEGIES", False)
                    tick_is_fresh = self._is_option_symbol_tick_fresh(symbol)
                    selected_symbols = {
                        normalize_symbol(sym)
                        for sym in (self._active_selected_ce, self._active_selected_pe)
                        if sym
                    }
                    is_selected_option = normalize_symbol(symbol) in selected_symbols
                    option_eval_min_live_bars = int(os.getenv("OPTION_EVAL_MIN_LIVE_BARS", "1") or "1")
                    spot_bars = len(self._indicator_engine.get_history("NSE:NIFTY") or [])
                    fut_symbol = next((sym for sym in self._active_symbols if self._is_context_symbol(sym) and sym != "NSE:NIFTY"), "")
                    fut_bars = len(self._indicator_engine.get_history(fut_symbol) or []) if fut_symbol else 0
                    context_ready = spot_bars >= self._context_required_bars and fut_bars >= self._context_required_bars
                    soft_pass = (
                        self._is_tradable_symbol(symbol)
                        and not strict_for_all
                        and is_selected_option
                        and history_count >= option_eval_min_live_bars
                        and tick_is_fresh
                        and context_ready
                    )
                    if self._should_log_throttled(
                        f"eval_block_cold_history:{symbol}", 120.0
                    ):
                        self._logger.warning(
                            "EVALUATION_BLOCKED_COLD_HISTORY symbol=%s bars=%d required=%d",
                            symbol,
                            history_count,
                            required_bars,
                            extra={
                                "event": "EVALUATION_BLOCKED_COLD_HISTORY",
                                "symbol": symbol,
                                "bars": history_count,
                                "required": required_bars,
                            },
                        )
                reason = 'insufficient_indicator_bar_count' if history_count < required_bars else 'indicator_history_integrity_failed'
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage='phase9',
                    reason=reason,
                    allowed=False,
                    trace_id=trace_id,
                )
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                'Failure in StrategyRunner._strategy_evaluation_allowed: %s',
                exc,
                exc_info=exc,
            )
            self._emit_runner_eval_decision(
                symbol=symbol,
                stage='phase9',
                reason='strategy_eval_gate_exception',
                allowed=False,
                trace_id=trace_id,
            )
            return False

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
                quarantined = symbol in self._quarantined_symbols

            if quarantined:
                self._logger.warning(
                    "Skipping quarantined symbol",
                    extra={
                        "event": "symbol_quarantined_skip",
                        "symbol": symbol,
                    },
                )
                return False

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

    def validate_market_depth(self) -> bool:
        """
        Validates that the WebSocket is actively streaming data for our requested universe.
        """
        mdm = self._market_data
        if not mdm:
            log_throttled(
                self._logger,
                "market_depth_mdm_missing",
                "market_depth_mdm_missing",
                interval_sec=self._cooldown_log_throttle_seconds,
                level=logging.WARNING,
                extra={"event": "market_depth_mdm_missing"},
            )
            return False

        token_map = getattr(mdm, "_symbol_by_token", {})
        current_token_count = len(token_map)
        tracked_symbols = getattr(self, "_active_symbols", {})
        expected_count = len(tracked_symbols)
        if expected_count == 0:
            log_throttled(
                self._logger,
                "market_depth_no_active_symbols",
                "market_depth_no_active_symbols",
                interval_sec=self._cooldown_log_throttle_seconds,
                level=logging.WARNING,
                extra={"event": "market_depth_no_active_symbols"},
            )
            return False
        minimum_required = max(2, int(expected_count * 0.8))
        if current_token_count < minimum_required:
            log_throttled(
                self._logger,
                "market_depth_insufficient",
                "market_depth_insufficient",
                interval_sec=self._cooldown_log_throttle_seconds,
                level=logging.WARNING,
                extra={
                    "event": "market_depth_insufficient",
                    "active_tokens": current_token_count,
                    "required_tokens": minimum_required,
                    "expected_symbols": expected_count,
                },
            )
            return False

        return True

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Handle incoming tick. Args: symbol, tick. Returns: None. Raises: Exception."""
        self._logger.debug(
            "Entered StrategyRunner._on_tick",
            extra={"event": "tick_enter", "symbol": symbol},
        )
        phase = "entry"
        try:
            is_live_mode = (
                str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"
                or str(os.getenv("ENABLE_LIVE", "false")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
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
            # ✅ FIX (9 Mar 2026): Use get_position() not get_active_contract() — ActiveContract
            #    has NO 'strategy' field so strat always resolved to "unknown", triggering
            #    adoption on EVERY tick causing the orphan-adoption storm in logs.
            if self._position_manager:
                open_pos = self._position_manager.get_position(symbol)
                if open_pos is not None:
                    strat = getattr(open_pos, "strategy", "") or "unknown"
                    if "manual" in strat.lower() or "unknown" in strat.lower():
                        log_throttled(
                            self._logger,
                            f"orphan_guard_{symbol}",
                            f"🛡️ ORPHAN GUARD: {symbol} is unmanaged. Adopting (tick continues)...",
                            interval_sec=30.0,
                            level=logging.DEBUG,
                        )
                        if hasattr(self, "_adopt_orphan_positions"):
                            self._adopt_orphan_positions()
                        # ✅ DO NOT return — tick must continue flowing for bracket SL/TP monitoring

            # =================================================================
            # PHASE 1: EXTRACT DATA FIRST (Must happen before any logging)
            # =================================================================

            now = datetime.now(timezone.utc)
            trace_id = str(tick.get("trace_id") or f"{symbol}-{time_module.monotonic_ns()}")

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
            has_explicit_delta = "volume_delta" in tick
            has_explicit_volume = "volume" in tick
            raw_volume_delta = _extract_int(tick, "volume_delta") if has_explicit_delta else (
                _extract_int(tick, "volume") if has_explicit_volume else 0
            )
            raw_volume_cumulative = _extract_int(
                tick, "volume_cumulative", "volume_traded", "volume_traded_today"
            )
            source = tick.get("source", "unknown")
            if source == "poll":
                self._mark_live(symbol)
            is_seed = bool(tick.get("seed"))
            normalized_symbol = symbol
            history_ready = bool(self._history_ready_by_symbol.get(symbol, False))
            spot_age: float | None = None
            current_regime = getattr(self, "_current_regime", None)
            capital_ok = bool(self._risk_manager.available_balance > 0.0)
            self._logger.debug(
                "EVAL_GATE_STATUS",
                extra={
                    "symbol": normalized_symbol,
                    "history_ready": history_ready,
                    "spot_age": spot_age,
                    "regime": str(current_regime),
                    "capital_ok": capital_ok,
                },
            )

            if is_seed and price <= 0:
                log_throttled(
                    self._logger,
                    f"seed_tick_price_zero_{symbol}",
                    f"Condition met: seed_tick_price_missing for {symbol}",
                    interval_sec=120.0,
                    level=logging.INFO,
                )
                return

            volume = 0
            if has_explicit_delta:
                volume = max(int(raw_volume_delta), 0)
            elif has_explicit_volume:
                if raw_volume_cumulative > 0 and raw_volume_delta == raw_volume_cumulative:
                    volume = 0
                    log_throttled(
                        self._logger,
                        f"runner_rejected_cumulative_volume:{symbol}",
                        "RUNNER_REJECTED_CUMULATIVE_VOLUME symbol=%s volume=%s cumulative=%s",
                        symbol,
                        raw_volume_delta,
                        raw_volume_cumulative,
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "RUNNER_REJECTED_CUMULATIVE_VOLUME",
                            "symbol": symbol,
                            "volume": raw_volume_delta,
                            "cumulative": raw_volume_cumulative,
                        },
                    )
                else:
                    volume = max(int(raw_volume_delta), 0)
            else:
                first_tick_seen = symbol not in self._last_cumulative_volume
                if raw_volume_cumulative > 0:
                    last_cum = self._last_cumulative_volume.get(symbol, -1)
                    if last_cum < 0:
                        volume = 0
                    elif raw_volume_cumulative >= last_cum:
                        volume = raw_volume_cumulative - last_cum
                    else:
                        volume = 0
                    self._last_cumulative_volume[symbol] = raw_volume_cumulative
                elif first_tick_seen:
                    volume = 0
                    self._last_cumulative_volume[symbol] = raw_volume_cumulative
                    log_throttled(
                        self._logger,
                        f"tick_volume_seeded_{symbol}",
                        "Condition met: tick_volume_baseline_only_first_tick",
                        interval_sec=60.0,
                        level=logging.INFO,
                    )
            if self._is_tradable_symbol(symbol):
                max_runner_delta = int(os.getenv("OPTION_MAX_REASONABLE_TICK_VOLUME_DELTA", "1000000") or "1000000")
                if volume > max_runner_delta:
                    log_throttled(
                        self._logger,
                        f"runner_option_volume_clamped:{symbol}",
                        "RUNNER_OPTION_VOLUME_CLAMPED symbol=%s volume=%s max=%s",
                        symbol,
                        volume,
                        max_runner_delta,
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "RUNNER_OPTION_VOLUME_CLAMPED",
                            "symbol": symbol,
                            "volume": volume,
                            "max": max_runner_delta,
                        },
                    )
                    volume = 0

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
                    except Exception as e:
                        LOGGER.exception(
                            "[CRITICAL] unhandled exception", exc_info=True
                        )
                        raise
                return

            skip_strategy = False
            if not self._is_market_open(now):
                log_throttled(
                    self._logger,
                    "market_closed_global",
                    "Condition met: market_closed_diagnostic_only",
                    interval_sec=300.0,
                    level=logging.INFO,
                    extra={
                        "event": "RUNNER_MARKET_HOURS_DIAGNOSTIC",
                        "symbol": symbol,
                        "trace_id": trace_id,
                        "reason": "market_closed_no_runner_block",
                    },
                )

            # 🚨 RELAXED LATENCY GUARD: Polling APIs have natural latency.
            # We allow ticks up to 5 seconds old to ensure strategies evaluate.
            tick_latency_ms = tick_age * 1000.0
            if tick_latency_ms > 5000.0:
                skip_strategy = True

            # Stale tick threshold is centrally derived per-symbol and
            # market-session aware (utils.market_hours.stale_threshold_for_symbol).
            market_open_now = is_market_open_now()
            stale_threshold = stale_threshold_for_symbol(symbol, market_open_now)

            if tick_age > stale_threshold:
                if not market_open_now:
                    log_throttled(
                        self._logger,
                        f"stale_tick_offmarket:{symbol}",
                        f"OFFMARKET_STALE_TICK symbol={symbol} age_s={tick_age:.1f} "
                        f"threshold={stale_threshold:.1f}",
                        interval_sec=900.0,
                        level=logging.DEBUG,
                        extra={
                            "event": "OFFMARKET_STALE_TICK",
                            "symbol": symbol,
                            "age_s": tick_age,
                            "threshold_s": stale_threshold,
                        },
                    )
                else:
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

            phase = "phase3_diagnostics"
            # Log first tick at INFO, routine ticks at DEBUG.
            if symbol not in self._first_tick_logged_symbols:
                self._first_tick_logged_symbols.add(symbol)
                self._logger.info(
                    "RUNNER_TICK_ACCEPTED symbol=%s trace_id=%s tick_price=%.2f tick_age_s=%.3f volume=%s",
                    symbol,
                    trace_id,
                    price,
                    tick_age,
                    volume,
                    extra={
                        "event": "RUNNER_TICK_ACCEPTED",
                        "symbol": symbol,
                        "trace_id": trace_id,
                        "tick_price": price,
                        "tick_age_s": tick_age,
                        "volume": volume,
                        "first_tick": True,
                    },
                )
            elif self._should_log_throttled(
                f"runner_tick_accepted:{symbol}",
                self._tick_log_throttle_seconds,
            ):
                self._logger.debug(
                    "RUNNER_TICK_ACCEPTED symbol=%s trace_id=%s tick_price=%.2f tick_age_s=%.3f volume=%s",
                    symbol,
                    trace_id,
                    price,
                    tick_age,
                    volume,
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
                    f"RUNNER_BOOT_GRACE: {15 - time_since_startup:.0f}s remaining before live evaluation allowed",
                    interval_sec=5.0,
                    level=logging.INFO,
                )
            elif is_live_mode and not bool(self._runtime_live_orders_armed):
                log_throttled(
                    self._logger,
                    "runner_post_grace_blocked",
                    "RUNNER_POST_GRACE_STILL_BLOCKED",
                    interval_sec=30.0,
                    level=logging.INFO,
                    extra={
                        "event": "RUNNER_POST_GRACE_STILL_BLOCKED",
                        "data_hard_ready": bool(self._runtime_data_hard_ready),
                        "evaluation_ready": bool(self._runtime_evaluation_ready),
                        "live_orders_armed": bool(self._runtime_live_orders_armed),
                        "reason": self._runtime_readiness_reason,
                    },
                )

            phase = "phase4_bar_build"
            # =================================================================
            # PHASE 4: BAR BUILDING (Always process, even during warmup)
            # =================================================================

            builder = self._bar_builders.setdefault(symbol, OneMinuteBarBuilder())
            try:
                completed_bar = builder.update(float(price), volume, timestamp)
                if completed_bar is not None:
                    self._ingest_bar(symbol, completed_bar)
                    runner_history_count = len(
                        self._indicator_engine.get_history(symbol) or []
                    )
                    should_info_live_bar = symbol not in self._first_live_bar_logged_symbols
                    if should_info_live_bar:
                        self._first_live_bar_logged_symbols.add(symbol)
                    should_info_live_bar = should_info_live_bar or self._should_log_throttled(
                        f"runner_live_bar_ingested:{symbol}",
                        self._bar_log_throttle_seconds,
                    )
                    if should_info_live_bar:
                        self._logger.debug(
                            "RUNNER_LIVE_BAR_INGESTED symbol=%s timestamp=%s open=%.4f high=%.4f low=%.4f close=%.4f volume=%s runner_history_count=%d candle_version=%d",
                            symbol,
                            completed_bar.timestamp.isoformat(),
                            completed_bar.open,
                            completed_bar.high,
                            completed_bar.low,
                            completed_bar.close,
                            completed_bar.volume,
                            runner_history_count,
                            int(self._candle_versions.get(symbol, 0)),
                            extra={
                                "event": "RUNNER_LIVE_BAR_INGESTED",
                                "symbol": symbol,
                                "timestamp": completed_bar.timestamp.isoformat(),
                                "open": completed_bar.open,
                                "high": completed_bar.high,
                                "low": completed_bar.low,
                                "close": completed_bar.close,
                                "volume": completed_bar.volume,
                                "runner_history_count": runner_history_count,
                                "candle_version": int(self._candle_versions.get(symbol, 0)),
                            },
                        )
                candle_count = len(self._indicator_engine.get_history(symbol) or [])
                should_log_bar_state = completed_bar is not None or self._should_log_throttled(
                    f"runner_bar_state:{symbol}:c0:{candle_count == 0}",
                    self._bar_log_throttle_seconds,
                )
                if should_log_bar_state:
                    self._logger.debug(
                        "RUNNER_BAR_STATE symbol=%s trace_id=%s candle_count=%d required_candles=%d completed_bar=%s",
                        symbol,
                        trace_id,
                        candle_count,
                        self._required_candles,
                        completed_bar is not None,
                        extra={
                            "event": "RUNNER_BAR_STATE",
                            "symbol": symbol,
                            "trace_id": trace_id,
                            "candle_count": candle_count,
                            "required_candles": self._required_candles,
                            "completed_bar": completed_bar is not None,
                        },
                    )
                # NOTE: Do NOT return here when completed_bar is None.
                # The position manager (PHASE 5) must receive every tick to
                # track unrealised P&L and update stop-loss levels in real time.
                # Strategy evaluation is already guarded by the same-bar-skip
                # check in PHASE 9 so it still runs only once per completed bar.
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
                except ValueError:
                    pass  # ✅ FIX: Ignore expected "No open position" errors
                except Exception as e:
                    LOGGER.error(
                        f"Failed to update position price for {symbol}: {e}"
                    )

            # =================================================================
            # PHASE 6: Global readiness gate
            # =================================================================
            if self._runner_state != RunnerState.EXECUTION_ENABLED:
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage="phase6",
                    reason="runner_not_execution_enabled",
                    allowed=False,
                    trace_id=trace_id,
                )
                return
            if not bool(self._runtime_startup_ready):
                now_mono = time_module.monotonic()
                if now_mono - float(self._startup_gate_last_log_ts) >= 5.0:
                    self._startup_gate_last_log_ts = now_mono
                    self._emit_runner_eval_decision(
                        symbol=symbol,
                        stage="phase6",
                        reason=(
                            f"{self._runtime_readiness_reason or 'startup_pipeline_incomplete'} "
                            f"selected_ce={self._active_selected_ce} selected_pe={self._active_selected_pe} "
                            f"startup_ready={self._runtime_startup_ready} data_hard_ready={self._runtime_data_hard_ready} "
                            f"evaluation_ready={self._runtime_evaluation_ready} live_orders_armed={self._runtime_live_orders_armed} "
                            f"runner_state={self._runner_state}"
                        ),
                        allowed=False,
                        trace_id=trace_id,
                    )
                return

            # =================================================================
            # PHASE 7: STRATEGY PREPARATION
            # =================================================================
            # NOTE: in_warmup is intentionally NOT a return gate here.
            # Warmup blocks execution via RunnerState.EXECUTION_ENABLED (PHASE 6
            # already returns if _runner_state != EXECUTION_ENABLED).  Adding a
            # second warmup return in PHASE 7 meant that even after the 15-second
            # grace period expired the evaluation flow never restarted on the same
            # tick, because ticks arriving between seconds 0–15 set _startup_timestamp
            # and the next tick (after 15s) continued past the bar builder but then
            # returned here unconditionally.  Remove it entirely.
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
                # ✅ FIX: DO NOT overwrite _last_cumulative_volume with the internal VWAP
                # accumulator (cum_vol = sum of bar deltas since session open).
                # _last_cumulative_volume must track the EXCHANGE's raw cumulative volume
                # so the delta computation in PHASE FIX S5 stays correct.
                # Overwriting it here with a different accumulator caused delta =
                # raw_exchange_vol - vwap_cum_vol → wrong huge delta → avg_volume = 1.357B.
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

                min_bars_needed = self._required_candles or 20
                

                # BUG W2 FIX: Emit a one-shot INFO log the first time each symbol
                # passes the warmup gate so Railway logs clearly show the moment
                # strategies become active.  Without this, logs only show
                # "Indicators fully hydrated" (from app.py) but never confirm that
                # per-symbol evaluation has actually been unblocked.
                if symbol not in self._warmup_complete_logged:
                    try:
                        _wc_bars = len(self._indicator_engine.get_history(symbol) or [])
                    except Exception:
                        _wc_bars = 0
                    if _wc_bars >= min_bars_needed:
                        self._warmup_complete_logged.add(symbol)
                        self._logger.info(
                            "WARMUP_COMPLETE symbol=%s runner_bars=%d required_candles=%d",
                            symbol,
                            _wc_bars,
                            min_bars_needed,
                            extra={
                                "event": "WARMUP_COMPLETE",
                                "symbol": symbol,
                                "runner_bars": _wc_bars,
                                "required_candles": min_bars_needed,
                            },
                        )
                    elif self._should_log_throttled(
                        f"runner_warmup_pending:{symbol}",
                        self._bar_log_throttle_seconds,
                    ):
                        self._logger.info(
                            "RUNNER_WARMUP_PENDING",
                            extra={
                                "event": "RUNNER_WARMUP_PENDING",
                                "symbol": symbol,
                                "runner_bars": _wc_bars,
                                "required_candles": min_bars_needed,
                                "source": "live_tick_path",
                            },
                        )

                # Strategy evaluation is now purely event-driven. 
                # ExecutionEngine handles any timing constraints.

                spot_tick = self._get_spot_tick()
                # BUG W2 FIX: Previous code had:
                #   if spot_tick is None: spot_stale = True
                #   spot_stale = False          ← unconditional overwrite — dead code
                #   if not spot_tick: spot_stale = True
                # The middle line always reset the flag, making the first branch useless.
                # Correct logic: stale if tick absent OR if timestamp is too old.
                spot_stale = not spot_tick
                spot_ts = None
                if spot_tick:
                    spot_ts = _extract_float(
                        spot_tick,
                        "received_at",
                        "wallclock",
                        "exchange_timestamp",
                        "timestamp",
                        "ts",
                        "ts_ms",
                        "last_trade_time",
                    )
                if spot_ts is not None and spot_ts > 1_000_000_000_000:
                    spot_ts = spot_ts / 1000.0
                spot_age = time.time() - float(spot_ts) if spot_ts is not None else None
                if spot_age is None and self._market_data is not None:
                    try:
                        since = self._market_data.time_since_last_tick("NSE:NIFTY")
                        if since is not None:
                            spot_age = float(since)
                    except Exception:
                        spot_age = None
                spot_max_age = float(
                    os.environ.get(
                        "RUNNER_INDEX_STALE_TICK_SECONDS",
                        str(self._index_stale_tick_seconds),
                    )
                )
                if spot_age is not None and spot_age > spot_max_age:
                    spot_stale = True

                if spot_stale:
                    log_throttled(
                        self._logger,
                        "spot_stale",
                        f"SPOT_STALE age_s={spot_age} threshold_s={spot_max_age}",
                        interval_sec=120.0,
                        level=logging.WARNING,
                    )
                    if not self._spot_stale_flag:
                        self._spot_stale_flag = True
                else:
                    if self._spot_stale_flag:
                        self._spot_stale_flag = False
                        self._logger.info("SPOT_RECOVERED")

                # =============================================================
                # PHASE 8: SIGNAL GENERATION
                # =============================================================
                generated_signal = None

                # 8A. FORCED SIGNAL (Testing only)
                if self._force_signal_enabled and not self._disable_early_forced_signals:
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

                # 8B. PREMIUM MOMENTUM SQUEEZE (disabled in runner by default)
                premium_squeeze_enabled = str(os.getenv("RUNNER_ENABLE_PREMIUM_SQUEEZE", "false")).strip().lower() in {"1", "true", "yes", "on"}
                if generated_signal is None and premium_squeeze_enabled and self._indicator_engine.has_min_bars(symbol, 20):
                    phase = "phase8_premium_squeeze"
                    try:
                        generated_signal = self._maybe_generate_premium_squeeze_signal(
                            symbol,
                            price,
                            trace_id=trace_id,
                        )
                    except Exception as exc:
                        self._logger.exception(
                            "PREMIUM_SQUEEZE_ERROR symbol=%s error_type=%s error=%s trace_id=%s",
                            symbol,
                            type(exc).__name__,
                            exc,
                            trace_id,
                            extra={
                                "event": "PREMIUM_SQUEEZE_ERROR",
                                "symbol": symbol,
                                "trace_id": trace_id,
                                "error_type": type(exc).__name__,
                            },
                        )
                        generated_signal = None

                # 8C. VWAP CROSSOVER (Requires VWAP > 0)
                runner_vwap_crossover_enabled = str(os.getenv("RUNNER_ENABLE_LEGACY_VWAP_CROSSOVER", "false")).strip().lower() in {"1", "true", "yes", "on"}
                if (
                    runner_vwap_crossover_enabled and self._vwap_crossover_enabled
                    and generated_signal is None
                    and state.vwap
                    and state.vwap > 0
                    and "FUT" not in symbol.upper()
                ):
                    phase = "phase8_vwap_crossover"
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

                        sl_pct = self._vwap_sl_pct  # 1.5% SL
                        tp_pct = self._vwap_tp_pct  # 2.0% TP (1:1.33 RR)

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
                                confidence=0.0,
                                reason="vwap_crossover_up",
                                stop_loss=calculated_sl,
                                take_profit=calculated_tp,
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
                                confidence=0.0,
                                reason="vwap_crossover_down",
                                stop_loss=calculated_sl,
                                take_profit=calculated_tp,
                                metadata={
                                    "strategy": "vwap_scalp",
                                    "vwap": curr_vwap,
                                    "tag": "vwap_scalp_short",
                                    "sl_pct": sl_pct,
                                    "tp_pct": tp_pct,
                                },
                            )

                # 8D. FALLBACK STRATEGY: Momentum Breakout (When VWAP is Missing/0)

                # Update last tick
                state.last_tick = dict(tick)

            # =================================================================
            # PHASE 9: SIGNAL SELECTION & STRATEGY MANAGER EVALUATION
            # =================================================================

            self._eval_counter = getattr(self, "_eval_counter", 0) + 1
            phase = "phase9_strategy_manager"
            # Visible INFO trace so Railway logs confirm PHASE 9 is reached
            log_throttled(
                self._logger,
                f"phase9_entry_{symbol}",
                f"🔍 PHASE9 ENTERED: {symbol} | price={price:.2f} | "
                f"skip={skip_strategy} | runner={self._runner_state}",
                interval_sec=60.0,
                level=logging.DEBUG,
            )

            self._last_global_eval_ts = time.monotonic()
            signal = generated_signal
            upstream_version = int(
                tick.get("candle_version")
                or tick.get("version")
                or tick.get("data_version")
                or 0
            )
            if upstream_version > int(self._candle_versions.get(symbol, 0)):
                self._candle_versions[symbol] = upstream_version
            current_version = int(self._candle_versions.get(symbol, 0))
            last_version = int(self._last_strategy_versions.get(symbol, 0))
            candle_count = len(self._indicator_engine.get_history(symbol) or [])
            if current_version <= last_version:
                now_eval_ts = time_module.time()
                last_same_bar_eval = float(
                    self._last_same_bar_eval_ts_by_symbol.get(symbol, 0.0)
                )
                if (
                    self._allow_eval_without_new_bar
                    and candle_count >= self._required_candles
                    and now_eval_ts - last_same_bar_eval
                    >= self._eval_without_new_bar_seconds
                ):
                    self._last_same_bar_eval_ts_by_symbol[symbol] = now_eval_ts
                    self._logger.info(
                        "RUNNER_EVAL_DECISION",
                        extra={
                            "event": "RUNNER_EVAL_DECISION",
                            "symbol": symbol,
                            "trace_id": trace_id,
                            "allowed": True,
                            "reason": "same_bar_periodic_eval",
                            "current_version": current_version,
                            "last_version": last_version,
                            "runner_history_count": candle_count,
                            "required_candles": self._required_candles,
                            "tick_price": price,
                        },
                    )
                else:
                    if self._should_log_throttled(
                        f"runner_eval_same_bar_skip:{symbol}",
                        self._eval_log_throttle_seconds,
                    ):
                        self._logger.info(
                            "RUNNER_EVAL_DECISION",
                            extra={
                                "event": "RUNNER_EVAL_DECISION",
                                "symbol": symbol,
                                "trace_id": trace_id,
                                "allowed": False,
                                "reason": "same_bar_version",
                                "current_version": current_version,
                                "last_version": last_version,
                                "runner_history_count": candle_count,
                                "required_candles": self._required_candles,
                                "tick_price": price,
                            },
                        )
                    return
            else:
                self._logger.info(
                    "RUNNER_EVAL_DECISION",
                    extra={
                        "event": "RUNNER_EVAL_DECISION",
                        "symbol": symbol,
                        "trace_id": trace_id,
                        "allowed": True,
                        "reason": "new_bar_version",
                        "current_version": current_version,
                        "last_version": last_version,
                        "runner_history_count": candle_count,
                        "required_candles": self._required_candles,
                        "tick_price": price,
                    },
                )
            if is_live_mode and not bool(self._runtime_data_hard_ready):
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage="phase9",
                    reason=str(self._runtime_readiness_reason or "runtime_data_not_ready"),
                    allowed=False,
                    trace_id=trace_id,
                )
                return
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
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage="phase9",
                    reason="symbol_rate_limit_backoff_active",
                    allowed=False,
                    trace_id=trace_id,
                    remaining_s=max(0.0, symbol_rate_until - now_ts),
                )
                return

            backoff_until = float(getattr(self, "_data_freshness_backoff_until", 0.0))
            if (
                self._should_enforce_freshness_backoff()
                and backoff_until
                and time_module.time() < backoff_until
            ):
                remaining = max(0.0, backoff_until - time_module.time())
                log_throttled(
                    self._logger,
                    f"freshness_backoff_{symbol}",
                    f"⏰ Freshness backoff: {symbol} remaining={remaining:.1f}s "
                    f"detail={getattr(self,'_data_freshness_backoff_detail',None)}",
                    interval_sec=30.0,
                    level=logging.WARNING,
                )
                self._emit_runner_eval_decision(
                    symbol=symbol,
                    stage="phase9",
                    reason="data_freshness_backoff_active",
                    allowed=False,
                    trace_id=trace_id,
                    remaining_s=remaining,
                    backoff_until=backoff_until,
                )
                return

            current_state = SymbolState.DEGRADED
            if state.active:
                ready_state = SymbolState.READY
                current_state = ready_state

            if signal is None and self._required_candles:
                should_evaluate = False
                phase = self._data_phase.get(symbol, "HYDRATION")
                if phase != "LIVE":
                    fallback_enabled = bool(
                        getattr(self, "_fallback_enabled", False)
                        or getattr(self, "_allow_polling_fallback", True)
                    )
                    if not fallback_enabled:
                        self._emit_runner_eval_decision(
                            symbol=symbol,
                            stage="phase9",
                            reason="non_live_phase_and_fallback_disabled",
                            allowed=False,
                            trace_id=trace_id,
                        )
                        return
                    if not self._symbol_history.get(symbol):
                        self._emit_runner_eval_decision(
                            symbol=symbol,
                            stage="phase9",
                            reason="non_live_phase_and_empty_symbol_history",
                            allowed=False,
                            trace_id=trace_id,
                        )
                        return
                with self._lock:
                    state = self._symbol_state.get(symbol)
                    if state:
                        last_eval = getattr(state, "_last_strategy_eval", None)
                        last_bar_ts = self._last_bar_ts.get(symbol)
                        if last_bar_ts is None:
                            # FIX (2026-02-27): Soft skip — do NOT call _mark_symbol_unready.
                            # That sets HYDRATING → PHASE 7 blocks → permanent cycle.
                            # Instead seed _last_bar_ts = now and retry next tick.
                            log_throttled(
                                self._logger,
                                f"bar_ts_init_{symbol}",
                                f"⚙️  Seeding _last_bar_ts for {symbol} (first eval tick)",
                                interval_sec=60.0,
                                level=logging.INFO,
                            )
                            self._last_bar_ts[symbol] = now
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="phase9",
                                reason="bar_ts_initialized_retry_next_tick",
                                allowed=False,
                                trace_id=trace_id,
                            )
                            return
                        # ✅ FIX K: Same-bar-skip for options with no live bars.
                        # When _symbol_history is empty (no live bars ever completed),
                        # _last_bar_ts[symbol] is the last HYDRATION bar timestamp —
                        # a static value that never advances.  Using it for the same-bar-skip
                        # comparison means after the FIRST evaluation, state._last_eval_bar_ts
                        # equals last_bar_ts PERMANENTLY, blocking every subsequent evaluation.
                        # Fix: when no live bars, use current-minute-bucket as the effective
                        # bar_ts (mirrors the PHASE 7 same-bar-skip logic exactly).
                        _has_live_bars_p9 = bool(self._symbol_history.get(symbol))
                        _effective_last_bar_ts = last_bar_ts
                        if not _has_live_bars_p9 and last_bar_ts is not None:
                            _now_bucket = time_module.time()
                            _bucket_ts = _now_bucket - (_now_bucket % 60)
                            _effective_last_bar_ts = datetime.fromtimestamp(
                                _bucket_ts, tz=timezone.utc
                            )
                        disable_same_bar_skip = phase != "LIVE"
                        if (
                            not disable_same_bar_skip
                            and _effective_last_bar_ts
                            and state._last_eval_bar_ts
                        ):
                            if _effective_last_bar_ts <= state._last_eval_bar_ts:
                                logged_map = getattr(
                                    self, "_same_bar_skip_logged", None
                                )
                                if logged_map is None:
                                    logged_map = {}
                                    self._same_bar_skip_logged = logged_map
                                extra_payload = {
                                    "event": "strategy_eval_skipped_same_bar",
                                    "symbol": symbol,
                                    "bar_ts": _effective_last_bar_ts.isoformat(),
                                }
                                if logged_map.get(symbol) != _effective_last_bar_ts:
                                    logged_map[symbol] = _effective_last_bar_ts
                                    self._logger.debug(
                                        "Condition met: strategy_eval_skipped_same_bar",
                                        extra=extra_payload,
                                    )
                                self._emit_runner_eval_decision(
                                    symbol=symbol,
                                    stage="phase9",
                                    reason="strategy_eval_skipped_same_bar",
                                    allowed=False,
                                    trace_id=trace_id,
                                    effective_bar_ts=_effective_last_bar_ts.isoformat(),
                                )
                                return
                        if last_bar_ts:
                            bar_age = (now - last_bar_ts).total_seconds()
                            # ✅ FIX E: Raise stale-bar threshold for options/futures.
                            # Options tick once per ~13 min; bar_builder needs 2 different
                            # BUG W1 FIX: Use _live_bar_seen (populated on first live
                            # bar, never during backfill) instead of _symbol_history
                            # (populated during hydration with OLD bars).  Before this
                            # fix: _symbol_history always non-empty at startup →
                            # has_live_bars=True → bar_age=(18h hydration age) >> threshold
                            # → stale-bar gate fires on the VERY FIRST tick → PHASE-7
                            # same-bar-skip (already set) locks all subsequent ticks →
                            # zero strategy evaluations until first live bar closes.
                            has_live_bars = symbol in self._live_bar_seen
                            _is_nfo = any(x in symbol for x in ("CE", "PE", "FUT"))
                            _stale_bar_max = 900.0 if _is_nfo else 180.0
                            if has_live_bars and bar_age > _stale_bar_max:
                                log_throttled(
                                    self._logger,
                                    f"strategy_eval_stale_bar_{symbol}",
                                    "Condition met: strategy_eval_stale_bar",
                                    interval_sec=300.0,
                                    level=logging.WARNING,
                                    extra={
                                        "event": "strategy_eval_stale_bar",
                                        "symbol": symbol,
                                        "bar_age_s": bar_age,
                                        "bar_ts": last_bar_ts.isoformat(),
                                    },
                                )
                                self._emit_runner_eval_decision(
                                    symbol=symbol,
                                    stage="phase9",
                                    reason="strategy_eval_stale_bar",
                                    allowed=False,
                                    trace_id=trace_id,
                                    bar_age_s=bar_age,
                                    stale_bar_threshold_s=_stale_bar_max,
                                )
                                return
                        # Frequency limit moved to ExecutionEngine
                        state._last_strategy_eval = now
                        pending_eval_bar_ts = _effective_last_bar_ts
                        should_evaluate = True

                if should_evaluate:
                    if not self._strategy_evaluation_allowed(symbol, trace_id):
                        return
                    log_throttled(
                        self._logger,
                        f"strategy_evaluation_triggered:{symbol}",
                        "Strategy evaluation triggered",
                        interval_sec=self._eval_log_throttle_seconds,
                        level=logging.DEBUG,
                        extra={"event": "strategy_evaluation_triggered", "symbol": symbol},
                    )
                    # ✅ DIAGNOSTIC LOG: Confirm evaluation is happening
                    log_throttled(
                        self._logger,
                        f"strategy_eval_{symbol}",
                        f"🎯 EVALUATING STRATEGIES: {symbol} | min_bars={self._required_candles}",
                        interval_sec=30.0,
                        level=logging.DEBUG,
                    )
                    # ✅ DIAGNOSTIC LOG: Confirm indicators are ready
                    log_throttled(
                        self._logger,
                        f"indicators_ready_{symbol}",
                        f"✅ INDICATORS READY: {symbol} | Calling StrategyManager...",
                        interval_sec=60.0,
                        level=logging.DEBUG,
                    )
                    if (
                        self._market_data is not None
                        and hasattr(self._market_data, "is_data_stale")
                        and self._market_data.is_data_stale()
                    ):
                        self._emit_runner_eval_decision(
                            symbol=symbol,
                            stage="phase9",
                            reason="market_data_global_stale_diagnostic",
                            allowed=True,
                            trace_id=trace_id,
                        )

                    mdm_last_tick = getattr(
                        self._market_data, "_last_tick_time", {}
                    ).get(symbol)
                    _stale_thresh = self._stale_tick_threshold_for_symbol(symbol)
                    if (
                        isinstance(mdm_last_tick, (int, float))
                        and time.time() - float(mdm_last_tick) > _stale_thresh
                    ):
                        _mdm_age = time.time() - float(mdm_last_tick)
                        upper_symbol = symbol.upper()
                        if upper_symbol in {"NSE:NIFTY", "NIFTY", "NSE:NIFTY 50", "NIFTY 50"}:
                            fallback_ltp = None
                            if self._market_data is not None and hasattr(self._market_data, "get_ltp"):
                                try:
                                    fallback_ltp = self._market_data.get_ltp(symbol)
                                except Exception:
                                    fallback_ltp = None
                            log_throttled(
                                self._logger,
                                f"stale_mdm_tick_{symbol}",
                                f"Stale MDM tick for {symbol} age={_mdm_age:.1f}s; using fallback and continuing",
                                interval_sec=60.0,
                                level=logging.WARNING,
                            )
                            if fallback_ltp and fallback_ltp > 0:
                                self._logger.info(
                                    "MDM_REST_FALLBACK_USED symbol=%s reason=stale_ws_tick",
                                    symbol,
                                    extra={"event": "MDM_REST_FALLBACK_USED", "symbol": symbol, "reason": "stale_ws_tick"},
                                )
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="phase9",
                                reason="stale_mdm_tick_index_fallback",
                                allowed=True,
                                trace_id=trace_id,
                                mdm_tick_age_s=_mdm_age,
                                stale_tick_threshold_s=_stale_thresh,
                            )
                        else:
                            log_throttled(
                                self._logger,
                                f"stale_mdm_tick_{symbol}",
                                f"Stale MDM tick — skipping: {symbol} age={_mdm_age:.1f}s",
                                interval_sec=60.0,
                                level=logging.WARNING,
                            )
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="phase9",
                                reason="stale_mdm_tick",
                                allowed=False,
                                trace_id=trace_id,
                                mdm_tick_age_s=_mdm_age,
                                stale_tick_threshold_s=_stale_thresh,
                            )
                            return
                    self._last_global_eval_ts = time.monotonic()
                    self._logger.debug(
                        "strategy_evaluation_start",
                        extra={
                            "event": "strategy_evaluation_start",
                            "symbol": symbol,
                            "price": price,
                            "state": current_state.value,
                        },
                    )
                    self._emit_runner_eval_decision(
                        symbol=symbol,
                        stage="phase9",
                        reason="evaluation_entered",
                        allowed=True,
                        trace_id=trace_id,
                        price=price,
                    )
                    try:
                        indicators_ctx = self._indicator_engine.get_indicators(symbol)
                        selected_ce = getattr(self, "_active_selected_ce", None)
                        selected_pe = getattr(self, "_active_selected_pe", None)
                        atm_strike = getattr(self, "_active_atm_strike", None)
                        symbol_strike = self._extract_strike_from_symbol(symbol)
                        normalized_symbol = normalize_symbol(symbol)
                        selected_set = {normalize_symbol(item) for item in [selected_ce, selected_pe] if item}
                        runtime_ctx: dict[str, Any] = {
                            "selected_ce": selected_ce,
                            "selected_pe": selected_pe,
                            "atm_strike": atm_strike,
                            "is_selected_option": normalized_symbol in selected_set,
                        }
                        option_side = None
                        symbol_upper = normalized_symbol.upper()
                        if symbol_upper.endswith("CE"):
                            option_side = "CE"
                        elif symbol_upper.endswith("PE"):
                            option_side = "PE"
                        if option_side:
                            runtime_ctx.setdefault("contract_side", option_side)
                            runtime_ctx.setdefault("option_contract_side", option_side)
                        existing_bias = (
                            indicators_ctx.get("direction_bias")
                            or indicators_ctx.get("underlying_direction_bias")
                            or getattr(self, "_latest_direction_bias", None)
                        )
                        if str(existing_bias or "").upper() in {"CE", "PE"}:
                            runtime_ctx["direction_bias"] = str(existing_bias).upper()
                        if symbol_strike is not None and atm_strike is not None:
                            runtime_ctx["strike_distance_from_atm"] = abs(float(symbol_strike) - float(atm_strike))
                        quote_payload = self.get_quote(symbol) if hasattr(self, "get_quote") else {}
                        quote_map = dict(quote_payload) if isinstance(quote_payload, Mapping) else {}
                        last_tick_store = getattr(self, "_last_tick", None)
                        if isinstance(last_tick_store, Mapping):
                            tick_map = dict(
                                last_tick_store.get(symbol)
                                or last_tick_store.get(normalized_symbol)
                                or {}
                            )
                        else:
                            tick_map = {}
                        depth_payload = quote_map.get("depth") or tick_map.get("depth")
                        bid = quote_map.get("bid") or quote_map.get("best_bid") or tick_map.get("bid") or tick_map.get("best_bid")
                        ask = quote_map.get("ask") or quote_map.get("best_ask") or tick_map.get("ask") or tick_map.get("best_ask")
                        bid_f = _extract_float({"value": bid}, "value")
                        ask_f = _extract_float({"value": ask}, "value")
                        bid_qty = quote_map.get("bid_qty") or tick_map.get("bid_qty") or quote_map.get("buy_qty") or tick_map.get("buy_qty")
                        ask_qty = quote_map.get("ask_qty") or tick_map.get("ask_qty") or quote_map.get("sell_qty") or tick_map.get("sell_qty")
                        spread = quote_map.get("spread")
                        mid = quote_map.get("mid")
                        if spread in (None, "") and bid_f is not None and ask_f is not None:
                            spread = ask_f - bid_f
                        if mid in (None, "") and bid_f is not None and ask_f is not None:
                            mid = (ask_f + bid_f) / 2.0
                        spread_pct = quote_map.get("spread_pct")
                        if spread_pct in (None, "") and spread not in (None, "") and mid not in (None, "", 0):
                            spread_f = _extract_float({"value": spread}, "value")
                            mid_f = _extract_float({"value": mid}, "value")
                            if spread_f is not None and mid_f not in (None, 0.0):
                                spread_pct = (spread_f / mid_f) * 100.0
                        tradable_quote = bool(quote_map.get("tradable_quote"))
                        if not tradable_quote and bid_f is not None and ask_f is not None:
                            tradable_quote = ask_f > bid_f
                        runtime_ctx.update({
                            "bid": bid,
                            "ask": ask,
                            "best_bid": bid,
                            "best_ask": ask,
                            "bid_qty": bid_qty,
                            "ask_qty": ask_qty,
                            "buy_qty": quote_map.get("buy_qty") or tick_map.get("buy_qty") or bid_qty,
                            "sell_qty": quote_map.get("sell_qty") or tick_map.get("sell_qty") or ask_qty,
                            "depth": depth_payload,
                            "depth_available": bool(quote_map.get("depth_available") or depth_payload),
                            "tradable_quote": tradable_quote,
                            "spread": spread,
                            "mid": mid,
                            "spread_pct": spread_pct,
                            "bid_ask_source": quote_map.get("bid_ask_source") or tick_map.get("bid_ask_source") or "runner_context",
                            "tick_direction": quote_map.get("tick_direction") or tick_map.get("tick_direction"),
                            "data_age_seconds": quote_map.get("data_age_seconds") or tick_map.get("data_age_seconds"),
                            "quote_age_s": quote_map.get("quote_age_s") or quote_map.get("data_age_seconds") or tick_map.get("data_age_seconds"),
                        })
                        if hasattr(self._indicator_engine, "set_runtime_context"):
                            self._indicator_engine.set_runtime_context(symbol, runtime_ctx)
                        elif self._should_log_throttled(
                            "indicator_ctx_missing_setter", 60.0
                        ):
                            self._logger.warning(
                                "INDICATOR_CONTEXT_SETTER_MISSING symbol=%s engine=%s",
                                symbol,
                                type(self._indicator_engine).__name__,
                            )
                        require_depth_for_signal = (
                            os.getenv("EXECUTION_MODE", "PAPER").strip().upper() == "LIVE"
                            and _env_bool("REQUIRE_MARKET_DEPTH_FOR_SIGNAL", False)
                        )
                        market_depth_ok = self.validate_market_depth()
                        if not market_depth_ok and require_depth_for_signal:
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="phase9",
                                reason="market_depth_invalid",
                                allowed=False,
                                trace_id=trace_id,
                            )
                            return
                        if not market_depth_ok:
                            log_throttled(
                                self._logger,
                                f"market_depth_soft_fail:{symbol}",
                                "MARKET_DEPTH_SOFT_FAIL_SIGNAL_CONTINUES",
                                interval_sec=60.0,
                                level=logging.INFO,
                                extra={"event": "MARKET_DEPTH_SOFT_FAIL_SIGNAL_CONTINUES", "symbol": symbol},
                            )

                        signal = self._strategy_manager.generate_signal(
                            symbol,
                            price,
                            trace_id=trace_id,
                        )
                        if pending_eval_bar_ts is not None:
                            with self._lock:
                                state_after = self._symbol_state.get(symbol)
                                if state_after is not None:
                                    state_after._last_eval_bar_ts = pending_eval_bar_ts
                        if signal is None:
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="phase9",
                                reason="evaluation_no_signal",
                                allowed=True,
                                trace_id=trace_id,
                                price=price,
                            )
                        else:
                            self._emit_runner_eval_decision(
                                symbol=symbol,
                                stage="signal_forward",
                                reason="signal_forwarded",
                                allowed=True,
                                trace_id=trace_id,
                                signal_action=signal.action,
                                signal_confidence=getattr(signal, "confidence", None),
                            )
                        self._logger.debug(
                            "STRATEGY_EVALUATED",
                            extra={
                                "event": "strategy_evaluated",
                                "symbol": symbol,
                                "price": price,
                                "has_signal": signal is not None,
                                "signal_action": signal.action if signal else None
                            }
                        )
                        self._last_strategy_versions[symbol] = current_version
                    except Exception as exc:
                        self._logger.exception(
                            "SIGNAL_EVALUATION_FAILURE symbol=%s phase=%s error_type=%s error=%s trace_id=%s",
                            symbol,
                            "phase9",
                            type(exc).__name__,
                            str(exc),
                            trace_id,
                            extra={"event": "SIGNAL_EVALUATION_FAILURE", "symbol": symbol, "phase": "phase9", "error_type": type(exc).__name__, "error": str(exc), "trace_id": trace_id},
                        )
                        signal = None
                    self._strategy_window_symbols.add(symbol)
                    if signal is not None:
                        self._signal_counter += 1
                        self._strategy_window_signals += 1
                        
                        # --- Objective 8: Prometheus metrics ---
                        signals_generated_total.labels(
                            symbol=symbol,
                            strategy=str(signal.metadata.get("strategy") if signal.metadata else "unknown")
                        ).inc()

                        self._logger.info(
                            "SIGNAL_CANDIDATE_GENERATED",
                            extra={
                                "event": "SIGNAL_CANDIDATE_GENERATED",
                                "strategy": str(
                                    signal.metadata.get("strategy")
                                    if signal.metadata
                                    else "unknown"
                                ),
                                "symbol": symbol,
                                "direction": signal.action,
                            },
                        )
                        self._signals_last_hour.append(time.time())

                    now_ts = time.time()
                    if now_ts - self._last_signal_frequency_check_ts >= 300.0:
                        self._last_signal_frequency_check_ts = now_ts
                        signals_last_60m = sum(
                            1
                            for signal_ts in self._signals_last_hour
                            if now_ts - signal_ts <= 3600
                        )
                        if signals_last_60m < 2:
                            self._logger.warning(
                                "Low signal frequency detected (%s in last hour)",
                                signals_last_60m,
                                extra={"event": "low_signal_frequency"},
                            )
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
                phase = "phase10_signal_execution"
                self._last_strategy_versions[symbol] = current_version
                current_regime = self._compute_regime_snapshot(symbol)
                signal_metadata = dict(signal.metadata or {})
                signal_metadata["runtime_regime"] = current_regime.value
                signal_metadata["runtime_regime_inputs"] = self._last_regime_inputs_by_symbol.get(symbol, {})
                signal = dataclasses.replace(signal, metadata=signal_metadata)
                signal_strategy = str((signal.metadata or {}).get("strategy") or "")
                coarse_regime = self.detect_market_regime(symbol)
                if coarse_regime == "low_volatility":
                    if self._should_log_throttled(
                        f"runner_regime_low_vol:{symbol}",
                        self._eval_log_throttle_seconds,
                    ):
                        self._logger.info(
                            "RUNNER_REGIME_DECISION",
                            extra={
                                "event": "RUNNER_REGIME_DECISION",
                                "symbol": symbol,
                                "regime": "low_volatility",
                                "hard_block": self._block_low_volatility,
                            },
                        )
                    if self._block_low_volatility:
                        self._logger.info(
                            "SIGNAL_EXECUTION_DECISION symbol=%s stage=phase10 decision=low_volatility_rejected trace_id=%s strategy=%s action=%s confidence=%.2f",
                            symbol, trace_id, signal_strategy or "unknown", signal.action, float(signal.confidence or 0.0),
                            extra={"event": "SIGNAL_EXECUTION_DECISION", "symbol": symbol, "stage": "phase10", "decision": "low_volatility_rejected", "trace_id": trace_id, "strategy": signal_strategy or "unknown", "action": signal.action, "confidence": float(signal.confidence or 0.0)},
                        )
                        return
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
                        symbol_now = time.time()
                        last_signal_ts = self._symbol_last_signal_ts.get(symbol, 0.0)
                        if (
                            symbol_now - last_signal_ts
                            < self._config.signal_cooldown_seconds
                        ):
                            elapsed_s = max(symbol_now - last_signal_ts, 0.0)
                            if self._should_log_throttled(
                                f"runner_signal_cooldown:{symbol}",
                                self._cooldown_log_throttle_seconds,
                            ):
                                self._logger.info(
                                    "RUNNER_SIGNAL_COOLDOWN",
                                    extra={
                                        "event": "RUNNER_SIGNAL_COOLDOWN",
                                        "symbol": symbol,
                                        "elapsed_s": round(elapsed_s, 3),
                                        "required_s": float(self._config.signal_cooldown_seconds),
                                        "allowed": False,
                                    },
                                )
                            return

                        state.strategy_data["last_signal"] = {
                            "action": signal.action,
                            "reason": signal.reason,
                            "timestamp": now.isoformat(),
                        }

                self._logger.debug(
                    "SIGNAL_EXECUTING symbol=%s action=%s reason=%s price=%.2f",
                    symbol, signal.action, signal.reason, price,
                    extra={"event": "signal_executing", "symbol": symbol,
                           "action": signal.action},
                )
                self._logger.info(
                    "SIGNAL_GENERATED symbol=%s action=%s reason=%s trace_id=%s",
                    symbol,
                    signal.action,
                    signal.reason,
                    trace_id,
                    extra={
                        "event": "SIGNAL_GENERATED",
                        "symbol": symbol,
                        "action": signal.action,
                        "reason": signal.reason,
                        "trace_id": trace_id,
                    },
                )
                scheduled, prepare_reason = self._schedule_signal_preparation(
                    signal, price, now, trace_id
                )
                if not scheduled:
                    self._emit_runner_eval_decision(
                        symbol=symbol,
                        stage="phase10_execute",
                        reason=str(prepare_reason or "signal_prepare_failed"),
                        allowed=False,
                        trace_id=trace_id,
                    )
                    self._logger.info(
                        "SIGNAL_EXECUTION_RESULT symbol=%s accepted=%s reason=%s order_id=%s trace_id=%s",
                        symbol, False, prepare_reason, None, trace_id,
                        extra={"event": "SIGNAL_EXECUTION_RESULT", "symbol": symbol, "accepted": False, "reason": prepare_reason, "order_id": None, "trace_id": trace_id},
                    )
                return
        except Exception as exc:
            self._logger.error(
                "RUNNER_ON_TICK_ERROR symbol=%s phase=%s error_type=%s error=%s",
                symbol,
                phase,
                type(exc).__name__,
                str(exc),
                extra={
                    "event": "RUNNER_ON_TICK_ERROR",
                    "symbol": symbol,
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                exc_info=True,
            )
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

    def verify_state(self) -> bool:
        """Validate broker state before order release. Args: none. Returns: bool. Raises: none."""
        try:
            broker = getattr(self, "_broker", None) or getattr(
                self._order_manager, "_broker", None
            )
            if broker is None or not hasattr(broker, "get_positions"):
                return True

            # Prefer the underlying sync client (ZerodhaKiteClient) when the
            # broker wrapper is an async RobustDataProvider — calling an async
            # method synchronously from within a running event loop is not safe.
            sync_broker = getattr(
                broker,
                "client",
                getattr(broker, "_broker", broker),
            )
            method = getattr(sync_broker, "get_positions", None)
            if method is None:
                return True

            broker_positions = method()
            if asyncio.iscoroutine(broker_positions):
                # If we still ended up with a coroutine, close it to prevent
                # "coroutine was never awaited" warnings and allow trading to
                # proceed — blocking on an unresolvable async call is worse.
                broker_positions.close()
                return True
            return broker_positions is not None
        except Exception as e:
            self._logger.error(f"State verification failed: {e}")
            return False

    def _risk_kill_switch_triggered(self) -> bool:
        """Check hard trading kill switches. Args: none. Returns: bool. Raises: none."""
        settings = self._settings if hasattr(self, "_settings") else get_settings()
        metrics_obj = getattr(self, "metrics", None)
        daily_pnl = float(getattr(metrics_obj, "daily_pnl", 0.0) or 0.0)
        consecutive_losses = int(getattr(metrics_obj, "consecutive_losses", 0) or 0)
        max_daily_loss = float(getattr(settings, "max_daily_loss", float("inf")) or 0.0)
        if max_daily_loss > 0 and daily_pnl <= -max_daily_loss:
            self._logger.critical("Max daily loss hit. Trading halted.")
            return True
        max_consecutive_losses = int(
            getattr(settings, "max_consecutive_losses", 0) or 0
        )
        if max_consecutive_losses > 0 and consecutive_losses >= max_consecutive_losses:
            self._logger.critical("Consecutive loss limit hit.")
            return True
        return False

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

    def _maybe_generate_premium_squeeze_signal(
        self,
        symbol: str,
        price: float,
        *,
        trace_id: str | None = None,
    ) -> Signal | None:
        """Args: symbol, price, trace_id. Returns: Signal | None. Raises: none."""
        upper_symbol = symbol.upper()
        if not upper_symbol.endswith(("CE", "PE")) or "FUT" in upper_symbol:
            return None
        underlying = self._extract_underlying(symbol) or "NIFTY"
        now_epoch = time.time()
        last_ts = float(self._premium_squeeze_last_signal_ts.get(underlying, 0.0))
        if now_epoch - last_ts < self._underlying_signal_cooldown_seconds:
            log_throttled(
                self._logger,
                f"premium_squeeze_generation_suppressed_{underlying}",
                "PREMIUM_SQUEEZE_GENERATION_SUPPRESSED",
                interval_sec=self._cooldown_log_throttle_seconds,
                level=logging.DEBUG,
                extra={
                    "event": "PREMIUM_SQUEEZE_GENERATION_SUPPRESSED",
                    "underlying": underlying,
                    "trace_id": trace_id,
                },
            )
            return None
        inds = self._indicator_engine.get_indicators(symbol)
        rsi = inds.get("rsi")
        vwap = inds.get("vwap")
        ema = inds.get("ema")
        if rsi is None or vwap is None or vwap <= 0:
            return None
        is_bullish_premium = price > vwap
        if ema is not None:
            is_bullish_premium = is_bullish_premium and price > ema
        is_momentum_active = 60 < rsi < 85
        if not (is_bullish_premium and is_momentum_active):
            return None
        selected = False
        near_atm = False
        in_active_universe = False
        if _env_flag("PREMIUM_FALLBACK_ONLY_SELECTED_OR_NEAR_ATM", True):
            max_strike_distance = float(
                os.getenv("PREMIUM_FALLBACK_MAX_STRIKE_DISTANCE", "100") or "100"
            )
            selected_ce = self._active_selected_ce
            selected_pe = self._active_selected_pe
            atm_strike = self._active_atm_strike
            active_option_symbols = {
                normalize_symbol(option_symbol)
                for option_symbol in getattr(self, "_active_option_symbols", set())
                if option_symbol
            }
            normalized_selected_ce = normalize_symbol(selected_ce) if selected_ce else ""
            normalized_selected_pe = normalize_symbol(selected_pe) if selected_pe else ""
            normalized_symbol = normalize_symbol(symbol)
            if not (normalized_selected_ce or normalized_selected_pe) and active_option_symbols and atm_strike:
                selected_candidates = [
                    item for item in active_option_symbols if self._extract_strike_from_symbol(item)
                ]
                ce_candidates = [item for item in selected_candidates if item.endswith("CE")]
                pe_candidates = [item for item in selected_candidates if item.endswith("PE")]
                if ce_candidates:
                    normalized_selected_ce = min(
                        ce_candidates,
                        key=lambda item: abs(float(self._extract_strike_from_symbol(item) or 0) - float(atm_strike)),
                    )
                if pe_candidates:
                    normalized_selected_pe = min(
                        pe_candidates,
                        key=lambda item: abs(float(self._extract_strike_from_symbol(item) or 0) - float(atm_strike)),
                    )
            strike_value = self._extract_strike_from_symbol(symbol)
            inds = self._indicator_engine.get_indicators(symbol)
            recovered_atm = inds.get("atm_strike") if isinstance(inds, dict) else None
            if atm_strike in (None, 0) and recovered_atm not in (None, ""):
                try:
                    atm_strike = int(float(recovered_atm))
                except (TypeError, ValueError):
                    atm_strike = atm_strike
            if (atm_strike in (None, 0)) and not selected_ce and not selected_pe:
                self._logger.info("PREMIUM_SQUEEZE_SKIPPED reason=missing_active_option_context symbol=%s trace_id=%s", symbol, trace_id)
                return None
            strike = float(strike_value or 0.0)
            atm_strike_float = float(atm_strike or 0.0)
            selected = normalized_symbol in {
                normalized_selected_ce,
                normalized_selected_pe,
            }
            near_atm = bool(
                atm_strike > 0
                and strike > 0
                and abs(strike - atm_strike_float) <= max_strike_distance
            )
            in_active_universe = normalized_symbol in active_option_symbols
            if not (selected or near_atm or in_active_universe):
                if self._should_log_throttled(f"premium_outside_window:{normalized_symbol}", 60.0):
                    self._logger.info(
                    "PREMIUM_SQUEEZE_SKIPPED reason=outside_selected_strike_window symbol=%s selected_ce=%s selected_pe=%s atm_strike=%s strike=%s max_distance=%s trace_id=%s",
                    symbol,
                    normalized_selected_ce,
                    normalized_selected_pe,
                    atm_strike_float,
                    strike,
                    max_strike_distance,
                    trace_id,
                )
                return None
        sl_pct = self._vwap_sl_pct
        tp_pct = self._vwap_tp_pct
        calculated_sl = price * (1 - sl_pct / 100)
        calculated_tp = price * (1 + tp_pct / 100)
        log_throttled(
            self._logger,
            f"premium_squeeze_signal_emitted:{symbol}",
            f"PREMIUM_SQUEEZE_SIGNAL_EMITTED symbol={symbol} rsi={float(rsi):.2f} trace_id={trace_id}",
            interval_sec=60.0,
            level=logging.INFO,
        )
        # --- Premium squeeze quality components: local to this function ---
        direction_score = 7.0
        strategy_score = 6.5
        option_score = 6.5
        data_score = 6.5
        rr_score = 6.0

        price_above_vwap = False
        price_above_ema = False
        momentum_window_active = False
        selected_or_near_atm = bool(selected or near_atm)

        try:
            price_above_vwap = bool(float(price) > float(vwap))
        except (TypeError, ValueError):
            price_above_vwap = False

        try:
            price_above_ema = bool(ema is not None and float(price) > float(ema))
        except (TypeError, ValueError):
            price_above_ema = False

        try:
            momentum_window_active = 65.0 <= float(rsi) <= 82.0
        except (TypeError, ValueError):
            momentum_window_active = False

        if price_above_vwap and price_above_ema:
            direction_score += 1.0
        if momentum_window_active:
            direction_score += 1.0

        if momentum_window_active:
            strategy_score += 1.0
        if price_above_vwap:
            strategy_score += 1.0

        if selected_or_near_atm:
            option_score = max(option_score, 7.5)
        elif in_active_universe:
            option_score = max(option_score, 7.0)

        try:
            history_count = len(self._indicator_engine.get_history(symbol))
        except Exception:
            history_count = 0

        required_history = int(getattr(self, "_warmup_bars_required", 20) or 20)
        if history_count >= required_history:
            data_score = 9.0
        elif history_count >= 5:
            data_score = 8.0

        risk = max(float(price) - float(calculated_sl), 0.0)
        reward = max(float(calculated_tp) - float(price), 0.0)
        if risk > 0 and reward > 0:
            rr_score = max(0.0, min(10.0, (reward / risk) * 5.0))

        direction_score = max(0.0, min(10.0, direction_score))
        strategy_score = max(0.0, min(10.0, strategy_score))
        option_score = max(0.0, min(10.0, option_score))
        data_score = max(0.0, min(10.0, data_score))
        rr_score = max(0.0, min(10.0, rr_score))
        premium_rr = (float(calculated_tp) - float(price)) / max(float(price) - float(calculated_sl), 1e-9)
        confidence = max(
            0.55,
            min(
                0.85,
                (direction_score + strategy_score + option_score + data_score + rr_score) / 50.0,
            ),
        )
        return Signal(
            action="BUY",
            symbol=symbol,
            quantity=1,
            confidence=confidence,
            reason="premium_momentum_squeeze",
            stop_loss=calculated_sl,
            take_profit=calculated_tp,
            metadata={
                "strategy": "premium_momentum_squeeze",
                "vwap": vwap,
                "rsi": rsi,
                "tag": "premium_squeeze",
                "feature": "premium_momentum_squeeze",
                "strategy_name": "premium_momentum_squeeze",
                "is_selected_option": bool(selected),
                "strike_distance_from_atm": abs(strike - atm_strike_float) if strike > 0 and atm_strike_float > 0 else None,
                "premium_stop_distance": max(float(price) - float(calculated_sl), 0.0),
                "premium_target_rr": premium_rr,
                "direction_score": direction_score,
                "strategy_score": strategy_score,
                "setup_quality": strategy_score,
                "confidence": confidence,
                "option_score": option_score,
                "data_score": data_score,
                "rr_score": rr_score,
            },
        )

    def _materialize_option_trade_plan(
        self,
        signal: Signal,
        *,
        execution_price: float,
        atr: float,
        entry_side: OrderSide,
    ) -> Signal:
        """Build final long-option SL/TP from metadata, then normalize geometry."""
        metadata = dict(signal.metadata or {})
        entry_price = float(execution_price or metadata.get("entry_price") or metadata.get("price") or 0.0)
        if entry_price <= 0:
            return signal

        def _to_float(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        stop_loss = _to_float(signal.stop_loss)
        take_profit = _to_float(signal.take_profit)
        rr = _to_float(metadata.get("premium_target_rr")) or 2.0
        stop_distance = _to_float(metadata.get("premium_stop_distance"))
        explicit_stop = _to_float(metadata.get("setup_invalidation_premium")) or _to_float(metadata.get("premium_stop_price"))
        plan_source = "existing_signal_levels"
        if stop_loss is None or stop_loss <= 0 or take_profit is None or take_profit <= 0:
            if metadata.get("premium_stop_pct") is not None:
                signal = self._apply_premium_targets(signal, premium=entry_price, entry_side=entry_side)
                stop_loss = _to_float(signal.stop_loss)
                take_profit = _to_float(signal.take_profit)
                plan_source = "premium_stop_pct"
            else:
                distance = stop_distance if stop_distance is not None and stop_distance > 0 else max(atr * 1.2, entry_price * 0.02, 1.0)
                if explicit_stop is not None and explicit_stop > 0 and str(entry_side).upper() == "BUY":
                    stop_loss = explicit_stop
                    plan_source = "explicit_premium_stop"
                else:
                    stop_loss = entry_price - distance
                    plan_source = "premium_stop_distance"
                risk = max(entry_price - float(stop_loss), max(atr * 0.8, 1.0))
                take_profit = entry_price + risk * rr
            signal = dataclasses.replace(
                signal,
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                metadata={**metadata, "entry_price": entry_price, "option_trade_plan_source": plan_source},
            )
        signal = self._validate_long_option_geometry(signal=signal, entry_price=entry_price, entry_side=entry_side, atr=atr)
        signal = self._anchor_sl_tp_to_execution(signal=signal, signal_price=entry_price, execution_price=entry_price, entry_side=entry_side, atr=atr)
        final_md = dict(signal.metadata or {})
        final_md["entry_price"] = entry_price
        final_md["stop_loss"] = signal.stop_loss
        final_md["take_profit"] = signal.take_profit
        final_md["materialized_trade_plan"] = True
        final_md["option_trade_plan_source"] = final_md.get("option_trade_plan_source", plan_source)
        self._logger.info(
            "OPTION_TRADE_PLAN_MATERIALIZED symbol=%s entry=%.2f sl=%s tp=%s source=%s",
            signal.symbol,
            entry_price,
            signal.stop_loss,
            signal.take_profit,
            final_md.get("option_trade_plan_source"),
        )
        return dataclasses.replace(signal, metadata=final_md)

    def _handle_signal(
        self,
        signal: Signal,
        price: float,
        timestamp: datetime,
        *,
        trace_id: str | None = None,
    ) -> SignalExecutionResult:
        """
        Handle signal execution with comprehensive error handling.

        ✅ FIX: Added early time guard to prevent processing outside market hours.
        """
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX: EARLY TIME GUARD (Check BEFORE any processing)
        # ═══════════════════════════════════════════════════════════

        exchange_open = is_market_hours_cached()
        if not bool(exchange_open):
            self._logger.info(
                "RUNNER_SIGNAL_DECISION",
                extra={
                    "event": "RUNNER_SIGNAL_DECISION",
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "proceed_to_order": False,
                    "reason": "outside_market_hours",
                    "trace_id": trace_id,
                },
            )
            return SignalExecutionResult(False, "outside_market_hours", details={"trace_id": trace_id})
        mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
        is_live_mode = mode == "LIVE" or (
            str(os.getenv("ENABLE_LIVE", "false")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if is_live_mode and not bool(self._runtime_data_hard_ready):
            self._logger.info(
                "RUNNER_BLOCKED_RUNTIME_READINESS",
                extra={
                    "event": "RUNNER_BLOCKED_RUNTIME_READINESS",
                    "stage": "handle_signal",
                    "runtime_data_hard_ready": bool(self._runtime_data_hard_ready),
                    "runtime_evaluation_ready": bool(self._runtime_evaluation_ready),
                    "runtime_live_orders_armed": bool(self._runtime_live_orders_armed),
                    "runtime_readiness_reason": self._runtime_readiness_reason,
                    "trace_id": trace_id,
                },
            )
            self._emit_runner_eval_decision(
                symbol=self._normalize_symbol(signal.symbol),
                stage="handle_signal",
                reason=str(self._runtime_readiness_reason or "runtime_data_not_ready"),
                allowed=False,
                trace_id=trace_id,
                readiness={},
            )
            return SignalExecutionResult(False, "runtime_data_not_ready")
        self._logger.info(
            "RUNNER_SIGNAL_DECISION",
            extra={
                "event": "RUNNER_SIGNAL_DECISION",
                "symbol": signal.symbol,
                "action": signal.action,
                "proceed_to_order": True,
                "reason": "market_hours_open",
                "trace_id": trace_id,
            },
        )
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
                if not self._transition_execution_state(
                    base_symbol, ExecutionState.SIGNAL_RECEIVED
                ):
                    return SignalExecutionResult(
                        False,
                        "signal_state_rejected",
                        details={"trace_id": trace_id},
                    )
                return self._handle_entry_signal(
                    signal,
                    base_symbol,
                    trade_symbol,
                    trade_price,
                    timestamp,
                    trace_id=trace_id,
                )

            elif action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                self._transition_execution_state(base_symbol, ExecutionState.EXIT_PENDING)
                self._handle_exit_signal(
                    signal, base_symbol, trade_symbol, trade_price, timestamp
                )
                return SignalExecutionResult(True, "exit_submitted", details={"trace_id": trace_id})

            return SignalExecutionResult(False, "unsupported_action", details={"trace_id": trace_id, "action": action})

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
                except Exception as e:
                    LOGGER.exception(
                        "[CRITICAL] unhandled exception", exc_info=True
                    )
                    self._logger.error("Failure in _handle_signal: %s", e, exc_info=True)
            return SignalExecutionResult(
                False,
                "exception",
                details={"trace_id": trace_id, "error": str(exc)},
            )

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

                now = time.time()
                if symbol in self._orphan_retry_last_attempt:
                    if now - self._orphan_retry_last_attempt[symbol] < 10:
                        self._active_orphan_guards.discard(symbol)
                        continue
                if self._orphan_retry_count.get(symbol, 0) >= 3:
                    self._logger.error(
                        "Orphan adoption disabled after max retries: %s", symbol
                    )
                    self._active_orphan_guards.discard(symbol)
                    continue

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
                    self._orphan_retry_count.pop(symbol, None)
                    self._orphan_retry_last_attempt.pop(symbol, None)
                    adopted_count += 1
                    self._logger.info(
                        f"✅ Orphan protected: {symbol} | Bracket={bracket_id}"
                    )

                    # Try to tag the position to prevent re-adoption
                    try:
                        pos.strategy = "Adopted_Orphan"
                    except (AttributeError, TypeError):
                        pass  # Position might be frozen/immutable

                except Exception:
                    self._orphan_retry_count[symbol] = (
                        self._orphan_retry_count.get(symbol, 0) + 1
                    )
                    self._orphan_retry_last_attempt[symbol] = now
                    self._logger.exception("Failed orphan adoption")
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
        allow_off_hours = self._session_allow_out_of_hours
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
            self._logger.info(
                "RUNNER_SIGNAL_DECISION",
                extra={
                    "event": "RUNNER_SIGNAL_DECISION",
                    "symbol": order_symbol,
                "underlying_symbol": base_symbol,
                    "action": action,
                    "proceed_to_order": False,
                    "reason": "strike_selector_none",
                },
            )
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
                self._logger.info(
                    "RUNNER_SIGNAL_DECISION",
                    extra={
                        "event": "RUNNER_SIGNAL_DECISION",
                        "symbol": base_symbol,
                        "action": action,
                        "proceed_to_order": False,
                        "reason": "missing_chain_data",
                    },
                )
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
                self._logger.info(
                    "RUNNER_SIGNAL_DECISION",
                    extra={
                        "event": "RUNNER_SIGNAL_DECISION",
                        "symbol": base_symbol,
                        "action": action,
                        "proceed_to_order": False,
                        "reason": "strike_selection_none",
                    },
                )
                self._logger.warning(
                    f"⚠️ Strike Selector returned None for {base_symbol} {action} @ {price}"
                )
                return None

            return selection

        except Exception as e:
            self._logger.info(
                "RUNNER_SIGNAL_DECISION",
                extra={
                    "event": "RUNNER_SIGNAL_DECISION",
                    "symbol": base_symbol,
                    "action": action,
                    "proceed_to_order": False,
                    "reason": "strike_selection_exception",
                },
            )
            self._logger.error(
                f"💥 EXCEPTION in strike selection for {base_symbol}: {e}",
                exc_info=True,
            )
            return None

    def _passes_spot_trend_filter(self, base_symbol: str, trade_symbol: str) -> bool:
        """Validate option direction against spot VWAP trend. Args: base_symbol, trade_symbol; Returns: bool; Raises: none."""
        with self._lock:
            state = self._symbol_state.get(base_symbol)
        if (
            state is None
            or state.last_tick is None
            or state.vwap is None
            or state.vwap <= 0
        ):
            return True
        spot_price = float(
            state.last_tick.get("price") or state.last_tick.get("ltp") or 0.0
        )
        if spot_price <= 0:
            return True
        upper_symbol = trade_symbol.upper()
        if "CE" in upper_symbol:
            return spot_price > float(state.vwap)
        if "PE" in upper_symbol:
            return spot_price < float(state.vwap)
        return True

    def _handle_entry_signal(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
        *,
        trace_id: str | None = None,
    ) -> SignalExecutionResult:
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
            return SignalExecutionResult(False, "entry_lock_busy")

        try:
            return self._handle_entry_signal_inner(
                signal,
                base_symbol,
                trade_symbol,
                trade_price,
                timestamp,
                trace_id=trace_id,
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
        *,
        trace_id: str | None = None,
    ) -> SignalExecutionResult:
        """Direct order_manager entry path — no ExecutionEngine middleman.

        Args: signal, base_symbol, trade_symbol, trade_price, timestamp.
        Returns: None. Raises: Exception.
        """
        try:
            self._logger.debug(
                "Entered StrategyRunner._handle_entry_signal_inner (Direct OrderManager Flow)",
                extra={"event": "entry_signal_inner", "symbol": base_symbol},
            )

            if not self._order_manager:
                self._logger.error(
                    "ORDER_BLOCKED: order_manager is None — cannot execute entry for %s",
                    base_symbol,
                )
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "order_manager_missing")

            qty = int(signal.quantity or 0)
            if qty <= 0:
                self._logger.critical(
                    "ORDER_BLOCKED: invalid quantity=%s for %s", qty, base_symbol
                )
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "invalid_quantity")

            # Cooldown + burst guard
            now_epoch = time.time()
            underlying = self._extract_underlying(base_symbol) or base_symbol
            reason_key = str(signal.reason or "unknown")
            underlying_reason_key = f"{underlying}:{reason_key}"
            reject_cooldown_key = f"{base_symbol}:{reason_key}:score_below_threshold"
            reject_last_ts = self._signal_reject_cooldown_ts.get(reject_cooldown_key)
            if reject_last_ts is not None and (now_epoch - float(reject_last_ts)) < float(os.getenv("SIGNAL_REJECT_COOLDOWN_SECONDS", "60") or "60"):
                self._logger.info("SIGNAL_REJECT_COOLDOWN_ACTIVE symbol=%s reason=%s trace_id=%s", base_symbol, "score_below_threshold", trace_id)
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "score_below_threshold_reject_cooldown")
            if reason_key == "premium_momentum_squeeze":
                upper_symbol = (trade_symbol or base_symbol).upper()
                if ("CE" not in upper_symbol and "PE" not in upper_symbol) or "FUT" in upper_symbol:
                    log_throttled(
                        self._logger,
                        f"premium_squeeze_skipped_{upper_symbol}",
                        "PREMIUM_SQUEEZE_SKIPPED",
                        interval_sec=self._cooldown_log_throttle_seconds,
                        level=logging.DEBUG,
                        extra={"event": "PREMIUM_SQUEEZE_SKIPPED", "symbol": trade_symbol or base_symbol, "reason": "non_option_instrument"},
                    )
                    self._reset_execution_state(base_symbol)
                    return SignalExecutionResult(False, "non_option_instrument")
                last_premium_ts = float(self._premium_squeeze_last_signal_ts.get(underlying, 0.0))
                if now_epoch - last_premium_ts < self._underlying_signal_cooldown_seconds:
                    log_throttled(
                        self._logger,
                        f"premium_squeeze_suppressed_{underlying}",
                        "PREMIUM_SQUEEZE_SUPPRESSED",
                        interval_sec=self._cooldown_log_throttle_seconds,
                        level=logging.INFO,
                        extra={"event": "PREMIUM_SQUEEZE_SUPPRESSED", "underlying": underlying, "reason": "cooldown"},
                    )
                    self._reset_execution_state(base_symbol)
                    return SignalExecutionResult(False, "premium_squeeze_cooldown")
            underlying_last_ts = self._underlying_last_signal_ts.get(underlying)
            reason_last_ts = self._reason_last_signal_ts.get(underlying_reason_key)
            underlying_age = None if underlying_last_ts is None else now_epoch - float(underlying_last_ts)
            reason_age = None if reason_last_ts is None else now_epoch - float(reason_last_ts)
            self._logger.info("COOLDOWN_CHECK symbol=%s cooldown_key=%s age_seconds=%s required_seconds=%.2f allowed=%s reason=%s trace_id=%s", base_symbol, underlying_reason_key, f"{reason_age:.2f}" if reason_age is not None else None, self._reason_signal_cooldown_seconds, True if reason_age is None else reason_age >= self._reason_signal_cooldown_seconds, "first_trade_for_key" if reason_age is None else "cooldown_elapsed", trace_id)
            self._logger.info("COOLDOWN_CHECK symbol=%s cooldown_key=%s age_seconds=%s required_seconds=%.2f allowed=%s reason=%s trace_id=%s", base_symbol, underlying, f"{underlying_age:.2f}" if underlying_age is not None else None, self._underlying_signal_cooldown_seconds, True if underlying_age is None else underlying_age >= self._underlying_signal_cooldown_seconds, "first_trade_for_key" if underlying_age is None else "cooldown_elapsed", trace_id)
            if underlying_age is not None and underlying_age < self._underlying_signal_cooldown_seconds:
                self._logger.info("COOLDOWN_REJECTED reason=underlying_cooldown symbol=%s underlying=%s reason_key=%s cooldown_key=%s last_ts=%.3f now_epoch=%.3f age_seconds=%.2f required_seconds=%.2f", base_symbol, underlying, reason_key, underlying, underlying_last_ts, now_epoch, underlying_age, self._underlying_signal_cooldown_seconds)
                log_throttled(self._logger, f"runner_underlying_cd_{underlying}", "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", interval_sec=self._cooldown_log_throttle_seconds, level=logging.INFO, extra={"event": "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", "symbol": base_symbol, "reason": "underlying_cooldown"})
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "underlying_cooldown")
            if reason_age is not None and reason_age < self._reason_signal_cooldown_seconds:
                self._logger.info("COOLDOWN_REJECTED reason=reason_cooldown symbol=%s underlying=%s reason_key=%s cooldown_key=%s last_ts=%.3f now_epoch=%.3f age_seconds=%.2f required_seconds=%.2f", base_symbol, underlying, reason_key, underlying_reason_key, reason_last_ts, now_epoch, reason_age, self._reason_signal_cooldown_seconds)
                log_throttled(self._logger, f"runner_reason_cd_{reason_key}", "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", interval_sec=self._cooldown_log_throttle_seconds, level=logging.INFO, extra={"event": "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", "symbol": base_symbol, "reason": "reason_cooldown"})
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "reason_cooldown")
            while self._order_attempt_window and (now_epoch - self._order_attempt_window[0]) > 60.0:
                self._order_attempt_window.popleft()
            if len(self._order_attempt_window) >= self._max_order_attempts_per_minute:
                log_throttled(self._logger, "runner_order_attempt_rate", "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", interval_sec=300.0, level=logging.INFO, extra={"event": "RUNNER_SIGNAL_SUPPRESSED_EXECUTION_HALTED", "symbol": base_symbol, "reason": "max_order_attempts_per_minute"})
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "max_order_attempts_per_minute")
            metadata = dict(signal.metadata or {})
            quality = None
            def _trace(stop_reason: str, executor_called: bool = False, risk_allowed: bool = False) -> None:
                self._logger.info(
                    "TRADING_PATH_TRACE symbol=%s strategy_name=%s live_orders_armed=%s selected_or_near_atm=%s history_bars_effective=%s signal_generated=%s consensus_side=%s quality_final=%s quality_threshold=%s quality_allowed=%s candidate_selected=%s candidate_snapshots_present=%s risk_allowed=%s executor_called=%s stop_reason=%s",
                    signal.symbol,
                    metadata.get("strategy_name") or metadata.get("strategy") or signal.reason,
                    bool(self._runtime_live_orders_armed),
                    bool(metadata.get("candidate_selected") or metadata.get("is_selected_option")),
                    metadata.get("history_bars_effective"),
                    True,
                    infer_option_side(signal.symbol, metadata),
                    getattr(quality, "final_score", None),
                    (quality.components.get("threshold") if quality is not None else None),
                    (quality.allowed if quality is not None else None),
                    bool(metadata.get("candidate_selected") or metadata.get("is_selected_option")),
                    isinstance(metadata.get("candidate_snapshots"), list) and bool(metadata.get("candidate_snapshots")),
                    risk_allowed,
                    executor_called,
                    stop_reason,
                )
            mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
            is_live_mode = mode == "LIVE" or (
                str(os.getenv("ENABLE_LIVE", "false")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if is_live_mode and not bool(self._runtime_live_orders_armed):
                self._logger.info("ORDER_PATH_BLOCKED reason=runtime_live_orders_not_armed symbol=%s trace_id=%s", base_symbol, trace_id)
                self._logger.info(
                    "RUNNER_BLOCKED_RUNTIME_READINESS",
                    extra={
                        "event": "RUNNER_BLOCKED_RUNTIME_READINESS",
                        "stage": "entry",
                        "runtime_data_hard_ready": bool(self._runtime_data_hard_ready),
                        "runtime_evaluation_ready": bool(self._runtime_evaluation_ready),
                        "runtime_live_orders_armed": bool(self._runtime_live_orders_armed),
                        "runtime_readiness_reason": self._runtime_readiness_reason,
                        "trace_id": trace_id,
                    },
                )
                self._reset_execution_state(base_symbol)
                _trace("runtime_live_orders_not_armed")
                return SignalExecutionResult(False, "runtime_live_orders_not_armed")
            self._logger.info("ORDER_PATH_ENTERED symbol=%s reason=%s live_orders_armed=%s trace_id=%s", base_symbol, reason_key, bool(self._runtime_live_orders_armed), trace_id)
            option_side = infer_option_side(signal.symbol, metadata)
            if is_live_mode and option_side == "UNKNOWN":
                self._reset_execution_state(base_symbol)
                _trace("unknown_option_side")
                return SignalExecutionResult(False, "unknown_option_side")
            candidate_snapshots_obj = metadata.get("candidate_snapshots")
            is_directional_option = option_side in {"CE", "PE"}
            if is_live_mode and is_directional_option and not isinstance(
                candidate_snapshots_obj, list
            ):
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason="candidate_refresh_pending",
                )
            if is_live_mode and is_directional_option and not candidate_snapshots_obj:
                signal_symbol = normalize_symbol(signal.symbol)
                selected_ce = normalize_symbol(str(metadata.get("selected_ce") or self._selected_ce_symbol or ""))
                selected_pe = normalize_symbol(str(metadata.get("selected_pe") or self._selected_pe_symbol or ""))
                strike_distance = float(metadata.get("strike_distance_from_atm") or 999.0)
                selected_or_near = (
                    signal_symbol in {selected_ce, selected_pe}
                    or bool(metadata.get("is_selected_option"))
                    or strike_distance <= float(os.getenv("PREMIUM_FALLBACK_MAX_STRIKE_DISTANCE", "100") or 100)
                )
                quote_fresh = self._is_option_symbol_tick_fresh(
                    signal.symbol,
                    max_age_s=float(os.getenv("ORDER_MAX_QUOTE_AGE_MS", "60000") or "60000") / 1000.0,
                )
                sl_ok = signal.stop_loss is not None and signal.take_profit is not None
                if selected_or_near and quote_fresh and sl_ok:
                    metadata["candidate_selected"] = True
                    metadata["candidate_symbol"] = signal.symbol
                    metadata["quote_usable_for_order_plan"] = True
                    signal = dataclasses.replace(signal, metadata=metadata)
                else:
                    self._reset_execution_state(base_symbol)
                    _trace("missing_candidate_snapshots")
                    return self._reject_signal_execution(
                        symbol=base_symbol,
                        trace_id=trace_id,
                        reason="missing_candidate_snapshots",
                    )
            if isinstance(candidate_snapshots_obj, list):
                atm_strike = int(metadata.get("atm_strike") or 0)
                if atm_strike <= 0:
                    for snap in candidate_snapshots_obj:
                        if isinstance(snap, dict) and snap.get("atm_strike"):
                            atm_strike = int(snap["atm_strike"])
                            metadata["atm_strike"] = atm_strike
                            break
                if atm_strike <= 0:
                    self._reset_execution_state(base_symbol)
                    return self._reject_signal_execution(
                        symbol=base_symbol,
                        trace_id=trace_id,
                        reason="missing_atm_strike",
                    )
                try:
                    candidate = self._trade_candidate_selector.select_best_candidate(
                        underlying=underlying,
                        direction_bias=option_side,
                        atm_strike=atm_strike,
                        snapshots=[
                            snap
                            for snap in candidate_snapshots_obj
                            if isinstance(snap, dict)
                        ],
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in trade candidate selection: %s",
                        exc,
                        extra={
                            "event": "candidate_selection_error",
                            "symbol": base_symbol,
                            "trace_id": trace_id,
                        },
                        exc_info=exc,
                    )
                    self._reset_execution_state(base_symbol)
                    return self._reject_signal_execution(
                        symbol=base_symbol, trace_id=trace_id, reason="no_valid_candidate"
                    )
                if candidate is None:
                    self._reset_execution_state(base_symbol)
                    return self._reject_signal_execution(
                        symbol=base_symbol, trace_id=trace_id, reason="no_valid_candidate"
                    )
                selected_symbol = normalize_symbol(candidate.symbol)
                original_symbol = normalize_symbol(signal.symbol)
                if selected_symbol != original_symbol:
                    self._logger.info(
                        "SIGNAL_SYMBOL_REPLACED_BY_CANDIDATE original=%s selected=%s trace_id=%s",
                        signal.symbol,
                        candidate.symbol,
                        trace_id,
                        extra={
                            "event": "SIGNAL_SYMBOL_REPLACED_BY_CANDIDATE",
                            "original_symbol": signal.symbol,
                            "selected_symbol": candidate.symbol,
                            "trace_id": trace_id,
                        },
                    )
                    trade_symbol = selected_symbol
                    base_symbol = selected_symbol
                    signal = dataclasses.replace(
                        signal,
                        symbol=selected_symbol,
                        stop_loss=candidate.stop_loss or signal.stop_loss,
                        take_profit=candidate.target or signal.take_profit,
                        metadata=metadata,
                    )
                metadata["option_score"] = max(float(metadata.get("option_score", 0.0) or 0.0), float(candidate.score or 0.0))
                metadata["data_score"] = max(float(metadata.get("data_score", 0.0) or 0.0), float(candidate.data_quality_score or 0.0))
                metadata["rr_score"] = max(float(metadata.get("rr_score", 0.0) or 0.0), min(10.0, float(candidate.rr or 0.0) * 5.0))
                if str(candidate.side or "").upper() == option_side:
                    metadata["direction_score"] = max(float(metadata.get("direction_score", 0.0) or 0.0), 7.5)
                metadata["strategy_score"] = max(
                    float(metadata.get("strategy_score", 0.0) or 0.0),
                    float(metadata.get("raw_setup_score", 0.0) or 0.0),
                    float(metadata.get("setup_score", 0.0) or 0.0),
                )
                metadata["spread_pct"] = candidate.spread_pct
                metadata["candidate_score"] = candidate.score
                metadata["candidate_selected"] = True
                metadata["candidate_symbol"] = candidate.symbol
                metadata["candidate_entry_price"] = candidate.entry_price
                metadata["candidate_stop_loss"] = candidate.stop_loss
                metadata["candidate_target"] = candidate.target
                metadata["candidate_rr"] = candidate.rr
                metadata["candidate_data_quality_score"] = candidate.data_quality_score
                metadata["candidate_spread_pct"] = candidate.spread_pct
                metadata["candidate_tick_age_s"] = getattr(candidate, "tick_age_s", None)
                selected_snapshot = next(
                    (
                        snap
                        for snap in candidate_snapshots_obj
                        if isinstance(snap, dict)
                        and normalize_symbol(str(snap.get("symbol") or ""))
                        == normalize_symbol(candidate.symbol)
                    ),
                    {},
                )
                selected_snapshot = dict(selected_snapshot or {})
                metadata["selected_snapshot_symbol"] = selected_snapshot.get("symbol")

                snapshot_bid = selected_snapshot.get("bid")
                snapshot_ask = selected_snapshot.get("ask")

                try:
                    snapshot_bid_f = float(snapshot_bid) if snapshot_bid is not None else 0.0
                except (TypeError, ValueError):
                    snapshot_bid_f = 0.0

                try:
                    snapshot_ask_f = float(snapshot_ask) if snapshot_ask is not None else 0.0
                except (TypeError, ValueError):
                    snapshot_ask_f = 0.0

                snapshot_bid_ask_valid = bool(
                    snapshot_bid_f > 0
                    and snapshot_ask_f > snapshot_bid_f
                )

                snapshot_tradable_quote = bool(
                    selected_snapshot.get("tradable_quote")
                    or snapshot_bid_ask_valid
                )

                mdm_tradable_quote = False
                mdm_bid_ask_valid = False
                if self._market_data is not None:
                    try:
                        latest_snapshot = self._market_data.get_symbol_snapshot(candidate.symbol)
                        latest_bid = float(latest_snapshot.bid or 0.0)
                        latest_ask = float(latest_snapshot.ask or 0.0)
                        mdm_bid_ask_valid = bool(latest_bid > 0 and latest_ask > latest_bid)
                        mdm_tradable_quote = bool(latest_snapshot.tradable_quote and mdm_bid_ask_valid)
                        metadata["latest_quote_bid"] = latest_bid
                        metadata["latest_quote_ask"] = latest_ask
                        metadata["latest_quote_tradable"] = bool(latest_snapshot.tradable_quote)
                    except Exception as quote_exc:  # noqa: BLE001
                        self._logger.warning(
                            "QUOTE_REVALIDATION_FAILED symbol=%s err=%s trace_id=%s",
                            candidate.symbol,
                            quote_exc,
                            trace_id,
                            extra={
                                "event": "QUOTE_REVALIDATION_FAILED",
                                "symbol": candidate.symbol,
                                "trace_id": trace_id,
                                "error": str(quote_exc),
                            },
                        )

                allow_ltp_live_plan = _env_flag("ALLOW_LTP_ONLY_LIVE_ORDER_PLAN", default=False)

                metadata["tradable_quote"] = bool(snapshot_tradable_quote or mdm_tradable_quote)
                metadata["quote_usable_for_order_plan"] = bool(
                    snapshot_tradable_quote
                    or mdm_tradable_quote
                    or (
                        allow_ltp_live_plan
                        and selected_snapshot.get("ltp_only_fallback")
                        and candidate.entry_price
                        and candidate.stop_loss
                        and candidate.target
                    )
                )
            requires_final_score = bool(metadata.get("preliminary_only")) or bool(
                metadata.get("requires_runner_final_score")
            )
            quality_hint = max(
                0.0,
                min(
                    10.0,
                    float(metadata.get("confidence", 0.0) or 0.0) * 10.0,
                ),
            )
            metadata.setdefault(
                "direction_score",
                float(metadata.get("direction_quality", quality_hint) or quality_hint),
            )
            metadata.setdefault(
                "strategy_score",
                float(metadata.get("setup_quality", quality_hint) or quality_hint),
            )
            metadata.setdefault(
                "option_score",
                float(metadata.get("option_quality", 5.5) or 5.5),
            )
            metadata.setdefault(
                "data_score",
                float(metadata.get("data_quality", quality_hint) or quality_hint),
            )
            atr_for_plan = max(float(metadata.get("atr", 0.0) or 0.0), 1.0)
            try:
                signal = self._materialize_option_trade_plan(
                    signal,
                    execution_price=float(trade_price or 0.0),
                    atr=atr_for_plan,
                    entry_side=str(signal.action or "BUY"),
                )
                metadata = dict(signal.metadata or metadata)
                metadata["entry_price"] = float(trade_price or metadata.get("entry_price") or 0.0)
                metadata["stop_loss"] = signal.stop_loss
                metadata["take_profit"] = signal.take_profit
            except Exception as materialize_exc:
                self._logger.exception(
                    "TRADE_PLAN_MATERIALIZATION_FAILED symbol=%s trace_id=%s error=%s",
                    base_symbol,
                    trace_id,
                    materialize_exc,
                    extra={
                        "event": "TRADE_PLAN_MATERIALIZATION_FAILED",
                        "symbol": base_symbol,
                        "trace_id": trace_id,
                        "error_type": type(materialize_exc).__name__,
                    },
                )
                self._reset_execution_state(base_symbol)
                _trace("trade_plan_materialization_failed")
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason="trade_plan_materialization_failed",
                    details={"error": str(materialize_exc)},
                )
            if metadata.get("rr_score") is None:
                rr_score = quality_hint
                try:
                    entry_price = float(metadata.get("entry_price", 0.0) or 0.0)
                    stop_price = float(metadata.get("stop_loss", 0.0) or 0.0)
                    target_price = float(metadata.get("take_profit", 0.0) or 0.0)
                    risk = abs(entry_price - stop_price)
                    reward = abs(target_price - entry_price)
                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk
                        rr_score = max(0.0, min(10.0, rr_ratio * 5.0))
                except (TypeError, ValueError):
                    rr_score = quality_hint
                metadata["rr_score"] = rr_score
            signal_strategy = str(
                metadata.get("strategy")
                or metadata.get("strategy_name")
                or signal.reason
                or ""
            )
            current_regime = self._compute_regime_snapshot(base_symbol)
            metadata["runtime_regime"] = current_regime.value
            metadata["runtime_regime_inputs"] = self._last_regime_inputs_by_symbol.get(
                base_symbol, {}
            )
            regime_allowed, regime_reason = self._strategy_regime_decision(
                strategy=signal_strategy,
                regime=current_regime,
                symbol=base_symbol,
                metadata=metadata,
            )
            metadata["regime_decision"] = "allow" if regime_allowed else "block"
            metadata["regime_reason"] = regime_reason
            if not regime_allowed:
                self._logger.info(
                    "REGIME_GATE_REJECTED symbol=%s strategy=%s regime=%s side=%s reason=%s selected=%s spread_pct=%s candidate_rr=%s trace_id=%s",
                    base_symbol,
                    signal_strategy or "unknown",
                    current_regime.value,
                    infer_option_side(signal.symbol, metadata),
                    regime_reason,
                    bool(
                        metadata.get("candidate_selected")
                        or metadata.get("is_selected_option")
                    ),
                    metadata.get("candidate_spread_pct") or metadata.get("spread_pct"),
                    metadata.get("candidate_rr"),
                    trace_id,
                    extra={
                        "event": "REGIME_GATE_REJECTED",
                        "symbol": base_symbol,
                        "strategy": signal_strategy or "unknown",
                        "regime": current_regime.value,
                        "side": infer_option_side(signal.symbol, metadata),
                        "regime_reason": regime_reason,
                        "selected": bool(
                            metadata.get("candidate_selected")
                            or metadata.get("is_selected_option")
                        ),
                        "spread_pct": metadata.get("candidate_spread_pct")
                        or metadata.get("spread_pct"),
                        "candidate_rr": metadata.get("candidate_rr"),
                        "trace_id": trace_id,
                        "regime_inputs": self._last_regime_inputs_by_symbol.get(
                            base_symbol, {}
                        ),
                    },
                )
                self._logger.info(
                    "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s regime=%s regime_reason=%s",
                    base_symbol,
                    signal_strategy or "unknown",
                    infer_option_side(signal.symbol, metadata),
                    False,
                    "runner_regime_gate",
                    "regime_not_allowed",
                    current_regime.value,
                    regime_reason,
                    extra={
                        "event": "TRADE_DECISION_TRACE",
                        "symbol": base_symbol,
                        "strategy": signal_strategy or "unknown",
                        "side": infer_option_side(signal.symbol, metadata),
                        "allowed": False,
                        "blocked_at": "runner_regime_gate",
                        "blocked_reason": "regime_not_allowed",
                        "regime": current_regime.value,
                        "regime_reason": regime_reason,
                        "trace_id": trace_id,
                    },
                )
                self._reset_execution_state(base_symbol)
                _trace("regime_not_allowed")
                return SignalExecutionResult(False, "regime_not_allowed")
            if requires_final_score:
                required_components = (
                    "direction_score",
                    "strategy_score",
                    "option_score",
                    "data_score",
                    "rr_score",
                )
                has_components = all(
                    metadata.get(component) is not None
                    for component in required_components
                )
                has_candidate = bool(metadata.get("candidate_selected"))
                has_quote_usable = bool(metadata.get("quote_usable_for_order_plan"))
                missing_components = [
                    component
                    for component in required_components
                    if metadata.get(component) is None
                ]

                if not (has_components and has_candidate and has_quote_usable):
                    if missing_components:
                        final_score_block_reason = "missing_final_score_components"
                    elif not has_candidate:
                        final_score_block_reason = "candidate_not_selected"
                    elif not has_quote_usable:
                        final_score_block_reason = "quote_not_usable_for_order_plan"
                    else:
                        final_score_block_reason = "final_score_precheck_failed_unknown"
                    strategy_name = metadata.get("strategy_name") or metadata.get("strategy") or signal.reason
                    self._logger.info(
                        "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s missing_components=%s has_candidate=%s has_quote_usable=%s candidate_symbol=%s selected_snapshot_symbol=%s latest_bid=%s latest_ask=%s latest_quote_tradable=%s",
                        base_symbol,
                        strategy_name,
                        infer_option_side(signal.symbol, metadata),
                        False,
                        "runner_final_score_precheck",
                        final_score_block_reason,
                        missing_components,
                        has_candidate,
                        has_quote_usable,
                        metadata.get("candidate_symbol"),
                        metadata.get("selected_snapshot_symbol"),
                        metadata.get("latest_quote_bid"),
                        metadata.get("latest_quote_ask"),
                        metadata.get("latest_quote_tradable"),
                        extra={
                            "event": "TRADE_DECISION_TRACE",
                            "symbol": base_symbol,
                            "strategy": strategy_name,
                            "side": infer_option_side(signal.symbol, metadata),
                            "allowed": False,
                            "blocked_at": "runner_final_score_precheck",
                            "blocked_reason": final_score_block_reason,
                            "trace_id": trace_id,
                            "missing_components": missing_components,
                            "has_candidate": has_candidate,
                            "has_quote_usable": has_quote_usable,
                            "candidate_symbol": metadata.get("candidate_symbol"),
                            "selected_snapshot_symbol": metadata.get("selected_snapshot_symbol"),
                            "latest_bid": metadata.get("latest_quote_bid"),
                            "latest_ask": metadata.get("latest_quote_ask"),
                            "latest_quote_tradable": metadata.get("latest_quote_tradable"),
                        },
                    )
                    self._reset_execution_state(base_symbol)
                    _trace(final_score_block_reason)
                    return SignalExecutionResult(False, final_score_block_reason)
            missing_components = missing_score_components(metadata)
            if missing_components and reason_key == "premium_momentum_squeeze":
                metadata["shadow_only"] = True
                metadata["missing_reason"] = (
                    "premium_squeeze_score_components_not_implemented"
                )
            if missing_components and is_live_mode:
                self._logger.info(
                    "SIGNAL_SCORE_BLOCKED reason=missing_signal_score_components missing=%s trace_id=%s",
                    missing_components,
                    trace_id,
                    extra={
                        "event": "SIGNAL_SCORE_BLOCKED",
                        "reason": "missing_signal_score_components",
                        "missing": missing_components,
                        "trace_id": trace_id,
                    },
                )
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason="missing_signal_score_components",
                    details={"missing": missing_components},
                )
            resolved_strategy_name = (
                metadata.get("strategy_name")
                or metadata.get("strategy")
                or getattr(signal, "reason", None)
                or reason_key
            )
            quality = score_signal_quality(
                direction_score=float(metadata.get("direction_score", 0.0) or 0.0),
                strategy_score=float(metadata.get("strategy_score", 0.0) or 0.0),
                option_score=float(metadata.get("option_score", 0.0) or 0.0),
                data_score=float(metadata.get("data_score", 0.0) or 0.0),
                rr_score=float(metadata.get("rr_score", 0.0) or 0.0),
                strategy_name=str(resolved_strategy_name or ""),
            )
            final_confidence = max(0.0, min(1.0, quality.final_score / 10.0))
            self._logger.info(
                "SIGNAL_SCORE strategy_name=%s threshold=%.2f final=%.2f direction=%.2f strategy=%.2f option=%.2f data=%.2f rr=%.2f confidence=%.2f allowed=%s reasons=%s trace_id=%s",
                str(quality.components.get("strategy_name", "")),
                float(quality.components.get("threshold", 0.0) or 0.0),
                quality.final_score,
                quality.direction_score,
                quality.strategy_score,
                quality.option_score,
                quality.data_score,
                quality.rr_score,
                final_confidence,
                quality.allowed,
                quality.reasons,
                trace_id,
                extra={
                    "event": "SIGNAL_SCORE",
                    "symbol": base_symbol,
                    "trace_id": trace_id,
                    "allowed": quality.allowed,
                    "final_score": quality.final_score,
                },
            )
            if requires_final_score:
                live_threshold = float(quality.components.get("threshold", 0.0) or 0.0)
                if quality.final_score < live_threshold:
                    self._logger.info(
                        "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s final_score=%.2f threshold=%.2f direction_score=%.2f strategy_score=%.2f option_score=%.2f data_score=%.2f rr_score=%.2f reasons=%s trace_id=%s",
                        base_symbol,
                        str(quality.components.get("strategy_name", "")),
                        infer_option_side(signal.symbol, metadata),
                        False,
                        "runner_final_score",
                        "final_score_below_live_threshold",
                        quality.final_score,
                        live_threshold,
                        float(quality.components.get("direction_score", quality.direction_score) or 0.0),
                        float(quality.components.get("strategy_score", quality.strategy_score) or 0.0),
                        float(quality.components.get("option_score", quality.option_score) or 0.0),
                        float(quality.components.get("data_score", quality.data_score) or 0.0),
                        float(quality.components.get("rr_score", quality.rr_score) or 0.0),
                        quality.reasons,
                        trace_id,
                        extra={
                            "event": "TRADE_DECISION_TRACE",
                            "symbol": base_symbol,
                            "trace_id": trace_id,
                            "final_score": quality.final_score,
                            "threshold": live_threshold,
                            "allowed": False,
                            "blocked_at": "runner_final_score",
                            "blocked_reason": "final_score_below_live_threshold",
                            "direction_score": quality.components.get("direction_score", quality.direction_score),
                            "strategy_score": quality.components.get("strategy_score", quality.strategy_score),
                            "option_score": quality.components.get("option_score", quality.option_score),
                            "data_score": quality.components.get("data_score", quality.data_score),
                            "rr_score": quality.components.get("rr_score", quality.rr_score),
                            "reasons": quality.reasons,
                        },
                    )
                    self._reset_execution_state(base_symbol)
                    _trace("final_score_below_live_threshold")
                    return SignalExecutionResult(False, "final_score_below_live_threshold")
            if not quality.allowed:
                delta = quality.final_score - float(quality.components.get("threshold", 0.0) or 0.0)
                self._logger.info(
                    "SIGNAL_SCORE_REJECTED symbol=%s strategy_name=%s final=%.2f threshold=%.2f delta=%.2f components=%s",
                    base_symbol,
                    quality.components.get("strategy_name", ""),
                    quality.final_score,
                    float(quality.components.get("threshold", 0.0) or 0.0),
                    delta,
                    quality.components,
                )
                self._signal_reject_cooldown_ts[reject_cooldown_key] = now_epoch
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason="score_below_threshold",
                    details={"score": quality.final_score},
                )
            signal = dataclasses.replace(
                signal,
                confidence=final_confidence,
                metadata={
                    **metadata,
                    "final_score": quality.final_score,
                    "signal_quality": quality.components,
                },
            )

            self._logger.info(
                "SIGNAL_RECEIVED symbol=%s action=%s qty_lots=%s price=%s sl=%s tp=%s confidence=%.2f reason=%s trace_id=%s",
                signal.symbol,
                signal.action,
                signal.quantity,
                trade_price,
                signal.stop_loss,
                signal.take_profit,
                signal.confidence,
                signal.reason,
                trace_id,
            )

            lot_size = 1
            final_qty = qty
            try:
                qty_lots = max(int(signal.quantity or 1), 1)
                if hasattr(self._order_manager, "resolve_lot_size"):
                    lot_size = int(self._order_manager.resolve_lot_size(trade_symbol or base_symbol))
                final_qty = qty_lots * lot_size
                self._logger.info(
                    "ORDER_QTY_NORMALIZED symbol=%s input_qty_lots=%s lot_size=%s final_qty=%s trace_id=%s",
                    trade_symbol or base_symbol,
                    qty_lots,
                    lot_size,
                    final_qty,
                    trace_id,
                    extra={"event": "ORDER_QTY_NORMALIZED", "symbol": trade_symbol or base_symbol, "input_qty_lots": qty_lots, "lot_size": lot_size, "final_qty": final_qty},
                )
            except Exception as lot_exc:
                self._logger.warning("ORDER_BLOCKED: invalid_lot_quantity symbol=%s error=%s", trade_symbol or base_symbol, lot_exc)
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol, trace_id=trace_id, reason="invalid_lot_quantity"
                )
            qty = final_qty
            if qty <= 0 or (lot_size > 0 and qty % lot_size != 0):
                self._logger.warning("ORDER_BLOCKED: invalid_lot_quantity symbol=%s qty=%s lot_size=%s", trade_symbol or base_symbol, qty, lot_size)
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol, trace_id=trace_id, reason="invalid_lot_quantity"
                )

            # Resolve price: prefer signal metadata, fall back to live tick
            price: float | None = None
            raw_price = signal.metadata.get("signal_price") or signal.metadata.get("price")
            if raw_price:
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    price = None
            if not price or price <= 0:
                price = trade_price if trade_price > 0 else None

            # Stop-loss is mandatory for all intraday entries
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            strategy_name = str(
                signal.metadata.get("strategy_name")
                or signal.metadata.get("strategy_id")
                or signal.metadata.get("strategy")
                or "runner"
            )

            if not self._transition_execution_state(
                base_symbol, ExecutionState.ORDER_PENDING
            ):
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason="order_state_rejected",
                )

            self._logger.info(
                "RUNNER_ORDER_REQUEST symbol=%s side=%s qty=%s price=%s sl=%s tp=%s strategy=%s trace_id=%s",
                base_symbol,
                signal.action,
                qty,
                price,
                stop_loss,
                take_profit,
                strategy_name,
                trace_id,
                extra={
                    "event": "RUNNER_ORDER_REQUEST",
                    "symbol": base_symbol,
                    "trace_id": trace_id,
                },
            )

            self._order_attempt_window.append(now_epoch)
            max_quote_age_ms = int(os.getenv("ORDER_MAX_QUOTE_AGE_MS", "60000") or "60000")
            max_spread_pct = float(os.getenv("ORDER_MAX_SPREAD_PCT", os.getenv("SPREAD_MAX_PCT", "10.0")) or "10.0")
            min_depth_qty = int(float(os.getenv("ORDER_MIN_DEPTH_QTY", os.getenv("MIN_DEPTH_QTY", "0")) or 0))
            allow_market_entry = str(os.getenv("ALLOW_MARKET_ENTRY", "false")).strip().lower() in {"1", "true", "yes", "on"}
            order_symbol = trade_symbol or signal.symbol or base_symbol
            plan = TradePlan(symbol=order_symbol, side=signal.action, quantity=qty, entry_price=price, stop_loss=stop_loss, take_profit=take_profit, strategy_name=strategy_name, signal_id=signal.deterministic_id, trace_id=trace_id, tag=f"runner_{signal.action.lower()}", product="MIS", variety="regular", max_quote_age_ms=max_quote_age_ms, max_spread_pct=max_spread_pct, min_depth_qty=min_depth_qty, allow_market_entry=allow_market_entry)
            order_id = self._order_manager.submit_trade_plan(plan)

            if order_id:
                self._logger.info(
                    "ORDER_SUBMITTED order_id=%s symbol=%s side=%s qty=%s trace_id=%s",
                    order_id, base_symbol, signal.action, qty, trace_id,
                )
                self._underlying_last_signal_ts[underlying] = now_epoch
                self._reason_last_signal_ts[underlying_reason_key] = now_epoch
                if reason_key == "premium_momentum_squeeze":
                    self._premium_squeeze_last_signal_ts[underlying] = now_epoch
                try:
                    self._record_trade(
                        base_symbol,
                        TradeRecord(timestamp, signal.action, qty, price or 0.0, "submitted", signal.reason, order_id),
                    )
                except Exception as rec_exc:
                    self._logger.error("record_trade failed: %s", rec_exc)
                return SignalExecutionResult(True, "order_submitted", order_id=order_id, details={"trace_id": trace_id})
            else:
                log_throttled(self._logger, f"runner_order_rejected_{base_symbol}", "ORDER_REJECTED by order_manager", interval_sec=300.0, level=logging.WARNING, extra={"event": "ORDER_REJECTED", "symbol": base_symbol})
                self._reset_execution_state(base_symbol)
                return SignalExecutionResult(False, "order_rejected", details={"trace_id": trace_id})

        except Exception as exc:
            self._logger.error("🔴 ENTRY LOGIC CRASH: %s", exc, exc_info=True)
            self._reset_execution_state(base_symbol)
            return SignalExecutionResult(False, "entry_exception", details={"trace_id": trace_id, "error": str(exc)})

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

        _bars_source = self._data_hub or self._market_data
        bars = _bars_source.get_ohlc_bars(symbol) if _bars_source else []
        if len(bars) < self._required_candles:
            fallback_atr = max(float(current_price) * 0.015, 1.0)
            self._logger.warning(
                "ATR_FALLBACK_USED symbol=%s atr=%.4f reason=insufficient_bars bars=%d required=%d",
                symbol,
                fallback_atr,
                len(bars),
                int(self._required_candles),
                extra={"event": "ATR_FALLBACK_USED", "symbol": symbol, "atr": fallback_atr, "bars": len(bars), "required": int(self._required_candles), "reason": "insufficient_bars"},
            )
            return fallback_atr

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
            except Exception as e:
                LOGGER.exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise

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
            fallback_atr = max(float(current_price) * 0.015, 1.0)
            self._logger.warning(
                "ATR_FALLBACK_USED symbol=%s atr=%.4f reason=atr_unavailable",
                symbol,
                fallback_atr,
                extra={"event": "ATR_FALLBACK_USED", "symbol": symbol, "atr": fallback_atr, "reason": "atr_unavailable"},
            )
            return fallback_atr

        spread = 0.0
        try:
            _quote_source = self._data_hub or self._market_data
            quote = _quote_source.get_quote(symbol) if _quote_source else None
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
        """Extract underlying symbol. Args: symbol. Returns: underlying. Raises: none."""
        raw = str(symbol or "").strip().upper()
        if not raw:
            return ""
        if ":" in raw:
            _, raw = raw.split(":", 1)
        compact = "".join(raw.replace("_", " ").split())
        if compact.startswith("BANKNIFTY"):
            return "BANKNIFTY"
        if compact.startswith("FINNIFTY"):
            return "FINNIFTY"
        if compact.startswith("NIFTY"):
            return "NIFTY"
        import re
        match = re.match(r"^([A-Z]+)", compact)
        if match:
            prefix = match.group(1)
            if prefix in {"NFO", "NSE"}:
                return "NIFTY"
            return prefix
        return compact

    @staticmethod
    def _extract_strike_from_symbol(symbol: str) -> int | None:
        """Extract option strike. Args: symbol. Returns: strike or None. Raises: none."""
        raw = str(symbol or "").strip().upper()
        if not raw:
            return None
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        raw = raw.replace("_", "").replace("-", "").replace(" ", "")
        match = re.search(r"(\d{4,6})(CE|PE)$", raw)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _stale_tick_threshold_for_symbol(self, symbol: str) -> float:
        """Return configured stale threshold for symbol type."""
        return float(stale_threshold_for_symbol(symbol, is_market_open_now()))

    def _handle_exit_signal(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
    ) -> None:
        """Direct order_manager exit path — no ExecutionEngine middleman."""
        try:
            self._logger.debug(
                "Entered StrategyRunner._handle_exit_signal (Direct OrderManager Flow)",
                extra={"event": "exit_signal_inner", "symbol": base_symbol},
            )

            # ── PHASE 2: SIGNAL_RECEIVED log ────────────────────────────────
            self._logger.info(
                "SIGNAL_RECEIVED symbol=%s action=%s reason=%s",
                signal.symbol,
                signal.action,
                signal.reason,
            )

            if not self._order_manager:
                self._logger.critical(
                    "ORDER_BLOCKED: order_manager is None — cannot execute exit for %s",
                    base_symbol,
                )
                return
            now_epoch = time.time()
            underlying = self._extract_underlying(base_symbol) or base_symbol
            reason_key = str(signal.reason or "unknown")

            # Determine exit side from action
            if signal.action == "CLOSE_LONG":
                exit_side = "SELL"
            elif signal.action == "CLOSE_SHORT":
                exit_side = "BUY"
            else:
                self._logger.critical(
                    "ORDER_BLOCKED: unknown exit action=%s for %s",
                    signal.action, base_symbol,
                )
                return

            # Resolve quantity from position manager
            qty = int(signal.quantity or 0)
            if qty <= 0 and self._position_manager:
                pos = self._position_manager.get_position(base_symbol)
                if pos:
                    qty = abs(int(getattr(pos, "quantity", 0)))
            if qty <= 0:
                self._logger.critical(
                    "ORDER_BLOCKED: cannot determine exit qty for %s", base_symbol
                )
                return

            from nifty_scalper_bot.execution.order_manager import ExitIntent

            exit_intent = ExitIntent(
                symbol=base_symbol,
                qty=qty,
                product="MIS",
                exchange="NFO",
                order_type="MARKET",
                tag=f"runner_{signal.action.lower()}",
            )

            self._logger.info(
                "EXIT_TRIGGERED symbol=%s side=%s qty=%s reason=%s",
                base_symbol, exit_side, qty, signal.reason,
            )

            order_id = self._order_manager.place_reduce_only_exit(exit_intent)

            if order_id:
                self._logger.info(
                    "ORDER_SUBMITTED order_id=%s symbol=%s side=%s qty=%s (EXIT)",
                    order_id, base_symbol, exit_side, qty,
                )
                self._underlying_last_signal_ts[underlying] = now_epoch
                self._reason_last_signal_ts[f"{underlying}:{reason_key}"] = now_epoch
                self._order_attempt_window.append(now_epoch)
                try:
                    self._record_trade(
                        base_symbol,
                        TradeRecord(timestamp, signal.action, qty, trade_price, "submitted", signal.reason, order_id),
                    )
                except Exception as rec_exc:
                    self._logger.error("record_trade failed: %s", rec_exc)
            else:
                self._logger.warning(
                    "🔴 EXIT REJECTED by order_manager for %s — check ORDER_BLOCKED logs above",
                    base_symbol,
                )

        except Exception as exc:
            self._logger.error("🔴 EXIT LOGIC CRASH: %s", exc, exc_info=True)

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
            ltp = self._get_spot_price()
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
        normalized = enforce_canonical(normalize_symbol(symbol))
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
