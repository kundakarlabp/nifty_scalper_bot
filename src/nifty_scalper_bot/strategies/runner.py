"""Event-driven strategy runner coordinating trading managers."""

from __future__ import annotations

import calendar
import os
import threading
import asyncio
import time
import time as time_module
from datetime import timedelta
import logging

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
from nifty_scalper_bot.core.strategy_manager import StrategyManager
# Assumes you created the data/constants.py file as advised
from nifty_scalper_bot.data.constants import OPTION_ALIAS_SUFFIX
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType
from nifty_scalper_bot.execution.position_manager import OrderSide, PositionManager
from nifty_scalper_bot.options.strike_selector import SelectedContract, StrikeSelector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.core.message_bus import MessageBus, Message, MessageType
from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar, OneMinuteBarBuilder
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.indicators.atr_provider import ATRSnapshot
from nifty_scalper_bot.utils import metrics
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.reasons import canonical
from nifty_scalper_bot.utils.market_hours import is_market_hours_cached, get_time_status

if TYPE_CHECKING:
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.persistent_state import (
        PersistentStateManager,
        TradeDict,
    )

LOGGER = get_logger(__name__)
_THROTTLE_CACHE: Dict[str, float] = {}
_THROTTLE_LOCK = threading.Lock()

def log_throttled(logger: Any, key: str, msg: str, interval_sec: float = 60.0, level: str = "info") -> None:
    """Log a message only if 'interval_sec' has passed since the last log for 'key'."""
    with _THROTTLE_LOCK:
        now = time.time()
        last_time = _THROTTLE_CACHE.get(key, 0.0)
        if now - last_time < interval_sec:
            return
        _THROTTLE_CACHE[key] = now

    # Normalize log level (accept str or logging.* int)
    if isinstance(level, int):
        log_method = logger.log
        log_method(level, msg)
    else:
        log_method = getattr(logger, str(level).lower(), logger.info)
        log_method(msg)

    
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
    min_indicator_bars: int = 5
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


@dataclass(slots=True)
class SymbolState:
    """Mutable state maintained per symbol."""

    symbol: str
    history_limit: int
    active: bool = True
    last_tick: dict[str, Any] | None = None
    last_signal_at: datetime | None = None
    last_trade_at: datetime | None = None
    cooldown_until: datetime | None = None
    strategy_data: dict[str, Any] = field(default_factory=dict)
    vwap: float | None = None
    _last_strategy_eval: datetime | None = None # [FIX] For Throttling strategy calls
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

        # Subscribe to MessageBus if available
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
        self._symbol_state: Dict[str, SymbolState] = {}
        self._callbacks: MutableMapping[str, Callable[[dict], None]] = {}
        self._bar_builders: Dict[str, OneMinuteBarBuilder] = {}
        self._last_bar_ts: dict[str, datetime] = {}
        self._orchestrator = getattr(strategy_manager, "orchestrator", None)
        self._persistent_state: PersistentStateManager | None = None
        self._orders_in_flight: dict[str, float] = {}  # symbol -> timestamp
        self._order_in_flight_timeout: float = 30.0     # seconds
        self._entry_lock = threading.Lock()             # Atomic entry lock
        self._post_exit_cooldown_seconds: float = float(
            os.getenv("POST_EXIT_COOLDOWN_SECONDS", "60.0")
        )
    

    # ==================== LIFECYCLE MANAGEMENT ====================

    def start(self) -> None:
        """Start processing market data events."""    
        with self._lock:
            if self._running:
                return
            self._running = True
            self._trading_paused = False
            symbols = list(self._active_symbols)

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
                state = SymbolState(
                    symbol=normalized,
                    history_limit=self._config.max_trade_history,
                )
                self._symbol_state[normalized] = state
            else:
                state.active = True

            self._active_symbols.add(normalized)

        running = False
        with self._lock:
            running = self._running

        try:
            self._strategy_manager.track_symbol(normalized)
        except AttributeError:
            pass

        if running:
            self._subscribe_symbol(normalized)

        self._logger.info("Tracking symbol %s", normalized)

    def remove_symbol(self, symbol: str) -> None:
        """Stop tracking a symbol."""
        normalized = self._normalize_symbol(symbol)
        with self._lock:
            state = self._symbol_state.get(normalized)
            if state is None:
                return

            state.active = False
            self._active_symbols.discard(normalized)
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
                    str(symbol)
                    for symbol in snapshot_fn()
                    if str(symbol or "").strip()
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
                    state = SymbolState(
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
                symbol: state.snapshot()
                for symbol, state in self._symbol_state.items()
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
            self._logger.info(f"🔔 SUBSCRIBING via DataHub: {symbol}")
            self._data_hub.subscribe_ticks(symbol, callback)
            self._logger.info(f"✅ SUBSCRIBED via DataHub: {symbol}")
        else:
            self._logger.info(f"🔔 SUBSCRIBING via MarketData: {symbol}")
            self._market_data.subscribe(symbol, callback)
            self._logger.info(f"✅ SUBSCRIBED via MarketData: {symbol}")

    
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
                end=end_ts
            )
            
            # 3. Ingest
            self._ingest_bar(data["symbol"], bar, is_backfill=True)

            # 4. Force Registration
            with self._lock:
                self._active_symbols.add(data["symbol"])
                if data["symbol"] not in self._symbol_state:
                    self._symbol_state[data["symbol"]] = SymbolState(
                        symbol=data["symbol"],
                        history_limit=2000
                    )

        except Exception as exc:
            self._logger.error(f"❌ Hydration Ingest Failed for {data.get('symbol')}: {exc}")

    def mark_ready(self, symbols: list[str]) -> None:
        """
        Public API to finalize startup hydration.
        Explicitly registers symbols and sets readiness flags.
        """
        with self._lock:
            for sym in symbols:
                # 1. Register Active (Critical for main loop)
                self._active_symbols.add(sym)

                # 2. Ensure SymbolState exists (Critical for Strategy Context)
                if sym not in self._symbol_state:
                    self._symbol_state[sym] = SymbolState(
                        symbol=sym, 
                        history_limit=2000
                    )

                # 3. Initialize BarBuilder (Prevent KeyErrors in internal checks)
                if sym not in self._bar_builders:
                    self._bar_builders[sym] = OneMinuteBarBuilder()

                # 4. Set High-Water Mark (Prevent dropping first live tick)
                if sym not in self._last_bar_ts:
                    self._last_bar_ts[sym] = datetime.now(timezone.utc)

        # 5. THE KILL SWITCH: Prevents fallback backfill logic from running
        self._startup_hydrated = True
        
        self._logger.info(f"✅ StrategyRunner marked READY with {len(symbols)} symbols")
            
            
    def _ingest_bar(self, symbol: str, bar: OneMinuteBar, is_backfill: bool = False) -> None:
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
                extra={"symbol": symbol, "bar_ts": bar.timestamp, "last_ts": last_ts}
            )
            return

        # 2. STATE: Update High-Water Mark
        if not last_ts or bar.timestamp > last_ts:
            self._last_bar_ts[symbol] = bar.timestamp

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
                    timestamp=bar.timestamp
                )

            # 4. BRACKET MANAGER: Inject Dynamic ATR (Volatility)
            if self._bracket_manager:
                # Compute ATR (Period 14 is standard)
                raw_atr = self._indicator_engine.compute_atr(symbol, period=14)
                
                # Robust Unwrapping
                atr_value = 0.0
                if isinstance(raw_atr, (int, float)):
                    atr_value = float(raw_atr)
                elif hasattr(raw_atr, 'value'):
                    atr_value = float(raw_atr.value)
                elif hasattr(raw_atr, 'atr'):
                    atr_value = float(raw_atr.atr)

                if atr_value > 0 and hasattr(self._bracket_manager, "update_market_stats"):
                    self._bracket_manager.update_market_stats(symbol, atr=atr_value)

            # [FIX] Force Regime Refresh: Ensure Detector sees the new bar immediately
            if hasattr(self, "_strategy_manager"):
                regime_mgr = getattr(self._strategy_manager, "regime_manager", None)
                if regime_mgr and hasattr(regime_mgr, "refresh_from_indicators"):
                    # Use the captured main loop to run the async refresh safely from this thread
                    if self._main_loop and self._main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            regime_mgr.refresh_from_indicators(),
                            self._main_loop
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
                        "volume": bar.volume
                    }
                    
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

    def aggregate_signals_by_symbol(
        self,
        signals: list[Signal],
    ) -> dict[str, Signal]:
        """Aggregate strategy signals by symbol to limit duplicated risk."""
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

                avg_confidence = sum(sig.confidence for sig in symbol_signals) / len(
                    symbol_signals
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

                best_signal = max(symbol_signals, key=lambda sig: sig.confidence)
                metadata = dict(best_signal.metadata)
                metadata["aggregated_count"] = len(symbol_signals)
                metadata["aggregated_sources"] = [
                    {
                        "strategy": sig.metadata.get("strategy"),
                        "confidence": sig.confidence,
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
                    _NIFTY_OPTION_SUCCESS_RATE.labels(
                        underlying=underlying_label
                    ).set(counters["success"] / total)

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
                    _NIFTY_OPTION_SUCCESS_RATE.labels(
                        underlying=underlying_label
                    ).set(counters["success"] / total)

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
                s for s, t in self._orders_in_flight.items()
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
            
            if underlying and underlying != symbol and underlying in self._orders_in_flight:
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
                f"📌 MARKED IN-FLIGHT: {symbol}" + 
                (f" (underlying: {underlying})" if underlying else "")
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
                        "cooldown_seconds": self._post_exit_cooldown_seconds
                    }
                )


    async def _handle_tick_message(self, message: Message) -> None:
        """Process incoming TICK messages from the MessageBus."""
        # [MODIFIED] Using defined helper correctly
        log_throttled(
            self._logger,
            "msg_bus_tick",
            f"🔔 MESSAGE BUS TICK: type={message.type}",
            interval_sec=60.0
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
        """
        Check if market is currently open (09:15 - 15:30 IST).
        Handles timezone conversion robustly.
        """
        try:
            # 1. Allow Override for Testing/Session Extension
            # Checks environment variable to bypass time restrictions
            if os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "false").lower() == "true":
                return True

            # 2. Define IST Timezone (UTC+5:30)
            ist_offset = timedelta(hours=5, minutes=30)
            ist_tz = timezone(ist_offset)
            
            # 3. Ensure 'now' is Timezone Aware
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            
            # 4. Convert to IST
            now_ist = now.astimezone(ist_tz)
            
            # 5. Check Weekend (Saturday=5, Sunday=6)
            if now_ist.weekday() >= 5: 
                return False
                
            # 6. Check Time Boundaries (09:15 to 15:30)
            t = now_ist.time()
            start = time(9, 15)
            end = time(15, 30)
            
            return start <= t <= end
            
        except Exception as e:
            # Fail safe: If check crashes, defaulting to True prevents locking the bot
            # (Risk is managed elsewhere)
            self._logger.warning(f"Market time check failed: {e}. Defaulting to OPEN.")
            return True

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
                self._logger.info("⏭️ Skipping StrategyRunner historical backfill (startup hydration already completed)")
                return

            # 2. FALLBACK: Only runs if App.py failed
            self._logger.warning("⚠️ StrategyRunner memory is empty! Triggering fallback backfill...")
            
            with self._lock:
                targets = list(self._active_symbols)
            
            if not targets:
                self._logger.warning("⚠️ Backfill skipped: No active symbols found.")
                return

            # Determine Data Source
            source = None
            if hasattr(self, "_data_hub") and self._data_hub and hasattr(self._data_hub, "fetch_history"):
                source = self._data_hub
            elif hasattr(self, "_orchestrator") and self._orchestrator:
                source = self._orchestrator

            if not source:
                return

            # 3. SEQUENTIAL FETCH
            for symbol in targets:
                try:
                    # [FIX] Added interval="minute" to fix TypeError
                    history = await source.fetch_history(symbol, interval="minute", days=5)
                    
                    if history:
                        for bar_data in history:
                            self.ingest_historical_bar(bar_data)
                            total_bars += 1
                        self._logger.info(f"✅ Fallback backfill: Ingested {len(history)} bars for {symbol}")
                    
                    await asyncio.sleep(0.5) 

                except Exception as e:
                    self._logger.error(f"❌ Fallback fetch failed for {symbol}: {e}")

        except Exception as exc:
             self._logger.error(f"❌ History backfill crashed: {exc}", exc_info=True)
        
        if total_bars > 0:
            self._logger.info(f"✅ Emergency Backfill complete. Ingested {total_bars} bars.")

    
    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """
        Handle incoming tick safely, updating state and triggering strategies.
        Includes robust data extraction, validation, and multi-tier strategy execution.
        """
        now = datetime.now(timezone.utc)
        
        if "FUT" in symbol.upper():
            return 
        
        # =================================================================
        # PHASE 0: EARLY EXIT CHECKS (Fast path for non-trading scenarios)
        # =================================================================
        
        # 1. Orphan Guard (Logic we added previously)
        if self._position_manager:
            active_pos = self._position_manager.get_active_contract(symbol)
            if active_pos:
                strat = getattr(active_pos, "strategy", "") or "unknown"
                if "manual" in strat.lower() or "unknown" in strat.lower():
                    log_throttled(
                        self._logger,
                        f"orphan_guard_{symbol}",
                        f"🛡️ ORPHAN GUARD: {symbol} is unmanaged. Attempting adoption...",
                        interval_sec=30.0,
                        level=logging.WARNING
                    )
                    # Try to adopt (ensure self._adopt_orphan_positions exists)
                    if hasattr(self, "_adopt_orphan_positions"):
                        self._adopt_orphan_positions()
                    return 

        # 2. Market Time Check (Now 'now' is valid!)
        if not self._is_market_open(now):
            return

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
            for k in keys:
                if d.get(k) is not None: 
                    try:
                        return float(d[k])
                    except (ValueError, TypeError):
                        continue
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
        broker_vwap = _extract_float(tick, "average_price", "vwap")
        volume = _extract_int(tick, "volume", "volume_traded")
        source = tick.get("source", "unknown")

        # =================================================================
        # PHASE 2: DATA VALIDATION
        # =================================================================
        
        # Stale tick check (increased threshold for REST polling to prevent false positives)
        stale_threshold = 30.0 if source in ("rest", "polling") else 10.0
        
        if tick_age > stale_threshold:
            log_throttled(
                self._logger, 
                f"stale_tick_{symbol}",
                f"⏰ STALE TICK: {symbol} ({tick_age:.1f}s old, threshold={stale_threshold}s)",
                interval_sec=30.0, 
                level=logging.WARNING
            )
            return

        # Price validity check
        if price <= 0:
            log_throttled(
                self._logger,
                f"invalid_price_{symbol}",
                f"⚠️ Invalid price ({price}) for {symbol}, skipping",
                interval_sec=60.0,
                level=logging.WARNING
            )
            return

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
            level=logging.INFO
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
                level=logging.INFO
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
        
        # Only check risk if not in warmup
        if not in_warmup:
            try:
                is_allowed = False
                rm = self._risk_manager
                
                if rm:
                    if hasattr(rm, "can_trade"):
                        is_allowed = rm.can_trade(symbol)
                    elif hasattr(rm, "risk_gate_should_trade"):
                        res = rm.risk_gate_should_trade()
                        is_allowed = res[0] if isinstance(res, tuple) else bool(res)
                    elif hasattr(rm, "can_trade_now"):
                        res = rm.can_trade_now()
                        is_allowed = res[0] if isinstance(res, tuple) else bool(res)
                    else:
                        is_allowed = False
                else:
                    # No risk manager = allow trading
                    is_allowed = True

                if not is_allowed:
                    log_throttled(
                        self._logger,
                        f"risk_block_{symbol}", 
                        f"⛔ Risk Block Active: {symbol}. Trading Halted.",
                        interval_sec=30.0,
                        level=logging.WARNING
                    )
                    return

            except Exception as e:
                self._logger.error(f"Critical error in risk check for {symbol}: {e}")
                return

        # =================================================================
        # PHASE 7: STRATEGY PREPARATION (Skip during warmup)
        # =================================================================
        
        if in_warmup:
            return 
        
        with self._lock:
            # Auto-track new symbols
            if symbol not in self._active_symbols:
                self._logger.info(f"🆕 Auto-tracking symbol from feed: {symbol}")
                self._active_symbols.add(symbol)
                self._symbol_state[symbol] = SymbolState(
                    symbol=symbol, 
                    history_limit=self._config.max_trade_history
                )

            state = self._symbol_state.get(symbol)
            if state is None or not state.active:
                return

            # Update VWAP from broker
            if broker_vwap and broker_vwap > 0:
                state.vwap = broker_vwap
            
            # Heartbeat logging for derivatives (confirms data flow)
            if "NIFTY" in symbol and any(x in symbol for x in ["FUT", "CE", "PE"]):
                log_throttled(
                    self._logger,
                    f"heartbeat_{symbol}",
                    f"💓 TICK HEARTBEAT: {symbol} | LTP={price:.2f} | VWAP={state.vwap or 0:.2f}",
                    interval_sec=30.0,
                    level=logging.INFO
                )

            # =============================================================
            # PHASE 8: SIGNAL GENERATION
            # =============================================================
            generated_signal = None

            # 8A. FORCED SIGNAL (Testing only)
            force_signal_enabled = os.getenv("FORCE_SIGNAL", "").lower() == "true"
            disable_early_forced = os.getenv("FEATURE_DISABLE_EARLY_FORCED_SIGNALS", "").lower() == "true"
            
            if force_signal_enabled and not disable_early_forced:
                generated_signal = Signal(
                    action="BUY", symbol=symbol, quantity=1, confidence=1.0,
                    reason="forced_signal_validation", stop_loss=None, take_profit=None,
                    metadata={"source": "forced"}
                )
                self._logger.warning(f"⚠️ FORCED SIGNAL EMITTED for {symbol}")

            # 8B. PRIMARY STRATEGY: VWAP Crossover (Requires VWAP > 0)
            vwap_crossover_enabled = os.getenv("ENABLE_VWAP_CROSSOVER", "false").lower() == "true"
            
            if (vwap_crossover_enabled
                and generated_signal is None 
                and state.vwap 
                and state.vwap > 0 
                and "FUT" not in symbol.upper()):
                prev_ltp = _extract_float(state.last_tick, "ltp", "last_price") if state.last_tick else None
                curr_vwap = state.vwap

                if prev_ltp and curr_vwap and price > 0:
                    threshold = curr_vwap * 0.0005  # 0.05% buffer
                    is_cross_up = (prev_ltp < (curr_vwap + threshold) and price > (curr_vwap + threshold))
                    is_cross_down = (prev_ltp > (curr_vwap - threshold) and price < (curr_vwap - threshold))
                    
                    # ✅ FIX: Calculate proper stop_loss and take_profit
                    sl_pct = float(os.getenv("VWAP_SL_PCT", "1.5"))  # 1.5% SL
                    tp_pct = float(os.getenv("VWAP_TP_PCT", "2.0"))  # 2.0% TP (1:1.33 RR)
                    
                    if is_cross_up:
                        # BUY signal - SL below, TP above
                        calculated_sl = price * (1 - sl_pct / 100)
                        calculated_tp = price * (1 + tp_pct / 100)
                        
                        self._logger.info(
                            f"⚡ VWAP CROSSOVER UP: {symbol} | {prev_ltp:.2f} -> {price:.2f} (VWAP: {curr_vwap:.2f})",
                            extra={"event": "vwap_crossover", "symbol": symbol}
                        )
                        generated_signal = Signal(
                            action="BUY", symbol=symbol, quantity=1, confidence=0.75,
                            reason="vwap_crossover_up", 
                            stop_loss=calculated_sl,      # ✅ NOW HAS PROPER SL
                            take_profit=calculated_tp,    # ✅ NOW HAS PROPER TP
                            metadata={
                                "strategy": "vwap_scalp", 
                                "vwap": curr_vwap, 
                                "tag": "vwap_scalp",
                                "sl_pct": sl_pct,
                                "tp_pct": tp_pct
                            }
                        )
                    elif is_cross_down:
                        # SELL signal - SL above, TP below
                        calculated_sl = price * (1 + sl_pct / 100)
                        calculated_tp = price * (1 - tp_pct / 100)
                        
                        self._logger.info(
                            f"⚡ VWAP CROSSOVER DOWN: {symbol} | {prev_ltp:.2f} -> {price:.2f} (VWAP: {curr_vwap:.2f})",
                            extra={"event": "vwap_crossover", "symbol": symbol}
                        )
                        generated_signal = Signal(
                            action="SELL", symbol=symbol, quantity=1, confidence=0.75,
                            reason="vwap_crossover_down", 
                            stop_loss=calculated_sl,      # ✅ NOW HAS PROPER SL
                            take_profit=calculated_tp,    # ✅ NOW HAS PROPER TP
                            metadata={
                                "strategy": "vwap_scalp", 
                                "vwap": curr_vwap, 
                                "tag": "vwap_scalp_short",
                                "sl_pct": sl_pct,
                                "tp_pct": tp_pct
                            }
                        )

            # 8C. FALLBACK STRATEGY: Momentum Breakout (When VWAP is Missing/0)
            if generated_signal is None and (not state.vwap or state.vwap == 0):
                prev_ltp = _extract_float(state.last_tick, "ltp", "last_price") if state.last_tick else None
                
                if prev_ltp and prev_ltp > 0 and price > 0:
                    price_change_pct = ((price - prev_ltp) / prev_ltp) * 100
                    MOMENTUM_THRESHOLD_PCT = 0.15
                    
                    # ✅ FIX: Calculate proper stop_loss for momentum signals too
                    sl_pct = float(os.getenv("MOMENTUM_SL_PCT", "2.0"))
                    tp_pct = float(os.getenv("MOMENTUM_TP_PCT", "2.5"))
                    
                    if price_change_pct > MOMENTUM_THRESHOLD_PCT:
                        calculated_sl = price * (1 - sl_pct / 100)
                        calculated_tp = price * (1 + tp_pct / 100)
                        
                        self._logger.info(
                            f"🚀 MOMENTUM FALLBACK BUY: {symbol} | Change={price_change_pct:.3f}% (VWAP=0)",
                            extra={"event": "momentum_fallback", "symbol": symbol}
                        )
                        generated_signal = Signal(
                            action="BUY", symbol=symbol, quantity=1, confidence=0.60,
                            reason="momentum_breakout_up", 
                            stop_loss=calculated_sl,      # ✅ NOW HAS PROPER SL
                            take_profit=calculated_tp,    # ✅ NOW HAS PROPER TP
                            metadata={
                                "strategy": "momentum_fallback", 
                                "price_change_pct": price_change_pct, 
                                "tag": "fallback_long"
                            }
                        )
                    elif price_change_pct < -MOMENTUM_THRESHOLD_PCT:
                        calculated_sl = price * (1 + sl_pct / 100)
                        calculated_tp = price * (1 - tp_pct / 100)
                        
                        self._logger.info(
                            f"🔻 MOMENTUM FALLBACK SELL: {symbol} | Change={price_change_pct:.3f}% (VWAP=0)",
                            extra={"event": "momentum_fallback", "symbol": symbol}
                        )
                        generated_signal = Signal(
                            action="SELL", symbol=symbol, quantity=1, confidence=0.60,
                            reason="momentum_breakout_down", 
                            stop_loss=calculated_sl,      # ✅ NOW HAS PROPER SL
                            take_profit=calculated_tp,    # ✅ NOW HAS PROPER TP
                            metadata={
                                "strategy": "momentum_fallback", 
                                "price_change_pct": price_change_pct, 
                                "tag": "fallback_short"
                            }
                        )

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

        # If no immediate signal, delegate to complex StrategyManager
        if signal is None and self._config.min_indicator_bars:
            should_evaluate = False
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state:
                    last_eval = getattr(state, "_last_strategy_eval", None)
                    # Limit evaluation frequency (max 2 per second)
                    if last_eval and (now - last_eval).total_seconds() < 0.5:
                        return 
                    state._last_strategy_eval = now
                    should_evaluate = True
            
            if should_evaluate:
                # ✅ DIAGNOSTIC LOG: Confirm evaluation is happening
                log_throttled(
                    self._logger,
                    f"strategy_eval_{symbol}",
                    f"🎯 EVALUATING STRATEGIES: {symbol} | min_bars={self._config.min_indicator_bars}",
                    interval_sec=30.0,
                    level=logging.INFO
                )

                is_ready = self._indicator_engine.is_ready(symbol, self._config.min_indicator_bars)
                
                if is_ready:
                    # ✅ DIAGNOSTIC LOG: Confirm indicators are ready
                    log_throttled(
                         self._logger,
                         f"indicators_ready_{symbol}",
                         f"✅ INDICATORS READY: {symbol} | Calling StrategyManager...",
                         interval_sec=60.0,
                         level=logging.INFO
                    )
                    
                    signal = self._strategy_manager.generate_signal(symbol, price)
                    if signal is None:
                        log_throttled(
                            self._logger,
                            f"no_signal_manager_{symbol}",
                            f"📉 Strategy Manager evaluated {symbol}: NO SIGNAL",
                            interval_sec=30.0,
                            level=logging.INFO
                        )
                else:
                    # ✅ DIAGNOSTIC LOG: Explain why no evaluation happened
                    log_throttled(
                        self._logger, 
                        f"not_ready_{symbol}", 
                        f"⏳ INDICATORS NOT READY: {symbol} (Need {self._config.min_indicator_bars} bars)", 
                        interval_sec=30.0,
                        level=logging.WARNING
                    )

        # =================================================================
        # PHASE 10: EXECUTE SIGNAL
        # =================================================================
        
        if signal and signal.action != "HOLD":
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
                        "timestamp": now.isoformat()
                    }

            self._logger.info(f"🚀 SIGNAL EXECUTING: {symbol} | Action={signal.action} | Reason={signal.reason}")
            self._handle_signal(signal, price, now)
            
            
    def _handle_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """
        Handle signal execution with comprehensive error handling.
        
        ✅ FIX: Added early time guard to prevent processing outside market hours.
        """
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX: EARLY TIME GUARD (Check BEFORE any processing)
        # ═══════════════════════════════════════════════════════════
        from nifty_scalper_bot.utils.market_hours import is_market_hours_cached, get_time_status
        
        if not is_market_hours_cached():
            # Throttle logging to once per minute per symbol
            cache_key = f"time_block_{signal.symbol}"
            if not hasattr(self, '_time_block_logged'):
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
            self._logger.error(
                f"🔴 HANDLER CRASHED: {exc}", exc_info=True
            )
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
                    
                # Check strategy tag
                strategy = (
                    getattr(pos, "strategy", "") or 
                    getattr(pos, "strategy_name", "") or 
                    getattr(pos, "tag", "") or 
                    ""
                )
                
                # Identify Orphan (Manual/Unknown/Empty)
                is_orphan = strategy.lower().strip() in ("manual", "unknown", "manual/unknown", "", "none")
                
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
                    getattr(pos, "entry_price", 0) or 
                    getattr(pos, "avg_price", 0) or 
                    getattr(pos, "average_price", 0) or 
                    0
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
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        entry_price=entry
                    )
                    adopted_count += 1
                    self._logger.info(f"✅ Orphan protected: {symbol} | Bracket={bracket_id}")
                    
                    # Try to tag the position to prevent re-adoption
                    try:
                        pos.strategy = "Adopted_Orphan"
                    except (AttributeError, TypeError):
                        pass  # Position might be frozen/immutable
                        
                except Exception as e:
                    self._logger.error(f"❌ Failed to adopt orphan {symbol}: {e}")
                    
            except Exception as e:
                self._logger.error(f"❌ Error processing position: {e}")
        
        if adopted_count > 0:
            self._logger.info(f"📊 Orphan Adoption Complete: {adopted_count} positions protected")

    def _calculate_signal_score(self, symbol: str, side: str, price: float) -> float:
        """
        Calculate confidence using INSTANT metrics (No history required).
        
        ✅ WORLD CLASS FIX: Better handling of market hours and volume.
        """
        import os
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
                vol = float(state.last_tick.get('volume', 0))
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
        self, 
        base_symbol: str, 
        action: str, 
        price: float,
        option_type: str | None
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
                level=logging.CRITICAL
            )
            return None

        # 2. GUARD: Check if we have actual chain data (Prevents selecting from empty chain)
        # We rely on DataHub to tell us if the chain is alive.
        if self._data_hub:
             if hasattr(self._data_hub, "has_chain_data") and not self._data_hub.has_chain_data(base_symbol):
                 log_throttled(
                    self._logger,
                    f"missing_chain_{base_symbol}",
                    f"🛑 MISSING CHAIN DATA: Cannot select strike for {base_symbol}. DataHub returned no chain.",
                    interval_sec=30.0,
                    level=logging.ERROR
                )
                 return None

        try:
            # 3. EXECUTE: Safe selection
            # Map action to selector side
            selector_side = "BUY" if action == "BUY" else "SELL"
            safe_opt_type = cast(Literal['CE', 'PE'], option_type) if option_type in ('CE', 'PE') else None
            
            selection = self._strike_selector.select_contract(
                underlying=base_symbol, 
                side=selector_side,
                underlying_price=price, 
                option_type=safe_opt_type,
            )
            
            if not selection:
                self._logger.warning(f"⚠️ Strike Selector returned None for {base_symbol} {action} @ {price}")
                return None
                
            return selection

        except Exception as e:
            self._logger.error(f"💥 EXCEPTION in strike selection for {base_symbol}: {e}", exc_info=True)
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
                f"🛡️ ENTRY LOCK BUSY: {base_symbol} | "
                "Another entry being processed",
                extra={"event": "entry_lock_busy", "symbol": base_symbol}
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
        """Inner implementation of entry signal handling (lock already held)."""
        
        # ═══════════════════════════════════════════════════════════════
        # 🛡️ GUARD 0.5: ORDER IN-FLIGHT CHECK
        # Prevents duplicate submissions before order fills
        # ═══════════════════════════════════════════════════════════════
        if self._is_order_in_flight(trade_symbol or base_symbol, base_symbol):
            self._logger.info(
                f"🛡️ ORDER IN-FLIGHT REJECT: {base_symbol} | "
                "Waiting for pending order to complete",
                extra={"event": "order_in_flight_reject", "symbol": base_symbol}
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
                        extra={"event": "signal_debounce_reject", "symbol": base_symbol}
                    )
                    return

        # -----------------------------------------------------------
        # 🛡️ GUARD 2: Position Check (No Pyramiding)
        # -----------------------------------------------------------
        if self._position_manager:
            active_contract = self._position_manager.get_active_contract(base_symbol)
            if active_contract and not self._risk_manager.settings.allow_pyramiding:
                self._logger.info(
                    f"🛡️ PYRAMID REJECT: {base_symbol} | "
                    f"Already active on {active_contract.symbol} | "
                    f"Pyramiding Disabled",
                    extra={"event": "signal_pyramid_reject", "symbol": base_symbol}
                )
                 # Update signal timer to prevent log spam
                with self._lock:
                     if state: state.last_signal_at = timestamp
                return

        side = "LONG" if signal.action == "BUY" else "SHORT"
        confidence = self._calculate_signal_score(signal.symbol, side, trade_price)

        # Confidence Threshold
        import os
        min_confidence = float(os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.45"))
        
        if confidence < min_confidence:
            self._logger.info(
                f"🚫 CONFIDENCE REJECT: {base_symbol} | "
                f"Score={confidence:.2f} < min={min_confidence:.2f}",
                extra={"event": "signal_confidence_reject", "symbol": base_symbol}
            )
            return
        action = signal.action

        # ===========================================================
        # ✅ SMART VWAP FILTER (Unshackles Elite Strategies)
        # ===========================================================
        should_check_vwap = True
        if signal.metadata and signal.metadata.get("ignore_vwap"):
            should_check_vwap = False
            self._logger.debug(f"ℹ️ VWAP Check Bypassed by Strategy: {signal.strategy_name}")

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
                    extra={"event": "signal_vwap_reject", "symbol": base_symbol}
                )
                self._record_trade(
                    base_symbol, 
                    TradeRecord(
                        timestamp, action, signal.quantity, trade_price, 
                        "skipped", "vwap_filter"
                    )
                )
                return
            
            self._logger.info(f"📊 VWAP PASS: Price={trade_price:.2f} > VWAP={current_vwap:.2f} Dist={vwap_dist:.2f}%")

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

            sell_premium = bool(metadata.get("sell_premium")) and not self._options_long_only
            entry_side: OrderSide = "SELL" if sell_premium else "BUY"

            # Check active contract reuse logic (Scaling In)
            if self._position_manager:
                active = self._position_manager.get_active_contract(base_symbol)
                if active:
                    if self._position_manager.is_flat(active.symbol):
                        self._position_manager.clear_active_contract(base_symbol)
                    else:
                        reuse = True
                        if active.option_type != option_type and not self._allow_hedge_entries:
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
            if not selection and (base_symbol.endswith("CE") or base_symbol.endswith("PE")):
                selection = SelectedContract(
                    symbol=base_symbol,
                    option_type="CE" if "CE" in base_symbol else "PE",
                    strike=0.0, expiry=timestamp, ltp=trade_price, delta=None,
                    metadata={"source": "explicit"}
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
                    option_type=option_type
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
                        safe_price = _extract_float(q, "ltp", "last_price", "close") if q else 0.0
                        
                        if safe_price > 0:
                            trade_price = safe_price
                            self._logger.info(f"🔄 Fetched fresh price for {trade_symbol}: {trade_price}")
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
                    extra={"event": "signal_contract_reject", "symbol": base_symbol}
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
                symbol=trade_symbol,
                metadata=metadata,
                current_price=trade_price
            )
            
            self._logger.info(
                f"📊 SIZING: {trade_symbol} | Price={trade_price:.2f} | "
                f"SL={signal.stop_loss:.2f} | ATR={atr_val:.2f}",
                extra={
                    "event": "sizing_calculation",
                    "symbol": trade_symbol,
                    "price": trade_price,
                    "stop_loss": signal.stop_loss,
                    "atr": atr_val
                }
            )
            
            sized_qty = self._risk_manager.suggest_position_size(
                side=entry_side, price=trade_price, stop_loss=signal.stop_loss,
                atr=atr_val, requested_quantity=signal.quantity,
                confidence=signal.confidence, symbol=trade_symbol,
            )

            if sized_qty <= 0:
                self._logger.warning(f"🔴 Risk Manager returned 0 qty")
                return

            # Validate Position Limits
            allowed, reason = self._risk_manager.validate_new_position(
                symbol=trade_symbol, side="LONG" if entry_side == "BUY" else "SHORT",
                quantity=int(sized_qty), entry_price=trade_price,
                stop_loss=signal.stop_loss, take_profit=signal.take_profit,
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

            self._logger.info(f"🟡 SUBMITTING ORDER: {trade_symbol} Qty: {sized_qty} Limit: {execution_price}")
            
            strat_name = signal.metadata.get("strategy", "MAN") if signal.metadata else "MAN"
            unique_tag = f"{strat_name[:3]}_{int(timestamp.timestamp())}"

            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                quantity=int(sized_qty),
                order_type=OrderType.LIMIT,
                price=execution_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_id=unique_tag,
                tag=unique_tag
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
                        if opt_state: opt_state.last_signal_at = timestamp

            if order_id:
                self._logger.info(f"🟢 ORDER SUBMITTED! ID: {order_id}")
                
                # Async Verification & Chase Logic
                if self._main_loop and self._main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._verify_order_status(order_id, trade_symbol, 3.0),
                        self._main_loop
                    )

                self._notify_orchestrator_submission(signal, base_symbol)
                
                # Update Active Contract Tracking
                if self._position_manager and selection:
                    if self._allow_hedge_entries or not self._position_manager.get_active_contract(base_symbol):
                        self._position_manager.set_active_contract(base_symbol, selection)

                if selector:
                    selector.register_open(base_symbol, selection)

                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp, action, int(sized_qty), trade_price,
                        "submitted", signal.reason, order_id,
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
                
                if not order: return

                status = str(order.status).upper()
                # ═══════════════════════════════════════════════════════
                if status in ["COMPLETE", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"]:
                    self._clear_order_in_flight(symbol)
                
                # 🛡️ ACTIVE CHASE LOGIC
                # If Limit Order is ignored by market (OPEN) after 3s, we must act.
                if status in ["OPEN", "PENDING", "SUBMITTED"]:
                    self._logger.warning(f"⏳ ORDER {order_id} STUCK ({status}). Initiating Chase...")
                    
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
                        self._logger.info(f"🏃 CHASING: Modifying {order_id} to {new_price}")
                        await asyncio.to_thread(
                            self._order_manager.modify_order, 
                            order_id=order_id, 
                            price=new_price
                        )
                    else:
                        self._logger.error("❌ Could not get fresh price for Chase.")

                elif status == "COMPLETE":
                    self._logger.info(f"✅ ORDER {order_id} FILLED.")
                

                    
        except Exception as exc:
            self._logger.warning(f"Order verification/chase warning: {exc}")

    def _get_atr_with_fallback(
        self,
        symbol: str,
        metadata: dict,
        current_price: float
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
        
        # 5. Fallback: Calculate from price
        # For NIFTY options, typical ATR is ~1-2% of premium
        if atr_val <= 0:
            atr_val = current_price * 0.015  # 1.5% of price
            source = "price_fallback"
            
            self._logger.warning(
                f"⚠️ ATR unavailable for {symbol}, using price-based estimate: {atr_val:.2f}",
                extra={"event": "atr_fallback", "symbol": symbol, "atr": atr_val}
            )
        
        self._logger.debug(
            f"ATR resolved: {symbol} = {atr_val:.2f} (source: {source})",
            extra={"event": "atr_resolved", "symbol": symbol, "atr": atr_val, "source": source}
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
        match = re.match(r'^([A-Z]+)', symbol)
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
            position = position_manager.get_position(trade_symbol) if position_manager else None

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
        spot = 26000.0 # Default fallback
        
        # ✅ FIX: Try both attribute names to be safe
        mdm = self._market_data
        
        if mdm:
            # Try getting LTP
            ltp = mdm.get_latest_price("NSE:NIFTY 50")
            if ltp and ltp > 0: 
                spot = ltp

        for pos in self._position_manager.get_all_positions():
            if pos.quantity == 0 or "NIFTY" not in pos.symbol: continue
            
            try:
                # Extract Strike & Type from Symbol (e.g. NIFTY25DEC26000CE)
                import re
                match = re.search(r'(\d{5})([CP]E)', pos.symbol)
                if not match: continue
                
                strike = float(match.group(1))
                opt_type = match.group(2)
                
                # Dynamic Time to Expiry (Target: 15:30 on Expiry Day)
                # Simplified: 1 day to expiry
                t_years = 1.0 / 365.0
                
                # IV Estimate (Using ATR proxy or fixed 15%)
                iv = 0.15 
                
                greeks = calculator.calculate_greeks(spot, strike, t_years, iv, opt_type)
                
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
        """Normalize symbol to uppercase trimmed string."""
        normalized = symbol.strip().upper()
        if not normalized:
            msg = "symbol must not be empty"
            raise ValueError(msg)
        
        # ✅ FIX: Strip 'NFO:' or 'NSE:' prefix if present
        if ":" in normalized:
            normalized = normalized.split(":", 1)[1]
            
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
    "TradeRecord",
    "OrderRouter",
]
