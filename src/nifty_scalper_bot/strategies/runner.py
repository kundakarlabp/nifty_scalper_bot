"""Event-driven strategy runner coordinating trading managers."""

from __future__ import annotations

import calendar
import os
import threading
import asyncio
import time

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

    log_method = getattr(logger, level.lower(), logger.info)
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
        self._data_hub = data_hub
        self._strike_selector = strike_selector
        self._bracket_manager = bracket_manager
        self._symbol_source: MarketDataManager | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None

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
        self._orchestrator = getattr(strategy_manager, "orchestrator", None)
        self._persistent_state: PersistentStateManager | None = None

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

    def _ingest_bar(self, symbol: str, bar: OneMinuteBar) -> None:
        """Persist the completed minute bar and UPDATE BRACKET MANAGER."""
        self._logger.debug(
            "Entered StrategyRunner._ingest_bar",
            extra={"event": "ingest_bar", "symbol": symbol},
        )

        payload = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
        }

        try:
            self._indicator_engine.update_price(
                symbol,
                payload,
                volume=bar.volume,
                timestamp=bar.end,
            )

            if self._bracket_manager:
                # Get ATR (Returns ATRSnapshot object OR float)
                raw_atr = self._indicator_engine.compute_atr(symbol, period=14)
                
                # Unwrap the value safely
                atr_value = 0.0
                if hasattr(raw_atr, 'value'):
                    atr_value = float(raw_atr.value) # It's a Snapshot
                elif hasattr(raw_atr, 'atr'):
                    atr_value = float(raw_atr.atr)   # Alternate format
                elif isinstance(raw_atr, (int, float)):
                    atr_value = float(raw_atr)       # It's a raw number

                if atr_value > 0:
                     if hasattr(self._bracket_manager, "update_market_stats"):
                         # Pass the simple float value to the manager
                         self._bracket_manager.update_market_stats(symbol, atr=atr_value)
                         self._logger.debug(f"💉 Injected ATR {atr_value:.2f} into BracketManager for {symbol}")

        except Exception as exc:
            self._logger.error(
                "Failure in _ingest_bar: %s",
                exc,
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
            "min_liquidity_score": float(self._option_min_liquidity),
            "side": side,
        }

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Process incoming market data ticks."""
        normalized = self._normalize_symbol(symbol)
        
        # --- Update BracketManager with every tick ---
        if self._bracket_manager:
            ltp = _extract_float(tick, "last_price", "ltp", "close")
            if ltp is not None and ltp > 0:
                 # Pass dict with 'ltp' to be safe
                 tick_data = {"ltp": ltp}
                 if hasattr(self._bracket_manager, "update_market_price"):
                     self._bracket_manager.update_market_price(normalized, tick_data)
        # ---------------------------------------------
        
        with self._lock:
            if not self._running:
                return

            state = self._symbol_state.get(normalized)
            if state is None:
                return

            # Keep a reference to dict to update state
            state.last_tick = dict(tick)
            
            # --- Throttle Strategy Evaluation ---
            now = datetime.now(timezone.utc)
            last_eval = state._last_strategy_eval or datetime.min.replace(tzinfo=timezone.utc)
            
            # Only evaluate strategy every 1 second per symbol
            if (now - last_eval).total_seconds() < 1.0:
                 return # Skip this tick for strategy evaluation
            
            state._last_strategy_eval = now
            # ------------------------------------

        self._update_bar_builder(normalized, tick)
        self._evaluate_strategy(normalized, tick)

    def _update_bar_builder(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Push tick into bar builder and ingest completed bars."""
        builder = self._bar_builders.get(symbol)
        if builder is None:
            builder = OneMinuteBarBuilder()
            self._bar_builders[symbol] = builder

        price = _extract_float(tick, "last_price", "ltp", "close")
        if price is None:
            return

        volume = _extract_int(tick, "volume", "quantity", "vol")
        
        # Use server timestamp if available, else local time
        timestamp_val = _extract_timestamp(tick, datetime.now(timezone.utc))

        completed = builder.add_tick(
            price=price,
            volume=volume,
            timestamp=timestamp_val,
        )

        if completed is not None:
            self._ingest_bar(symbol, completed)

    def _evaluate_strategy(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Execute strategy logic and process generated signals."""
        with self._lock:
            if self._trading_paused:
                return

            state = self._symbol_state.get(symbol)
            if state is None or not state.active:
                return

            if state.cooldown_until:
                if datetime.now(timezone.utc) < state.cooldown_until:
                    return
                state.cooldown_until = None

        try:
            signals = self._strategy_manager.evaluate(symbol, tick)
            if not signals:
                return
        except Exception as exc:
            self._logger.error(
                "Strategy evaluation failed for %s: %s",
                symbol,
                exc,
                extra={"event": "strategy_eval_error", "symbol": symbol},
                exc_info=exc,
            )
            return

        aggregated = self.aggregate_signals_by_symbol(signals)
        signal = aggregated.get(symbol)

        if signal:
            self._process_signal(symbol, signal, tick)

    def _process_signal(
        self,
        symbol: str,
        signal: Signal,
        tick: Mapping[str, Any],
    ) -> None:
        """Route valid signal to execution logic."""
        self._logger.debug(
            "Entered StrategyRunner._process_signal",
            extra={
                "event": "process_signal_enter",
                "symbol": symbol,
                "confidence": signal.confidence,
            },
        )

        timestamp = datetime.now(timezone.utc)
        
        # Check against last trade time to enforce trade cooldown
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state:
                # Signal Cooldown
                last_sig = state.last_signal_at
                if last_sig:
                     delta = (timestamp - last_sig).total_seconds()
                     if delta < self._config.signal_cooldown_seconds:
                         log_throttled(self._logger, f"sig_cool_{symbol}", f"Signal cooldown active for {symbol}")
                         return

                # Trade Cooldown
                last_trade = state.last_trade_at
                if last_trade:
                    delta = (timestamp - last_trade).total_seconds()
                    if delta < self._config.trade_cooldown_seconds:
                        log_throttled(self._logger, f"trade_cool_{symbol}", f"Trade cooldown active for {symbol}")
                        return
                
                state.last_signal_at = timestamp

        # Execute
        try:
            self._execute_signal(symbol, signal, tick)
        except Exception as exc:
            self._logger.error(
                "Signal execution failed for %s: %s",
                symbol,
                exc,
                extra={"event": "signal_exec_error", "symbol": symbol},
                exc_info=exc,
            )

    def _execute_signal(
        self,
        symbol: str,
        signal: Signal,
        tick: Mapping[str, Any],
    ) -> None:
        """Convert signal to order and route to order manager."""
        self._logger.debug(
            "Entered StrategyRunner._execute_signal",
            extra={"symbol": symbol, "action": signal.action},
        )

        # 1. Check Risk
        if not self._risk_manager.can_trade(
            symbol=symbol,
            side=signal.action,
            quantity=signal.quantity,
            price=0.0, # Market order check
        ):
             _STRATEGY_SKIP_COUNTER.labels(reason="risk_check_failed").inc()
             self._logger.warning(
                 "Risk check failed for %s signal on %s",
                 signal.action,
                 symbol,
                 extra={"event": "risk_check_failed", "symbol": symbol},
             )
             return

        # 2. Handle Options Logic (if applicable)
        trade_symbol = symbol
        execution_price = _extract_float(tick, "last_price", "ltp", "close") or 0.0
        
        # NIFTY Options Selection Logic
        is_nifty = "NIFTY" in symbol.upper() and not symbol.upper().endswith(OPTION_ALIAS_SUFFIX)
        if is_nifty and self._strike_selector:
            try:
                # Determine side
                selector_side: Literal["BUY", "SELL"] = "BUY"
                if self._options_long_only:
                     selector_side = "BUY"
                else:
                     selector_side = cast(Literal["BUY", "SELL"], signal.action)

                score_config = self._build_option_score_config(selector_side)
                
                # Fetch candidates
                candidates = self._strike_selector.select_strikes(
                    underlying_symbol=symbol,
                    direction=signal.action, # "BUY" or "SELL" (Trend)
                    reference_price=execution_price,
                    score_config=score_config
                )
                
                best_option = self._select_best_option(symbol, candidates)
                
                if best_option:
                    trade_symbol = best_option.symbol
                    # Update price to option price? Or keep underlying?
                    # Ideally we need option LTP.  If not available, we use 0 (Market)
                    # But for LIMIT orders we need a price.
                    # We will assume Market Order for now unless price is known.
                    if best_option.ltp and best_option.ltp > 0:
                        execution_price = best_option.ltp
                    else:
                        # Fetch quote for option
                         q = self._market_data.get_quote(trade_symbol)
                         execution_price = _extract_float(q, "ltp", "last_price") or 0.0

                    self._logger.info(
                        "Selected option %s for %s signal",
                        trade_symbol,
                        symbol,
                         extra={"event": "option_selected", "underlying": symbol, "option": trade_symbol}
                    )
                else:
                     self._logger.warning(
                         "No suitable option strikes found for %s",
                         symbol,
                         extra={"event": "no_option_found"}
                     )
                     return

            except Exception as exc:
                self._logger.error(
                    "Option selection failed: %s", exc, exc_info=True
                )
                return

        # 3. Sizing
        try:
             sized_qty = self._risk_manager.calculate_position_size(
                 symbol=trade_symbol,
                 price=execution_price,
                 stop_loss=signal.stop_loss,
                 account_balance=100000.0 # Placeholder, should come from account
             )
        except Exception:
             sized_qty = signal.quantity

        if sized_qty <= 0:
             _STRATEGY_SKIP_COUNTER.labels(reason="zero_quantity").inc()
             return

        # 4. Place Order
        timestamp = datetime.now(timezone.utc)
        entry_side = cast(Literal["BUY", "SELL"], signal.action)
        
        try:
            start_time = time.perf_counter()
            
            # ✅ CORRECT FIX: Read from metadata safely
            strat_name = signal.metadata.get("strategy", "MAN") if signal.metadata else "MAN"
            unique_tag = f"{strat_name[:3]}_{int(timestamp.timestamp())}"  # <--- You create this here

            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                quantity=int(sized_qty),
                order_type=OrderType.LIMIT,
                price=execution_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_id=unique_tag,  # ✅ FIX: Use the local variable (Guaranteed to exist)
                tag=unique_tag
            )
            
            latency = time.perf_counter() - start_time
            _NIFTY_OPTION_SIGNAL_LATENCY.labels(underlying=symbol).observe(latency)
            _NIFTY_OPTION_EXECUTION_COUNTER.labels(underlying=symbol, result="success").inc()

            self._logger.info(
                "Order placed successfully: %s",
                order_id,
                extra={
                    "event": "order_placed",
                    "order_id": order_id,
                    "symbol": trade_symbol,
                    "side": entry_side,
                    "qty": sized_qty,
                    "tag": unique_tag
                }
            )

            # 5. Record Trade
            record = TradeRecord(
                timestamp=timestamp,
                action=entry_side,
                quantity=int(sized_qty),
                price=execution_price,
                status="SUBMITTED",
                order_id=order_id,
                reason=signal.reason,
                reason_tags=signal.metadata
            )
            
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state:
                    state.trade_history.append(record)
                    state.last_trade_at = timestamp
                    # Set cooldown
                    state.cooldown_until = timestamp + timedelta(seconds=self._config.trade_cooldown_seconds)

        except OrderPlacementError as err:
             _NIFTY_OPTION_EXECUTION_COUNTER.labels(underlying=symbol, result="failure").inc()
             self._logger.error(
                 "Order placement failed: %s",
                 err,
                 extra={"event": "order_placement_failed", "error": str(err)}
             )
        except Exception as exc:
             _NIFTY_OPTION_EXECUTION_COUNTER.labels(underlying=symbol, result="error").inc()
             self._logger.error(
                 "Unexpected execution error: %s",
                 exc,
                 exc_info=True,
                 extra={"event": "execution_crash"}
             )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol string for consistent state keys."""
        return str(symbol).strip().upper()
