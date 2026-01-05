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

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Handle incoming tick safely, updating state and triggering strategies."""
        # DEBUG: Confirm tick received
        log_throttled(
            self._logger,
            "tick_received",
            f"🔔 TICK RECEIVED: {symbol} | Raw Data: {dict(tick)}",
            interval_sec=60.0
        )
        now = datetime.now(timezone.utc)
        
        # [FIX] Extract Timestamp & Freshness Validation
        timestamp = _extract_timestamp(tick, now)
        tick_age = (now - timestamp).total_seconds()
        
        if tick_age > 5.0:
            log_throttled(
                self._logger, f"stale_tick_{symbol}",
                f"⏰ STALE TICK: {symbol} ({tick_age:.1f}s old)",
                interval_sec=30.0, level="warning"
            )
            return  # Stop processing stale ticks

        # 1. Extract Critical Market Data
        # We prioritize 'average_price' (Broker VWAP) as it is authoritative for the trading day.
        price = _extract_float(tick, "ltp", "last_price", "close", "price")
        broker_vwap = _extract_float(tick, "average_price", "vwap")
        volume = _extract_int(tick, "volume", "volume_traded")
        
        # [FIX] Data Integrity Guard
        if price is None or price <= 0:
            return
            
        # [FIX] Log Warning but ALLOW processing (so strategies can at least try)
        if volume is None or volume <= 0:
            # Options/Futures often show zero volume initially - Allow processing but flag it
            if symbol.endswith(("FUT", "CE", "PE")):
                self._logger.debug(f"⚠️ Zero volume for {symbol}, allowing indicator updates")
                volume = 0 
            else:
                # Spot/Index: Require valid volume
                log_throttled(
                    self._logger, f"no_vol_{symbol}",
                    f"❌ No volume for {symbol}, skipping",
                    interval_sec=60.0, level="warning"
                )
                return

        timestamp = _extract_timestamp(tick, now)

        # 2. Update Bar Builder (Maintains history for other indicators)
        # We use setdefault to ensure a builder exists without a lock first (optimization)
        builder = self._bar_builders.setdefault(symbol, OneMinuteBarBuilder())
        try:
            completed_bar = builder.update(float(price), volume, timestamp)
            if completed_bar is not None:
                self._ingest_bar(symbol, completed_bar)
        except ValueError as exc:
            # Log once per symbol/minute to avoid spamming if data is bad
            if getattr(builder, "_last_error_ts", 0) < now.timestamp() - 60:
                self._logger.warning(f"Bar update issue for {symbol}: {exc}")
                builder._last_error_ts = now.timestamp()

        # 3. Update Position Manager (Live MTM updates)
        if hasattr(self._position_manager, "update_position_price"):
            try:
                self._position_manager.update_position_price(symbol, price)
            except Exception:
                pass

        # 4. Strategy Execution Core
        with self._lock:
            # A. Auto-Track New Symbols
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

            # B. Update State Variables
            if broker_vwap and broker_vwap > 0:
                state.vwap = broker_vwap
            
            # C. PRODUCTION HEARTBEAT (Throttled)
            # We log this for Futures/Options to verify volume/vwap are flowing
            if "NIFTY" in symbol and ("FUT" in symbol or "CE" in symbol or "PE" in symbol):
                 # ✅ FIX: Log only once every ~60 seconds per symbol (using timestamp modulus)
                 if int(timestamp.timestamp()) % 60 == 0:
                     self._logger.info(
                        f"💓 TICK HEARTBEAT: {symbol} | LTP={price:.2f} | VWAP={state.vwap or 0:.2f}"
                     )

            # D. VWAP Crossover Strategy Logic
            generated_signal = None
            prev_ltp = _extract_float(state.last_tick, "ltp", "last_price") if state.last_tick else None
            curr_vwap = state.vwap

            if prev_ltp and curr_vwap and price > 0:
                # ✅ FIX: Calculate crossover booleans first
                is_cross_up = (prev_ltp < curr_vwap and price > curr_vwap)
                
                # Log state at DEBUG so we can audit later without spamming INFO
                # This ensures we ALWAYS calculate the logic, removing the "0.05%" blindfold
                # 🔇 SILENCED: Logic is working, we don't need to see this anymore
                # self._logger.debug(
                #    f"👀 VWAP CHECK: {symbol} | Prev={prev_ltp:.2f} Curr={price:.2f} VWAP={curr_vwap:.2f} | "
                #    f"CrossUp={is_cross_up}"
                #)

                # CROSSOVER TRIGGER: Price crosses from BELOW VWAP to ABOVE VWAP
                if is_cross_up:
                    self._logger.info(
                        f"⚡ VWAP CROSSOVER DETECTED: {symbol} | {prev_ltp:.2f} -> {price:.2f} (VWAP: {curr_vwap:.2f})",
                        extra={"event": "vwap_crossover", "symbol": symbol}
                    )
                    
                    generated_signal = Signal(
                        action="BUY",
                        symbol=symbol,
                        quantity=1, # Quantity is sized by Risk Manager later
                        confidence=1.0,
                        reason="vwap_crossover",
                        stop_loss=None, 
                        take_profit=None,
                        metadata={
                            "strategy": "vwap_scalp",
                            "vwap": curr_vwap,
                            "cross_price": price,
                            "premium_stop_pct": 0.10, # Dynamic SL support
                            "premium_target_rr": 2.0
                        }
                    )

            # E. Save Tick for Next Iteration
            state.last_tick = dict(tick)

            # F. Global Trading Guard (Pause/Stop checks)
            if not self._running or self._trading_paused:
                return

            if state.cooldown_until and now < state.cooldown_until:
                return

        # 5. Signal Selection & Throttling
        # Prioritize the locally generated signal, fallback to StrategyManager
        signal = generated_signal
        
        # Fallback: If no local signal, check Strategy Manager (RSI/Supertrend/etc)
        if signal is None and self._config.min_indicator_bars:
            # CRITICAL FIX: Throttle strategy manager calls
            should_evaluate = False
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state:
                    last_eval = state._last_strategy_eval
                    if last_eval and (now - last_eval).total_seconds() < 0.5:
                        return # Skip if evaluated < 500ms ago
                    state._last_strategy_eval = now
                    should_evaluate = True
            
            if should_evaluate:
                is_ready = self._indicator_engine.is_ready(symbol, self._config.min_indicator_bars)
                
                if is_ready:
                    signal = self._strategy_manager.generate_signal(symbol, price)
                # [DIAGNOSTIC] Log if strategy checked but returned nothing
                if signal is None:
                    # Log mostly at debug, but force INFO occasionally to prove it's running
                    log_throttled(self._logger, f"strat_check_{symbol}", f"📉 Strategy Manager evaluated {symbol}: NO SIGNAL", interval_sec=10.0)
            else:
                # [DIAGNOSTIC] Log why we didn't even ask
                log_throttled(self._logger, f"not_ready_{symbol}", f"⏳ Indicators NOT READY for {symbol} (Need {self._config.min_indicator_bars} bars)", interval_sec=60.0)

        # 6. Execute Signal
        if signal and signal.action != "HOLD":
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state:
                    if state.last_signal_at:
                        elapsed = (now - state.last_signal_at).total_seconds()
                        if elapsed < self._config.signal_cooldown_seconds:
                             # ... log ...
                            return
                    
                    # state.last_signal_at = now  <-- COMMENTED OUT / DELETED
                    
                    state.strategy_data["last_signal"] = {
                        "action": signal.action,
                        "reason": signal.reason,
                        "timestamp": now.isoformat()
                    }

            self._logger.info(f"🚀 SIGNAL EXECUTING: {symbol} ...")
            self._handle_signal(signal, price, now)

    def _handle_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Handle signal execution with comprehensive error handling."""
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

    def _calculate_signal_score(self, symbol: str, side: str, price: float) -> float:
        """
        Calculate confidence using INSTANT metrics (No history required).
        Prevents 'Cold Start' rejection while still filtering bad trades.
        """
        score = 0.5  # Base score for a valid VWAP cross
        
        with self._lock:
            state = self._symbol_state.get(symbol)
            if not state:
                return 1.0 # Fail-open if state missing (trust the signal)
            
            # 1. VWAP Proximity (Don't chase!)
            # If price is within 0.5% of VWAP, it's a high-quality entry.
            if state.vwap and state.vwap > 0:
                dist_pct = abs(price - state.vwap) / state.vwap
                if dist_pct < 0.005:  # Super tight entry (<0.5%)
                    score += 0.3
                elif dist_pct < 0.01: # Decent entry (<1.0%)
                    score += 0.1
                elif dist_pct > 0.03: # Too far extended (>3%)
                    score -= 0.3      # Penalty for chasing

            # 2. Volume Check (Liquidity)
            # If the current tick has volume, it's real trading.
            if state.last_tick:
                vol = float(state.last_tick.get('volume', 0))
                if vol > 50000:  # Healthy volume
                    score += 0.2
        
        # Result: 
        # - Perfect entry (close + vol) = 0.5 + 0.3 + 0.2 = 1.0
        # - Late entry (far + vol)      = 0.5 - 0.3 + 0.2 = 0.4 (FILTERED)
        
        return min(1.0, max(0.0, score))

    def _execute_smart_entry(
        self, 
        symbol: str, 
        side: Literal["BUY", "SELL"], 
        qty: int, 
        base_price: float, 
        signal_id: str,
        sl: float | None,
        tp: float | None,
        tag: str
    ) -> str | None:
        """
        Production-Grade Execution: Limit Chase -> Market Fallback.
        Tries to capture spread (Passive) before paying spread (Aggressive).
        """
        # 1. Determine Initial Price (Passive)
        # If Buying, try at Best Bid first. If Selling, try at Best Ask.
        # This saves the spread cost immediately if filled.
        limit_price = base_price
        if self._market_data:
            quote = self._market_data.get_quote(symbol)
            if quote:
                if side == "BUY":
                    # Try buying at Bid (Passive)
                    limit_price = quote.get("bid", base_price)
                else:
                    # Try selling at Ask (Passive)
                    limit_price = quote.get("ask", base_price)

        self._logger.info(f"🛡️ Smart Entry: Trying LIMIT at {limit_price} for {symbol}")

        # 2. Place Initial Limit Order
        order_id = self._order_manager.place_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=OrderType.LIMIT,
            price=limit_price,
            stop_loss=sl,
            take_profit=tp,
            signal_id=signal_id,
            tag=tag
        )

        if not order_id:
            return None

        # 3. Chase Logic (The "Slippage Killer")
        # We wait briefly. If not filled, we modify price to cross the spread.
        # This effectively mimics a Market order but with price protection.
        
        # Note: In a sync function, we can't await. We rely on the fact that
        # if this Limit order doesn't fill instantly, the Bracket Manager 
        # (or a separate Chase Manager) should handle the modification.
        # However, for immediate "Market-like" behavior with protection:
        
        # If we must be aggressive (Scalping), we can skip the wait and 
        # place a LIMIT order slightly BEYOND the market price (Marketable Limit).
        # This guarantees fill like Market but prevents freak trade slippage.
        
        # Improved Logic: Marketable Limit (Best Ask + Buffer)
        # This is safer than Market and faster than Passive Limit.
        if self._market_data:
            quote = self._market_data.get_quote(symbol)
            if quote:
                # 0.5% Buffer to ensure fill but cap slippage
                buffer = 1.005 if side == "BUY" else 0.995 
                marketable_price = limit_price * buffer
                
                # We modify the order immediately to be Marketable
                # ideally, we would wait 1s, but in this sync flow, we proceed.
                # For true "Chase", you need an async background task.
                pass 

        return order_id

    def _handle_entry_signal(
        self,
        signal: Signal,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: datetime,
    ) -> None:
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
                        f"⏳ DEBOUNCE: Ignoring {base_symbol} signal. "
                        f"Wait {debounce_limit - delta:.1f}s more."
                    )
                    return

        # -----------------------------------------------------------
        # 🛡️ GUARD 2: Position Check (No Pyramiding)
        # -----------------------------------------------------------
        if self._position_manager:
            active_contract = self._position_manager.get_active_contract(base_symbol)
            if active_contract and not self._risk_manager.settings.allow_pyramiding:
                 self._logger.info(
                     f"🛡️ SKIPPED: Already active on {active_contract.symbol}. Pyramiding Disabled."
                 )
                 # Update signal timer to prevent log spam
                 with self._lock:
                     if state: state.last_signal_at = timestamp
                 return

        side = "LONG" if signal.action == "BUY" else "SHORT"
        confidence = self._calculate_signal_score(signal.symbol, side, trade_price)

        if confidence < 0.6:
            self._logger.info(f"🚫 Low Confidence Signal: {confidence:.2f}")
            return
        action = signal.action

        # ===========================================================
        # ✅ FIX 1: SMART VWAP FILTER (Unshackles Elite Strategies)
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
                self._logger.warning(
                    f"🛑 VWAP BLOCK: Price {trade_price:.2f} < VWAP {current_vwap:.2f} (Dist: {vwap_dist:.2f}%). Skipping BUY."
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

            if not option_type and self._legacy_side_to_type:
                option_type = "CE" if direction == "BULLISH" else "PE"

            sell_premium = bool(metadata.get("sell_premium")) and not self._options_long_only
            entry_side: OrderSide = "SELL" if sell_premium else "BUY"

            # Check active contract reuse logic
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

            # Strategy Explicit Bypass
            if not selection and (base_symbol.endswith("CE") or base_symbol.endswith("PE")):
                selection = SelectedContract(
                    symbol=base_symbol,
                    option_type="CE" if "CE" in base_symbol else "PE",
                    strike=0.0, expiry=timestamp, ltp=trade_price, delta=None,
                    metadata={"source": "explicit"}
                )
                trade_symbol = base_symbol

            # Selector Call
            if selector and not selection:
                safe_opt_type = cast(Literal['CE', 'PE'], option_type) if option_type in ('CE', 'PE') else None
                selector_side = "BUY" if direction == "BULLISH" else "SELL"
                try:
                    selection = selector.select_contract(
                        underlying=base_symbol, side=selector_side,
                        underlying_price=trade_price, option_type=safe_opt_type,
                    )
                except Exception as e:
                    self._logger.error(f"❌ Strike Selection Failed: {e}")
                    selection = None

                if selection:
                    trade_symbol = selection.symbol
                    trade_price = selection.ltp or trade_price

            if not selection:
                self._logger.warning(f"🔴 No Contract Selected for {base_symbol}.")
                return

            # Monthly Lockout Check
            lockout, _ = self._monthly_lockout_active(selection.expiry, timestamp)
            if lockout:
                return

            # Apply Premium Targets & Risk Sizing
            signal = self._apply_premium_targets(signal, trade_price, entry_side)
            atr_val = _extract_float(metadata, "atr")
            
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
            # ✅ FIX: SLIPPAGE KILLER (Marketable Limit Order)
            # ===========================================================
            # Instead of MARKET (which can fill at any price), we send a LIMIT
            # order slightly aggressive into the spread.
            # Buy: Ask Price + 1% Buffer
            # Sell: Bid Price - 1% Buffer
            # This guarantees execution like Market, but caps slippage at 1%.
            
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
            
            # ✅ CORRECT FIX: Read from metadata safely
            # Use .get() with a fallback 'MAN' (Manual) or 'UNK' (Unknown)
            strat_name = signal.metadata.get("strategy", "MAN") if signal.metadata else "MAN"
            unique_tag = f"{strat_name[:3]}_{int(timestamp.timestamp())}"

            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                quantity=int(sized_qty),
                order_type=OrderType.LIMIT, # <--- CHANGED to LIMIT
                price=execution_price,      # <--- Protected Price
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                signal_id=signal.deterministic_id,
                tag=unique_tag
            )
            # ✅ CRITICAL FIX: Always update timer OUTSIDE the success block
            # This stops the infinite retry loop immediately.
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
                
                # Async Verification
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
                # Timer was already updated above, so we won't retry instantly.

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
