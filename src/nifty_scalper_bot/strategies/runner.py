"""Event-driven strategy runner coordinating trading managers."""

from __future__ import annotations

import calendar
import os
import threading
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
    cast,
)

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType
from nifty_scalper_bot.execution.position_manager import OrderSide, PositionManager
from nifty_scalper_bot.options.strike_selector import SelectedContract, StrikeSelector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar, OneMinuteBarBuilder
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal
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


_STRATEGY_SKIP_COUNTER = Counter(
    "strategy_skips_total", "Strategy skip counts by reason", ["reason"]
)


class OrderRouter(Protocol):
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

    signal_cooldown_seconds: float = 30.0
    trade_cooldown_seconds: float = 60.0
    min_indicator_bars: int = 50
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
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _extract_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_int(payload: Mapping[str, Any], *keys: str) -> int:
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
    raw = (
        payload.get("timestamp")
        or payload.get("ts")
        or payload.get("ts_ms")
        or payload.get("last_trade_time")
    )
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1_000_000_000_000:
            value /= 1000.0
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return fallback


def _is_monthly_expiry(expiry: datetime) -> bool:
    """Return ``True`` when *expiry* corresponds to the monthly contract.

    Args:
        expiry: Expiry timestamp to classify.

    Returns:
        bool: ``True`` when the timestamp falls on the monthly expiry session.

    Raises:
        None.
    """

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
        config: StrategyRunnerConfig | None = None,
        data_hub: "DataHub | None" = None,
        strike_selector: StrikeSelector | None = None,
    ) -> None:
        self._market_data = market_data_manager
        self._indicator_engine = indicator_engine
        self._strategy_manager = strategy_manager
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._config = config or StrategyRunnerConfig()
        self._logger = get_logger(__name__)
        self._data_hub = data_hub
        self._strike_selector = strike_selector
        hedge_env = os.getenv("NSB__ALLOW_HEDGE_ENTRIES", "false").strip().lower()
        self._allow_hedge_entries = hedge_env in {"1", "true", "yes", "on"}
        self._options_long_only = True
        self._legacy_side_to_type = False
        self._monthly_halt_minutes = 0
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
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "Unable to read selector settings: %s", exc, exc_info=True
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

    # ------------------------------------------------------------------
    # Lifecycle management
    def start(self) -> None:
        """Start processing market data events."""

        with self._lock:
            if self._running:
                return
            self._running = True
            self._trading_paused = False
            symbols = list(self._active_symbols)

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
        # self._trading_paused = True  # DISABLED - Enable trading

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

    # ------------------------------------------------------------------
    # Symbol management
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

    # ------------------------------------------------------------------
    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        """Attach persistent state manager used for trade recovery.

        Args:
            manager: Persistent state manager coordinating snapshots.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered attach_persistent_state",
            extra={"event": "runner_attach_persistent"},
        )
        self._persistent_state = manager

    def restore_trades(
        self, trades: Iterable["TradeDict | Mapping[str, object]"]
    ) -> None:
        """Restore trade history from persisted *trades* payloads.

        Args:
            trades: Iterable of trade dictionaries previously persisted.

        Returns:
            None.

        Raises:
            None.
        """

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

    # ------------------------------------------------------------------
    # Status helpers
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

    # ------------------------------------------------------------------
    # Internal helpers
    def _subscribe_symbol(self, symbol: str) -> None:
        callback = self._callbacks.get(symbol)
        if callback is None:

            def _callback(tick: Mapping[str, Any], sym: str = symbol) -> None:
                self._on_tick(sym, tick)

            callback = _callback
            self._callbacks[symbol] = callback
        if self._data_hub is not None:
            self._data_hub.subscribe_ticks(symbol, callback)
        else:
            self._market_data.subscribe(symbol, callback)

    def _ingest_bar(self, symbol: str, bar: OneMinuteBar) -> None:
        """Persist the completed minute bar in the indicator engine cache.

        Args:
            symbol: Canonical instrument identifier.
            bar: Completed one-minute bar ready for indicator consumption.

        Returns:
            None.

        Raises:
            None.
        """

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
        except Exception as exc:  # noqa: BLE001
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
        """Derive stop-loss and target from option premium metadata.

        Args:
            signal: Original strategy signal.
            premium: Option premium used for bracket placement.
            entry_side: Order side of the planned entry.

        Returns:
            Signal: Updated signal containing premium-derived brackets.

        Raises:
            None.
        """

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
        """Aggregate strategy signals by symbol to limit duplicated risk.

        Args:
            signals: Signals generated during the evaluation cycle.

        Returns:
            dict[str, Signal]: Mapping of symbols to aggregated signal objects.

        Raises:
            None.
        """

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
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StrategyRunner.aggregate_signals_by_symbol: %s",
                exc,
                extra={"event": "aggregate_signals_error"},
                exc_info=exc,
            )
            return {}

        return aggregated

    def _notify_orchestrator_submission(self, signal: Signal, underlying: str) -> None:
        """Inform the orchestrator about a successful submission.

        Args:
            signal: Signal that triggered the order submission.
            underlying: Underlying symbol tracked by the orchestrator.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StrategyRunner._notify_orchestrator_submission",
            extra={"event": "orchestrator_notify_submission", "underlying": underlying},
        )
        orchestrator = self._orchestrator
        if orchestrator is None:
            return
        try:
            orchestrator.notify_submission(signal, underlying)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "orchestrator_notify_submission_failed",
                extra={
                    "event": "orchestrator_notify_submission_failed",
                    "error": str(exc),
                },
            )

    def _notify_orchestrator_exit(self, underlying: str) -> None:
        """Alert the orchestrator when reduce-only exit completes.

        Args:
            underlying: Symbol whose exposure was released.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StrategyRunner._notify_orchestrator_exit",
            extra={"event": "orchestrator_notify_exit", "underlying": underlying},
        )
        orchestrator = self._orchestrator
        if orchestrator is None:
            return
        try:
            orchestrator.notify_exit(underlying)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "orchestrator_notify_exit_failed",
                extra={"event": "orchestrator_notify_exit_failed", "error": str(exc)},
            )

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        price = _extract_float(tick, "ltp", "last_price", "close", "price")
        if price is None or price <= 0:
            self._logger.debug("Ignoring tick without price for %s: %s", symbol, tick)
            return

        volume = _extract_int(tick, "volume", "volume_traded")
        timestamp = _extract_timestamp(tick, now)

        builder = self._bar_builders.setdefault(symbol, OneMinuteBarBuilder())
        try:
            completed_bar = builder.update(float(price), volume, timestamp)
        except ValueError as exc:
            self._logger.error(
                "Failure in one minute bar update for %s: %s",
                symbol,
                exc,
                extra={"event": "one_minute_bar_invalid"},
            )
            return

        if completed_bar is not None:
            self._ingest_bar(symbol, completed_bar)

        if hasattr(
            self._position_manager, "has_position"
        ) and self._position_manager.has_position(symbol):
            try:
                self._position_manager.update_position_price(symbol, price)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "Unable to update position price for %s: %s", symbol, exc
                )

        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return
            state.last_tick = dict(tick)
            active = state.active
            trading_paused = self._trading_paused
            running = self._running
            cooldown_until = state.cooldown_until
            min_bars = self._config.min_indicator_bars

        if not running or not active:
            return
        if trading_paused:
            return
        if cooldown_until is not None and now < cooldown_until:
            return
        if min_bars:
            is_ready = True
            if hasattr(self._indicator_engine, "is_ready"):
                try:
                    is_ready = self._indicator_engine.is_ready(symbol, min_bars)  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "Indicator readiness check failed for %s: %s", symbol, exc
                    )
            if not is_ready:
                return

        try:
            signal = self._strategy_manager.generate_signal(symbol, price)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Failed to generate signal for %s: %s", symbol, exc)
            return

        if signal is None or signal.action == "HOLD":
            return

        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None or not state.active:
                return
            if state.last_signal_at is not None:
                elapsed = (now - state.last_signal_at).total_seconds()
                if elapsed < self._config.signal_cooldown_seconds:
                    return
            state.last_signal_at = now
            state.strategy_data["last_signal"] = {
                "action": signal.action,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "generated_at": now.isoformat(),
            }

        self._handle_signal(signal, price, now)

    def _handle_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Process a trading signal and coordinate order placement.

        Args:
            signal: Strategy generated trading signal.
            price: Latest traded price of the underlying instrument.
            timestamp: Timestamp associated with the signal event.

        Returns:
            None: The method performs side effects without returning a value.

        Raises:
            RuntimeError: Not raised directly; downstream components may raise.
        """

        self._logger.debug(
            "Entered StrategyRunner._handle_signal",
            extra={"action": signal.action, "symbol": signal.symbol},
        )
        action = signal.action
        base_symbol = self._normalize_symbol(signal.symbol)
        trade_symbol = base_symbol
        trade_price = price
        selection: SelectedContract | None = None
        selector = self._strike_selector
        logged_reason_codes: set[str] = set()

        def _log_risk_rejection(template: str, *fmt_args: object, reason: str) -> str:
            code = canonical(reason)
            if code not in logged_reason_codes:
                logged_reason_codes.add(code)
                self._logger.info(template, *fmt_args, code)
            return code

        if action in {"BUY", "SELL"}:
            side = "LONG" if action == "BUY" else "SHORT"
            metadata: Mapping[str, Any] = {}
            if isinstance(signal.metadata, dict):
                metadata = signal.metadata

            direction_source = "signal_action"
            direction = "BULLISH" if action == "BUY" else "BEARISH"
            if metadata:
                direction_tokens = (
                    metadata.get("option_direction"),
                    metadata.get("direction"),
                    metadata.get("bias"),
                )
                for raw_token in direction_tokens:
                    if raw_token is None:
                        continue
                    try:
                        candidate = str(raw_token).strip().upper()
                    except Exception as exc:  # noqa: BLE001
                        self._logger.error(
                            "Failure in option_direction_coercion: %s", exc
                        )
                        continue
                    if not candidate:
                        continue
                    if candidate in {"BULLISH", "BULL", "LONG"}:
                        direction = "BULLISH"
                        direction_source = "metadata"
                        break
                    if candidate in {"BEARISH", "BEAR", "SHORT"}:
                        direction = "BEARISH"
                        direction_source = "metadata"
                        break
                    self._logger.error(
                        "Failure in option_direction_unrecognised: %s", candidate
                    )

            selector_side: Literal["BUY", "SELL"] = (
                "BUY" if direction == "BULLISH" else "SELL"
            )
            option_type: Literal["CE", "PE"] | None = None
            option_source = "metadata"
            override_token = metadata.get("option_type") or metadata.get(
                "option_type_override"
            )
            if override_token is not None:
                try:
                    candidate = str(override_token).strip().upper()
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in resolve_option_type_override: %s", exc
                    )
                else:
                    if candidate in {"CE", "PE"}:
                        option_type = cast("Literal['CE', 'PE']", candidate)
                    elif candidate:
                        self._logger.error(
                            "Failure in resolve_option_type_override_token: %s",
                            candidate,
                        )
            if option_type is None and self._legacy_side_to_type:
                option_type = cast(
                    "Literal['CE', 'PE']",
                    "CE" if direction == "BULLISH" else "PE",
                )
                option_source = "legacy_side_to_type"
            if option_type is None:
                self._logger.info(
                    "Condition met: missing_option_type",
                    extra={
                        "underlying": base_symbol,
                        "direction": direction,
                        "side": selector_side,
                        "metadata": dict(metadata),
                    },
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        price,
                        "skipped",
                        "missing_option_type",
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            requested_sell_premium = bool(metadata.get("sell_premium"))
            sell_premium_enabled = (
                requested_sell_premium and not self._options_long_only
            )
            if requested_sell_premium and self._options_long_only:
                self._logger.info(
                    "Condition met: sell_premium_blocked_long_only",
                    extra={
                        "underlying": base_symbol,
                        "direction": direction,
                        "requested_option_type": option_type,
                    },
                )
            entry_side: OrderSide = "SELL" if sell_premium_enabled else "BUY"
            self._logger.info(
                "Condition met: resolved_option_leg",
                extra={
                    "underlying": base_symbol,
                    "direction": direction,
                    "direction_source": direction_source,
                    "option_type": option_type,
                    "order_side": entry_side,
                    "selector_side": selector_side,
                    "sell_premium": sell_premium_enabled,
                    "legacy_side_to_type": self._legacy_side_to_type,
                    "option_source": option_source,
                },
            )
            atr_value = None
            if metadata:
                atr_value = metadata.get("atr")
            position_manager = self._position_manager
            active_contract = None
            reuse_active_contract = True
            if position_manager is not None:
                active_contract = position_manager.get_active_contract(base_symbol)
                if active_contract is not None:
                    if position_manager.is_flat(active_contract.symbol):
                        position_manager.clear_active_contract(base_symbol)
                    else:
                        expected_option = option_type
                        if active_contract.option_type != expected_option:
                            payload = {
                                "event": "conflicting_active_contract_blocked",
                                "underlying": base_symbol,
                                "active_symbol": active_contract.symbol,
                                "requested_action": action,
                            }
                            if not self._allow_hedge_entries:
                                self._logger.info(
                                    (
                                        "Condition met: "
                                        "conflicting_active_contract_blocked"
                                    ),
                                    extra=payload,
                                )
                                self._record_trade(
                                    base_symbol,
                                    TradeRecord(
                                        timestamp,
                                        action,
                                        signal.quantity,
                                        price,
                                        "blocked",
                                        "conflicting_active_contract",
                                    ),
                                )
                                self._set_signal_cooldown(base_symbol, timestamp)
                                return
                            payload["event"] = "conflicting_active_contract_allowed"
                            self._logger.info(
                                (
                                    "Condition met: "
                                    "conflicting_active_contract_allowed"
                                ),
                                extra=payload,
                            )
                            reuse_active_contract = False
                        if reuse_active_contract:
                            last_price = price
                            existing_position = position_manager.get_position(
                                active_contract.symbol
                            )
                            if existing_position is not None:
                                latest_price = getattr(
                                    existing_position, "current_price", None
                                )
                                if (
                                    isinstance(latest_price, (int, float))
                                    and latest_price > 0
                                ):
                                    last_price = float(latest_price)
                            selection = SelectedContract(
                                symbol=active_contract.symbol,
                                option_type=active_contract.option_type,
                                strike=active_contract.strike,
                                expiry=active_contract.expiry,
                                ltp=float(last_price),
                                delta=None,
                                metadata={"source": "position_manager"},
                            )
                            trade_symbol = selection.symbol
                            trade_price = selection.ltp or trade_price
                            self._update_last_signal_selection(base_symbol, selection)
            if selector is not None and selection is None:
                selection = selector.select_contract(
                    underlying=base_symbol,
                    side=selector_side,
                    underlying_price=price,
                    option_type=option_type,
                )
                if selection is None:
                    reason = "no contract passed strike selection"
                    self._logger.info(
                        "Strike selector rejected %s for %s", option_type, base_symbol
                    )
                    self._record_trade(
                        base_symbol,
                        TradeRecord(
                            timestamp,
                            action,
                            signal.quantity,
                            price,
                            "skipped",
                            reason,
                        ),
                    )
                    self._set_signal_cooldown(base_symbol, timestamp)
                    return
                trade_symbol = selection.symbol
                trade_price = selection.ltp or price
                if trade_price <= 0:
                    trade_price = price
                self._update_last_signal_selection(base_symbol, selection)
            if selection is not None:
                lockout_active, minutes_to_expiry = self._monthly_lockout_active(
                    selection.expiry, timestamp
                )
                if lockout_active:
                    self._logger.info(
                        "Condition met: monthly_expiry_lockout",
                        extra={
                            "event": "monthly_expiry_lockout",
                            "underlying": base_symbol,
                            "contract_expiry": selection.expiry.isoformat(),
                            "minutes_to_expiry": round(minutes_to_expiry, 2),
                            "halt_window_minutes": self._monthly_halt_minutes,
                        },
                    )
                    self._record_trade(
                        base_symbol,
                        TradeRecord(
                            timestamp,
                            action,
                            signal.quantity,
                            trade_price,
                            "skipped",
                            "monthly_expiry_lockout",
                            reason_tags={
                                "rule": "monthly_expiry_lockout",
                                "value": round(minutes_to_expiry, 2),
                                "threshold": float(self._monthly_halt_minutes),
                            },
                        ),
                    )
                    self._set_signal_cooldown(base_symbol, timestamp)
                    return
            signal = self._apply_premium_targets(signal, trade_price, entry_side)
            sized_quantity = int(
                self._risk_manager.suggest_position_size(
                    side=entry_side,
                    price=trade_price,
                    stop_loss=signal.stop_loss,
                    atr=atr_value,
                    requested_quantity=signal.quantity,
                    confidence=signal.confidence,
                    symbol=trade_symbol,
                )
            )
            if sized_quantity <= 0:
                reason = "Risk sizing rejected order"
                _log_risk_rejection(
                    "Risk rejected %s for %s: %s",
                    action,
                    trade_symbol,
                    reason=reason,
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        trade_price,
                        "blocked",
                        reason,
                        reason_tags={
                            "rule": "risk_sizing",
                            "value": float(sized_quantity),
                            "requested": float(signal.quantity),
                        },
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return
            if sized_quantity < signal.quantity:
                self._logger.debug(
                    "Risk sizing reduced quantity for %s from %s to %s",
                    trade_symbol,
                    signal.quantity,
                    sized_quantity,
                )
            allowed, reason = self._risk_manager.validate_new_position(
                symbol=trade_symbol,
                side=side,  # type: ignore[arg-type]
                quantity=sized_quantity,
                entry_price=trade_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            if not allowed:
                _log_risk_rejection(
                    "Risk rejected %s for %s: %s",
                    action,
                    trade_symbol,
                    reason=reason,
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        trade_price,
                        "blocked",
                        reason,
                        reason_tags={
                            "rule": str(reason),
                            "value": float(sized_quantity),
                            "requested": float(signal.quantity),
                        },
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            try:
                order_id = self._order_manager.place_order(
                    symbol=trade_symbol,
                    side=entry_side,
                    quantity=sized_quantity,
                    order_type=OrderType.MARKET,
                    price=trade_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
            except OrderPlacementError as exc:
                reason_hint = self._order_manager.consume_skip_reason()
                fallback_reason = str(exc).strip() or "order_error"
                raw_reason = reason_hint or fallback_reason
                reason_token = canonical(raw_reason)
                self._logger.info(
                    "skip_order",
                    extra={
                        "event": "strategy_order_skip",
                        "symbol": trade_symbol,
                        "base_symbol": base_symbol,
                        "reason": reason_token,
                        "raw_reason": str(raw_reason),
                        "stage": "entry",
                    },
                )
                try:
                    _STRATEGY_SKIP_COUNTER.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                try:
                    metrics.strategy_skips_total.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                if reason_token in {"no_reference_price", "margin_no_qty"}:
                    self._set_signal_cooldown(base_symbol, timestamp)
                    return
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        sized_quantity,
                        trade_price,
                        "skipped",
                        reason_token,
                        reason_tags={
                            "rule": reason_token,
                            "value": float(sized_quantity),
                            "requested": float(signal.quantity),
                        },
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.exception(
                    "Order placement failed for %s: %s", trade_symbol, exc
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        sized_quantity,
                        trade_price,
                        "error",
                        str(exc),
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            if not order_id:
                fallback_reason = (
                    self._order_manager.consume_skip_reason() or "risk_block"
                )
                reason_token = canonical(fallback_reason)
                self._logger.info(
                    "skip_order",
                    extra={
                        "event": "strategy_order_skip",
                        "symbol": trade_symbol,
                        "base_symbol": base_symbol,
                        "reason": reason_token,
                        "raw_reason": str(fallback_reason),
                        "stage": "entry_ack",
                    },
                )
                try:
                    _STRATEGY_SKIP_COUNTER.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                try:
                    metrics.strategy_skips_total.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                return

            self._logger.info(
                "Submitted %s order for %s qty=%s price=%.2f id=%s",
                action,
                trade_symbol,
                signal.quantity,
                trade_price,
                order_id,
            )
            self._notify_orchestrator_submission(signal, base_symbol)
            if position_manager is not None and selection is not None:
                should_update_active = True
                if (
                    self._allow_hedge_entries
                    and active_contract is not None
                    and active_contract.option_type != selection.option_type
                ):
                    should_update_active = False
                if should_update_active:
                    position_manager.set_active_contract(base_symbol, selection)
            if selector is not None and selection is not None:
                selector.register_open(base_symbol, selection)
            self._record_trade(
                base_symbol,
                TradeRecord(
                    timestamp,
                    action,
                    sized_quantity,
                    trade_price,
                    "submitted",
                    self._format_reason(signal.reason, trade_symbol, base_symbol),
                    order_id,
                ),
            )
            self._set_trade_cooldown(base_symbol, timestamp)
            return

        if action in {"CLOSE_LONG", "CLOSE_SHORT"}:
            selector = self._strike_selector
            position_manager = self._position_manager
            if selector is not None:
                selection = selector.resolve_active(base_symbol)
            if selection is None and position_manager is not None:
                cached = position_manager.get_active_contract(base_symbol)
                if cached is not None and not position_manager.is_flat(cached.symbol):
                    selection = SelectedContract(
                        symbol=cached.symbol,
                        option_type=cached.option_type,
                        strike=cached.strike,
                        expiry=cached.expiry,
                        ltp=price,
                        delta=None,
                        metadata={"source": "position_manager"},
                    )
                    trade_symbol = selection.symbol
            if selection is None:
                self._logger.info("No active contract to close for %s", base_symbol)
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        price,
                        "skipped",
                        "no contract",
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return
            trade_symbol = selection.symbol
            trade_price = selection.ltp or price
            if self._data_hub is not None:
                quote = self._data_hub.get_quote(trade_symbol)
            else:
                quote = None
            if quote is not None:
                refreshed = _extract_float(quote, "ltp", "last_price", "close", "price")
                if refreshed:
                    trade_price = refreshed
            if position_manager is not None:
                position_view = position_manager.get_position(trade_symbol)
                if position_view is not None:
                    latest_price = getattr(position_view, "current_price", None)
                    if isinstance(latest_price, (int, float)) and latest_price > 0:
                        trade_price = float(latest_price)
            if trade_price <= 0:
                trade_price = price
            self._update_last_signal_selection(base_symbol, selection)
            position = self._position_manager.get_position(trade_symbol)
            if position is None or position.quantity <= 0:
                self._logger.info("No open position to close for %s", trade_symbol)
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        signal.quantity,
                        trade_price,
                        "skipped",
                        "no position",
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            allowed, reason = self._risk_manager.validate_close_position(
                symbol=trade_symbol, exit_price=trade_price
            )
            if not allowed:
                _log_risk_rejection(
                    "Risk rejected close signal for %s: %s",
                    trade_symbol,
                    reason=reason,
                )
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        position.quantity,
                        trade_price,
                        "blocked",
                        reason,
                        reason_tags={
                            "rule": str(reason),
                            "value": float(position.quantity),
                        },
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            exit_side: OrderSide = "SELL" if action == "CLOSE_LONG" else "BUY"
            exit_order_id: str | None = None
            try:
                exchange = (
                    trade_symbol.split(":", maxsplit=1)[0]
                    if ":" in trade_symbol
                    else "NFO"
                )
                tradingsymbol = (
                    trade_symbol.split(":", maxsplit=1)[-1]
                    if ":" in trade_symbol
                    else trade_symbol
                )
                intent = ExitIntent(
                    symbol=tradingsymbol,
                    qty=position.quantity,
                    product="MIS",
                    exchange=exchange,
                    order_type="MARKET",
                    tag="reduce_only",
                )
                exit_order_id = self._order_manager.place_reduce_only_exit(intent)
            except OrderPlacementError as exc:
                reason_hint = self._order_manager.consume_skip_reason()
                fallback_reason = str(exc).strip() or "order_error"
                raw_reason = reason_hint or fallback_reason
                reason_token = canonical(raw_reason)
                self._logger.info(
                    "skip_order",
                    extra={
                        "event": "strategy_exit_skip",
                        "symbol": trade_symbol,
                        "base_symbol": base_symbol,
                        "reason": reason_token,
                        "raw_reason": str(raw_reason),
                        "stage": "exit",
                    },
                )
                try:
                    _STRATEGY_SKIP_COUNTER.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                try:
                    metrics.strategy_skips_total.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                if reason_token in {"no_reference_price", "margin_no_qty"}:
                    self._set_signal_cooldown(base_symbol, timestamp)
                    return
                self._record_trade(
                    base_symbol,
                    TradeRecord(
                        timestamp,
                        action,
                        position.quantity,
                        trade_price,
                        "skipped",
                        reason_token,
                        reason_tags={
                            "rule": reason_token,
                            "value": float(position.quantity),
                        },
                    ),
                )
                self._set_signal_cooldown(base_symbol, timestamp)
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.exception(
                    "Failed to place close order for %s: %s", trade_symbol, exc
                )
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
                self._set_signal_cooldown(base_symbol, timestamp)
                return

            if not exit_order_id:
                fallback_reason = (
                    self._order_manager.consume_skip_reason() or "risk_block"
                )
                reason_token = canonical(fallback_reason)
                self._logger.info(
                    "skip_order",
                    extra={
                        "event": "strategy_exit_skip",
                        "symbol": trade_symbol,
                        "base_symbol": base_symbol,
                        "reason": reason_token,
                        "raw_reason": str(fallback_reason),
                        "stage": "exit_ack",
                    },
                )
                try:
                    _STRATEGY_SKIP_COUNTER.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                try:
                    metrics.strategy_skips_total.labels(reason=reason_token).inc()
                except Exception:  # pragma: no cover - metrics optional
                    pass
                return

            self._logger.info(
                "Submitted reduce-only %s for %s qty=%s price=%.2f id=%s",
                exit_side,
                trade_symbol,
                position.quantity,
                trade_price,
                exit_order_id,
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
                    self._format_reason(
                        signal.reason or "Close position", trade_symbol, base_symbol
                    ),
                    exit_order_id,
                ),
            )
            self._set_trade_cooldown(base_symbol, timestamp)

    def _set_trade_cooldown(self, symbol: str, timestamp: datetime) -> None:
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
        cooldown = self._config.signal_cooldown_seconds
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None:
                return
            state.cooldown_until = (
                timestamp + timedelta(seconds=cooldown) if cooldown > 0 else None
            )

    def _record_trade(self, symbol: str, record: TradeRecord) -> None:
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
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in _record_trade persistence: %s", exc)
        if record.status in {"skipped", "blocked"}:
            self._log_skip_tags(symbol, record)

    def _log_skip_tags(self, symbol: str, record: TradeRecord) -> None:
        """Log detailed reason tags for skip/block trade records.

        Args:
            symbol: Base instrument symbol used for trade tracking.
            record: Trade record capturing the skip or block outcome.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StrategyRunner._log_skip_tags",
            extra={"event": "strategy_skip_tags_enter", "symbol": symbol},
        )
        try:
            tags = dict(record.reason_tags) if record.reason_tags else {}
            raw_reason = record.reason or "unknown"
            reason_code = canonical(raw_reason) if raw_reason else "unknown"
            rule_tag = tags.get("rule")
            canonical_rule = (
                canonical(str(rule_tag)) if rule_tag is not None else reason_code
            )
            if rule_tag and canonical_rule != rule_tag:
                tags.setdefault("raw_rule", str(rule_tag))
            tags["rule"] = canonical_rule
            payload = {
                "event": "strategy_skip_tags",
                "symbol": symbol,
                "action": record.action,
                "status": record.status,
                "reason": tags.get("rule", reason_code),
                "raw_reason": raw_reason,
                "tags": tags,
            }
            value = tags.get("value")
            rule = tags.get("rule", reason_code)
            self._logger.info(
                "Condition met: strategy_%s_reason reason=%s value=%s rule=%s",
                record.status,
                rule,
                value,
                rule,
                extra=payload,
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in StrategyRunner._log_skip_tags: %s",
                exc,
                extra={"event": "strategy_skip_tags_error", "symbol": symbol},
            )

    def _monthly_lockout_active(
        self, expiry: datetime, timestamp: datetime
    ) -> tuple[bool, float]:
        """Return monthly expiry lockout state and minutes to expiry.

        Args:
            expiry: Contract expiry timestamp.
            timestamp: Current evaluation timestamp.

        Returns:
            tuple[bool, float]: Tuple of (lockout_active, minutes_to_expiry).

        Raises:
            None.
        """

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
        normalized = symbol.strip().upper()
        if not normalized:
            msg = "symbol must not be empty"
            raise ValueError(msg)
        return normalized

    def _update_last_signal_selection(
        self, symbol: str, selection: SelectedContract
    ) -> None:
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
]
