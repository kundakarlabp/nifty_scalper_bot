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
from nifty_scalper_bot.utils.smart_symbol import (
    generate_candidate_symbols_for_expiry,
)

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
            raise ValueError("signal_cooldown_seconds must be non-negative")
        if self.trade_cooldown_seconds < 0:
            raise ValueError("trade_cooldown_seconds must be non-negative")
        if self.min_indicator_bars < 0:
            raise ValueError("min_indicator_bars must be non-negative")
        if self.max_trade_history <= 0:
            raise ValueError("max_trade_history must be positive")


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
    """Return ``True`` when *expiry* corresponds to the monthly contract."""
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
            except Exception as exc:
                self._logger.debug("Unable to read selector settings: %s", exc)

        self._lock = threading.RLock()
        self._running = False
        self._trading_paused = False

        self._active_symbols: set[str] = set()
        self._symbol_state: Dict[str, SymbolState] = {}
        self._callbacks: MutableMapping[str, Callable[[dict], None]] = {}
        self._bar_builders: Dict[str, OneMinuteBarBuilder] = {}
        self._orchestrator = getattr(strategy_manager, "orchestrator", None)
        self._persistent_state: PersistentStateManager | None = None

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

    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        self._logger.debug("Entered attach_persistent_state")
        self._persistent_state = manager

    def restore_trades(self, trades: Iterable["TradeDict | Mapping[str, object]"]) -> None:
        self._logger.debug("Entered restore_trades")
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
                timestamp = ts_value.replace(tzinfo=timezone.utc) if ts_value.tzinfo is None else ts_value.astimezone(timezone.utc)
            elif isinstance(raw_ts, str):
                try:
                    parsed = datetime.fromisoformat(raw_ts)
                except ValueError:
                    parsed = timestamp
                timestamp = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            
            action = str(trade.get("action") or trade.get("side") or "").upper()
            quantity = _extract_int(trade, "quantity")
            price = _extract_float(trade, "price") or 0.0
            status = str(trade.get("status") or "").upper()
            
            if not action or quantity == 0 or price <= 0.0 or not status:
                continue
                
            record = TradeRecord(
                timestamp=timestamp,
                action=action,
                quantity=quantity,
                price=price,
                status=status,
                reason=str(trade.get("reason")) if trade.get("reason") else None,
                order_id=str(trade.get("order_id") or trade.get("orderId")) if trade.get("order_id") else None,
            )
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state is None:
                    state = SymbolState(symbol=symbol, history_limit=self._config.max_trade_history)
                    state.active = False
                    self._symbol_state[symbol] = state
                state.trade_history.append(record)
                state.last_trade_at = record.timestamp
            restored += 1
            
        if restored == 0:
            self._logger.info("Condition met: restore_trades_empty")
        else:
            self._logger.info("Condition met: restore_trades_applied", extra={"count": restored})

    def get_status(self) -> dict[str, Any]:
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
        self._logger.debug("Entered StrategyRunner._ingest_bar", extra={"symbol": symbol})
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
        self._logger.debug("Entered StrategyRunner._apply_premium_targets", extra={"side": entry_side})
        metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        stop_pct = _extract_float(metadata, "premium_stop_pct")
        target_rr = _extract_float(metadata, "premium_target_rr")
        
        if stop_pct is None or stop_pct <= 0:
            return signal
        if target_rr is None or target_rr <= 0:
            target_rr = 2.0
        if premium <= 0:
            return signal

        if entry_side == "BUY":
            stop_loss = max(premium * (1.0 - stop_pct), 0.01)
            risk = premium - stop_loss
            if risk <= 0: return signal
            take_profit = premium + risk * target_rr
        else:
            stop_loss = premium * (1.0 + stop_pct)
            risk = stop_loss - premium
            if risk <= 0: return signal
            take_profit = max(premium - risk * target_rr, 0.01)
            
        updated_metadata = dict(metadata)
        updated_metadata["computed_from_premium"] = True
        updated_metadata["entry_side"] = entry_side
        
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

    def aggregate_signals_by_symbol(self, signals: list[Signal]) -> dict[str, Signal]:
        if not signals:
            return {}
        try:
            grouped: dict[str, list[Signal]] = defaultdict(list)
            for signal in signals:
                grouped[signal.symbol].append(signal)
            
            aggregated: dict[str, Signal] = {}
            for symbol, symbol_signals in grouped.items():
                if len(symbol_signals) == 1:
                    aggregated[symbol] = symbol_signals[0]
                    continue
                    
                # Simple aggregation: best confidence
                best_signal = max(symbol_signals, key=lambda sig: sig.confidence)
                aggregated[symbol] = best_signal
                
            return aggregated
        except Exception as exc:
            self._logger.error("Failure in aggregate_signals_by_symbol: %s", exc)
            return {}

    def _notify_orchestrator_submission(self, signal: Signal, underlying: str) -> None:
        orchestrator = self._orchestrator
        if orchestrator is None: return
        try:
            orchestrator.notify_submission(signal, underlying)
        except Exception as exc:
            self._logger.debug("orchestrator_notify_submission_failed: %s", exc)

    def _notify_orchestrator_exit(self, underlying: str) -> None:
        orchestrator = self._orchestrator
        if orchestrator is None: return
        try:
            orchestrator.notify_exit(underlying)
        except Exception as exc:
            self._logger.debug("orchestrator_notify_exit_failed: %s", exc)

    def _on_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Handle incoming tick safely, updating state and triggering strategies."""
        try:
            # Safe math extraction
            price = _extract_float(tick, "ltp", "last_price", "close", "price")
            if price is None or price <= 0:
                return

            now = datetime.now(timezone.utc)
            volume = _extract_int(tick, "volume", "volume_traded")
            timestamp = _extract_timestamp(tick, now)

            # 1. Bar Building
            builder = self._bar_builders.setdefault(symbol, OneMinuteBarBuilder())
            try:
                completed_bar = builder.update(float(price), volume, timestamp)
                if completed_bar is not None:
                    self._ingest_bar(symbol, completed_bar)
            except ValueError as exc:
                self._logger.error("Bar update failed for %s: %s", symbol, exc)
                return

            # 2. Update Position Tracker
            if hasattr(self._position_manager, "has_position") and self._position_manager.has_position(symbol):
                try:
                    self._position_manager.update_position_price(symbol, price)
                except Exception:
                    pass

            # 3. Check Run State
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state is None: return
                state.last_tick = dict(tick)
                if not self._running or not state.active or self._trading_paused:
                    return
                if state.cooldown_until is not None and now < state.cooldown_until:
                    return

            # 4. Check Indicators Ready
            if self._config.min_indicator_bars:
                try:
                    if hasattr(self._indicator_engine, "is_ready"):
                        if not self._indicator_engine.is_ready(symbol, self._config.min_indicator_bars):
                            return
                except Exception:
                    return

            # 5. Generate Signal
            signal = self._strategy_manager.generate_signal(symbol, price)
            if signal is None or signal.action == "HOLD":
                return

            # 6. Throttle Signal
            with self._lock:
                state = self._symbol_state.get(symbol)
                if state is None or not state.active: return
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

            # 7. Handle Signal
            self._handle_signal(signal, price, now)

        except Exception as exc:
            # Catch-all to prevent thread death
            self._logger.error("Critical error in _on_tick for %s: %s", symbol, exc, exc_info=True)

    def _handle_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        self._logger.debug("Entered _handle_signal: %s %s", signal.action, signal.symbol)
        
        action = signal.action
        base_symbol = self._normalize_symbol(signal.symbol)
        trade_symbol = base_symbol
        trade_price = price
        selection: SelectedContract | None = None
        selector = self._strike_selector
        
        # Resolve Direction
        direction = "BULLISH" if action == "BUY" else "BEARISH"
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        
        # Resolve Option Type (Legacy/Metadata)
        option_type = metadata.get("option_type")
        if not option_type and self._legacy_side_to_type:
            option_type = "CE" if direction == "BULLISH" else "PE"
        
        # Determine Entry Side (Long vs Short premium)
        sell_premium = bool(metadata.get("sell_premium")) and not self._options_long_only
        entry_side: OrderSide = "SELL" if sell_premium else "BUY"
        
        # Position Manager - Active Contract Check
        if self._position_manager:
            active = self._position_manager.get_active_contract(base_symbol)
            if active:
                if self._position_manager.is_flat(active.symbol):
                    self._position_manager.clear_active_contract(base_symbol)
                else:
                    # Reuse active contract if logic permits
                    reuse = True
                    if active.option_type != option_type and not self._allow_hedge_entries:
                        reuse = False
                    
                    if reuse:
                        # Update price from active pos if available
                        pos = self._position_manager.get_position(active.symbol)
                        if pos:
                            trade_price = getattr(pos, "current_price", trade_price) or trade_price
                        
                        selection = SelectedContract(
                            symbol=active.symbol,
                            option_type=active.option_type,
                            strike=active.strike,
                            expiry=active.expiry,
                            ltp=trade_price,
                            delta=None,
                            metadata={"source": "position_manager"}
                        )
                        trade_symbol = selection.symbol

        # Strike Selection (if no active contract)
        if selector and not selection:
            # Safe coercion for option type literal
            safe_opt_type: Literal['CE', 'PE'] | None = None
            if option_type in ('CE', 'PE'):
                safe_opt_type = cast(Literal['CE', 'PE'], option_type)

            selector_side: Literal["BUY", "SELL"] = "BUY" if direction == "BULLISH" else "SELL"
            selection = selector.select_contract(
                underlying=base_symbol,
                side=selector_side,
                underlying_price=price,
                option_type=safe_opt_type,
            )

            # SMART SYMBOL VALIDATION
            if selection:
                try:
                    resolver = getattr(self._data_hub, "instrument_resolver", None) or getattr(self._data_hub, "resolver", None)
                    if resolver:
                        # Check if symbol is known
                        meta = resolver.lookup(selection.symbol)
                        if not meta:
                            # Try finding valid candidate
                            candidates = []
                            if selection.expiry and selection.strike:
                                candidates = generate_candidate_symbols_for_expiry(
                                    selection.expiry.date(), 
                                    int(selection.strike), 
                                    selection.option_type or "CE"
                                )
                            
                            found_sym = None
                            for cand in candidates:
                                # Try NFO: format first
                                if resolver.lookup(f"NFO:{cand}"):
                                    found_sym = f"NFO:{cand}"
                                    break
                                if resolver.lookup(cand):
                                    found_sym = cand
                                    break
                            
                            if found_sym:
                                selection.symbol = found_sym
                            else:
                                self._logger.warning("Strike selector symbol %s unknown to resolver", selection.symbol)
                                selection = None
                except Exception as exc:
                    self._logger.debug("Smart symbol validation failed: %s", exc)

        if not selection:
            self._record_trade(base_symbol, TradeRecord(timestamp, action, signal.quantity, price, "skipped", "no_contract"))
            return

        trade_symbol = selection.symbol
        trade_price = selection.ltp or price
        
        # Monthly Lockout Check
        lockout, _ = self._monthly_lockout_active(selection.expiry, timestamp)
        if lockout:
            self._record_trade(base_symbol, TradeRecord(timestamp, action, signal.quantity, trade_price, "skipped", "monthly_lockout"))
            return

        # Risk Sizing & Validation
        signal = self._apply_premium_targets(signal, trade_price, entry_side)
        
        # 1. Suggest Size
        atr_val = _extract_float(metadata, "atr")
        sized_qty = self._risk_manager.suggest_position_size(
            side=entry_side,
            price=trade_price,
            stop_loss=signal.stop_loss,
            atr=atr_val,
            requested_quantity=signal.quantity,
            confidence=signal.confidence,
            symbol=trade_symbol
        )
        
        if sized_qty <= 0:
             self._record_trade(base_symbol, TradeRecord(timestamp, action, signal.quantity, trade_price, "blocked", "risk_sizing_zero"))
             return

        # 2. Validate New Position
        allowed, reason = self._risk_manager.validate_new_position(
            symbol=trade_symbol,
            side="LONG" if entry_side == "BUY" else "SHORT",
            quantity=int(sized_qty),
            entry_price=trade_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        if not allowed:
            self._record_trade(base_symbol, TradeRecord(timestamp, action, int(sized_qty), trade_price, "blocked", str(reason)))
            self._set_signal_cooldown(base_symbol, timestamp)
            return

        # Execute Order
        try:
            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=entry_side,
                quantity=int(sized_qty),
                order_type=OrderType.MARKET,
                price=trade_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_id:
                self._logger.info("Submitted %s %s qty=%s id=%s", entry_side, trade_symbol, sized_qty, order_id)
                self._notify_orchestrator_submission(signal, base_symbol)
                
                if self._position_manager and selection:
                     # Update active contract tracking
                     if self._allow_hedge_entries or not self._position_manager.get_active_contract(base_symbol):
                        self._position_manager.set_active_contract(base_symbol, selection)

                if selector:
                    selector.register_open(base_symbol, selection)
                
                self._record_trade(base_symbol, TradeRecord(
                    timestamp, action, int(sized_qty), trade_price, "submitted", signal.reason, order_id
                ))
                self._set_trade_cooldown(base_symbol, timestamp)
                
        except Exception as exc:
            self._logger.error("Order placement failed: %s", exc)
            self._record_trade(base_symbol, TradeRecord(timestamp, action, int(sized_qty), trade_price, "error", str(exc)))

    def _set_trade_cooldown(self, symbol: str, timestamp: datetime) -> None:
        cooldown = self._config.trade_cooldown_seconds
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None: return
            state.last_trade_at = timestamp
            state.cooldown_until = timestamp + timedelta(seconds=cooldown) if cooldown > 0 else None

    def _set_signal_cooldown(self, symbol: str, timestamp: datetime) -> None:
        cooldown = self._config.signal_cooldown_seconds
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state is None: return
            state.cooldown_until = timestamp + timedelta(seconds=cooldown) if cooldown > 0 else None

    def _record_trade(self, symbol: str, record: TradeRecord) -> None:
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state:
                state.trade_history.append(record)
        
        if self._persistent_state:
            try:
                self._persistent_state.save_trade({"symbol": symbol, **record.to_dict()})
            except Exception:
                pass

    def _monthly_lockout_active(self, expiry: datetime, timestamp: datetime) -> tuple[bool, float]:
        if self._monthly_halt_minutes <= 0:
            return False, 0.0
        if not _is_monthly_expiry(expiry):
            return False, 0.0
        
        expiry_dt = expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry
        now_dt = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        
        minutes = (expiry_dt - now_dt).total_seconds() / 60.0
        if minutes < 0: return False, minutes
        return minutes <= self._monthly_halt_minutes, minutes

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        return normalized

    def _update_last_signal_selection(self, symbol: str, selection: SelectedContract) -> None:
        with self._lock:
            state = self._symbol_state.get(symbol)
            if state:
                info = state.strategy_data.setdefault("last_signal", {})
                if isinstance(info, dict):
                    info["selected_symbol"] = selection.symbol

    @staticmethod
    def _format_reason(reason: str | None, trade_symbol: str, base_symbol: str) -> str:
        reason = reason or ""
        if trade_symbol != base_symbol and trade_symbol not in reason:
            return f"{reason} [{trade_symbol}]".strip()
        return reason


__all__ = [
    "StrategyRunner",
    "StrategyRunnerConfig",
    "SymbolState",
    "TradeRecord",
]
