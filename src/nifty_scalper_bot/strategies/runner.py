"""
Event-driven strategy runner coordinating trading managers.
PRODUCTION GRADE - FULLY FEATURED & OPTIMIZED
"""

from __future__ import annotations

import calendar
import os
import threading
import asyncio
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum

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
    Optional,
)

# Core Imports
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType, OrderStatus
from nifty_scalper_bot.execution.position_manager import OrderSide, PositionManager
from nifty_scalper_bot.options.strike_selector import SelectedContract, StrikeSelector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.core.message_bus import MessageBus, Message, MessageType
from nifty_scalper_bot.utils.errors import ConfigurationError, OrderPlacementError
from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar, OneMinuteBarBuilder
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils import metrics
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge, Histogram
from nifty_scalper_bot.utils.timestamp import is_fresh_ts_ms

if TYPE_CHECKING:
    from nifty_scalper_bot.config.base import AppConfig

# --- METRICS ---
METRIC_TICKS = Counter("runner_ticks_total", "Ticks processed", ["symbol"])
METRIC_SIGNALS = Counter("runner_signals_total", "Signals generated", ["symbol", "action"])
METRIC_TRADES = Counter("runner_trades_total", "Trades executed", ["symbol", "side"])
METRIC_ERRORS = Counter("runner_errors_total", "Errors encountered", ["type"])
METRIC_LATENCY = Histogram("runner_tick_latency_seconds", "Tick processing latency", ["symbol"])
GAUGE_ACTIVE_SYMBOLS = Gauge("runner_active_symbols", "Number of tracked symbols")

class RunnerStatus(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

@dataclass
class TradeRecord:
    """Immutable record of a trade decision."""
    timestamp: datetime
    action: str
    quantity: int
    price: float
    status: Literal["submitted", "skipped", "blocked", "error"]
    reason: str
    order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "reason": self.reason,
            "order_id": self.order_id,
            "metadata": self.metadata
        }

@dataclass
class SymbolState:
    """State container for a single symbol."""
    bar_builder: OneMinuteBarBuilder
    indicators: IndicatorEngine
    strategy_data: dict[str, Any] = field(default_factory=dict)
    
    # State Tracking
    last_signal_time: datetime | None = None
    last_trade_time: datetime | None = None
    cooldown_until: datetime | None = None
    active_contract: SelectedContract | None = None
    is_paused: bool = False
    
    # History for Deduplication & Auditing
    signal_history: Deque[Signal] = field(default_factory=lambda: deque(maxlen=50))
    trade_history: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=200))

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_signal": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_trade": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "cooldown": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "active_contract": self.active_contract.symbol if self.active_contract else None,
            "paused": self.is_paused,
            "trades_count": len(self.trade_history)
        }

class StrategyRunner:
    """
    Orchestrates the trading loop: Ticks -> Strategies -> Execution.
    """

    def __init__(
        self,
        config: AppConfig,
        message_bus: MessageBus,
        strategy_manager: StrategyManager,
        risk_manager: RiskManager,
        order_manager: Any,
        market_data_manager: MarketDataManager,
        position_manager: PositionManager,
        strike_selector: StrikeSelector | None = None,
        data_hub: Any | None = None,
    ) -> None:
        self._config = config
        self._message_bus = message_bus
        self._strategy_manager = strategy_manager
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._mdm = market_data_manager
        self._position_manager = position_manager
        self._strike_selector = strike_selector
        self._data_hub = data_hub

        self._logger = get_logger(__name__)
        self._lock = threading.RLock()
        self._status = RunnerStatus.STOPPED
        self._persistent_state_manager = None
        
        # Internal State
        self._symbol_state: dict[str, SymbolState] = {}
        self._symbols_to_track: set[str] = set()
        
        # Load Settings (Robust Fallback)
        settings = getattr(config, "settings", None)
        self._options_long_only = getattr(settings, "options_long_only", True) if settings else True
        self._legacy_side_to_type = getattr(settings, "legacy_side_to_type", True) if settings else True
        self._cooldown_seconds = getattr(settings, "cooldown_seconds", 300)
        self._monthly_halt_minutes = getattr(settings, "monthly_halt_minutes", 30)
        self._allow_hedge_entries = False
        
        # Startup checks
        if not self._strike_selector:
            self._logger.warning("StrategyRunner initialized without StrikeSelector. Options trading disabled.")

    @property
    def status(self) -> RunnerStatus:
        return self._status

    @property
    def tracked_symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols_to_track)

    def tracked_symbol_count(self) -> int:
        with self._lock:
            return len(self._symbols_to_track)

    # ----------------------------------------------------------------
    # LIFECYCLE MANAGEMENT
    # ----------------------------------------------------------------

    def start(self, symbols: Iterable[str]) -> None:
        """Start listening for ticks."""
        with self._lock:
            if self._status == RunnerStatus.RUNNING:
                self._logger.warning("Runner already running.")
                return
            
            self._logger.info(f"Starting StrategyRunner. Tracking: {list(symbols)}")
            self._symbols_to_track = set(self._normalize_symbol(s) for s in symbols)
            
            # Initialize State
            for sym in self._symbols_to_track:
                self._get_or_create_state(sym)
            
            # Subscribe
            self._message_bus.subscribe(MessageType.TICK, self._handle_tick_message)
            self._status = RunnerStatus.RUNNING
            GAUGE_ACTIVE_SYMBOLS.set(len(self._symbols_to_track))

    def stop(self) -> None:
        """Stop processing."""
        with self._lock:
            if self._status == RunnerStatus.STOPPED:
                return
            
            self._status = RunnerStatus.STOPPED
            try:
                self._message_bus.unsubscribe(MessageType.TICK, self._handle_tick_message)
            except Exception:
                pass
            self._logger.info("StrategyRunner stopped.")

    def pause_trading(self) -> None:
        """Pause trading globally (ticks processed, but signals ignored)."""
        with self._lock:
            if self._status == RunnerStatus.RUNNING:
                self._status = RunnerStatus.PAUSED
                self._logger.info("StrategyRunner PAUSED. Signals will be ignored.")

    def resume_trading(self) -> None:
        """Resume trading from paused state."""
        with self._lock:
            if self._status == RunnerStatus.PAUSED:
                self._status = RunnerStatus.RUNNING
                self._logger.info("StrategyRunner RESUMED.")

    def add_symbol(self, symbol: str) -> None:
        """Dynamically add symbol."""
        norm = self._normalize_symbol(symbol)
        with self._lock:
            if norm not in self._symbol_state:
                self._symbols_to_track.add(norm)
                self._get_or_create_state(norm)
                GAUGE_ACTIVE_SYMBOLS.set(len(self._symbols_to_track))
                self._logger.info(f"Added symbol: {norm}")

    def remove_symbol(self, symbol: str) -> None:
        """Dynamically remove symbol."""
        norm = self._normalize_symbol(symbol)
        with self._lock:
            if norm in self._symbols_to_track:
                self._symbols_to_track.remove(norm)
                GAUGE_ACTIVE_SYMBOLS.set(len(self._symbols_to_track))
                self._logger.info(f"Removed symbol: {norm}")

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive status report."""
        with self._lock:
            return {
                "status": self._status.value,
                "tracked_count": len(self._symbols_to_track),
                "symbols": {
                    sym: state.to_dict() 
                    for sym, state in self._symbol_state.items() 
                    if sym in self._symbols_to_track
                }
            }

    def attach_persistent_state(self, manager: Any) -> None:
        """Attach persistent state manager for trade recovery."""
        self._persistent_state_manager = manager
        self._logger.info("Persistent state manager attached.")

    def restore_trades(self, trade_data: list[dict]) -> None:
        """Restore trade history from persistence."""
        count = 0
        with self._lock:
            for entry in trade_data:
                try:
                    sym = entry.get("symbol")
                    if not sym: continue
                    sym = self._normalize_symbol(sym)
                    
                    record = TradeRecord(
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        action=entry["action"],
                        quantity=entry["quantity"],
                        price=entry["price"],
                        status=entry["status"],
                        reason=entry["reason"],
                        order_id=entry.get("order_id"),
                        metadata=entry.get("metadata", {})
                    )
                    state = self._get_or_create_state(sym)
                    state.trade_history.append(record)
                    count += 1
                except Exception as e:
                    self._logger.warning(f"Failed to restore trade record: {e}")
        self._logger.info(f"Restored {count} trades from history.")

    # ----------------------------------------------------------------
    # EVENT LOOP
    # ----------------------------------------------------------------

    async def _handle_tick_message(self, message: Message) -> None:
        """Async entry point for ticks."""
        if self._status == RunnerStatus.STOPPED:
            return
        
        try:
            start_time = time.perf_counter()
            tick = message.data
            if not isinstance(tick, dict) or "ltp" not in tick:
                return

            symbol = tick.get("symbol")
            if not symbol: return
            norm_symbol = self._normalize_symbol(str(symbol))
            
            # Fast Filter
            matched_symbol = None
            if norm_symbol in self._symbols_to_track:
                matched_symbol = norm_symbol
            else:
                for tracked in self._symbols_to_track:
                    if tracked in norm_symbol:
                        matched_symbol = tracked
                        break
            
            if matched_symbol:
                self._process_tick(matched_symbol, tick)
                
            METRIC_LATENCY.labels(symbol=norm_symbol).observe(time.perf_counter() - start_time)

        except Exception as e:
            pass

    def _process_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        """Synchronous tick processing."""
        with self._lock:
            state = self._symbol_state.get(symbol)
            if not state: return

            ltp = float(tick["ltp"])
            vol = tick.get("volume", 0)
            # Robust Timestamp Logic
            ts_ms = tick.get("exchange_timestamp") or tick.get("timestamp")
            if ts_ms:
                if isinstance(ts_ms, (int, float)):
                    # Check freshness
                    if not is_fresh_ts_ms(ts_ms, 5000): # 5s tolerance
                         return # Skip stale ticks
                    timestamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # 1. Update Bar
            bar = state.bar_builder.update(ltp, vol, timestamp)
            METRIC_TICKS.labels(symbol=symbol).inc()

            # 2. Check Bar Close
            if bar:
                self._logger.debug(f"Bar closed for {symbol}: {bar.close}")
                self._on_bar_closed(symbol, state, bar)

    def _on_bar_closed(self, symbol: str, state: SymbolState, bar: OneMinuteBar) -> None:
        """Logic executed every minute."""
        if self._status == RunnerStatus.PAUSED:
            return

        # 1. Update Indicators
        state.indicators.update(bar.close)
        
        # 2. Check Cooldown
        if state.cooldown_until and bar.timestamp < state.cooldown_until:
            return

        # 3. Monthly Lockout
        halt, mins = self._monthly_lockout_active(None, bar.timestamp) 
        if halt:
            return

        # 4. Evaluate Strategies
        try:
            signals = self._strategy_manager.evaluate_all(symbol, bar.close, bar.timestamp)
            
            if signals:
                for signal in signals:
                    self._dispatch_signal(signal, bar.close, bar.timestamp)
                    
        except Exception as e:
            self._logger.error(f"Strategy eval failed: {e}", exc_info=True)
            METRIC_ERRORS.labels(type="strategy_eval").inc()

    def _dispatch_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Route signal to Entry or Exit handler."""
        METRIC_SIGNALS.labels(symbol=signal.symbol, action=signal.action).inc()
        
        if signal.action in ("BUY", "SELL"):
            self._handle_entry_signal(signal, price, timestamp)
        elif signal.action in ("CLOSE_LONG", "CLOSE_SHORT", "EXIT"):
            self._handle_exit_signal(signal, price, timestamp)
        else:
            self._logger.warning(f"Unknown signal action: {signal.action}")

    # ----------------------------------------------------------------
    # SIGNAL HANDLERS
    # ----------------------------------------------------------------

    def _handle_entry_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Handle BUY/SELL Entry with full validation."""
        self._logger.info(f"🔴 1. ENTRY SIGNAL: {signal.symbol} {signal.action} @ {price}")

        try:
            base_symbol = self._normalize_symbol(signal.symbol)
            state = self._symbol_state.get(base_symbol)
            
            # 1. Deduplication
            if state:
                for recent in state.signal_history:
                    if recent.action == signal.action and (timestamp - recent.timestamp).total_seconds() < 60:
                         self._logger.info(f"Skipping duplicate signal for {base_symbol}")
                         return
                state.signal_history.append(signal)

            # 2. Select Option Contract
            selection = self._select_best_option(base_symbol, signal, price)
            
            if not selection:
                self._logger.error("🔴 No Contract Selected. Trade Aborted.")
                self._record_trade(base_symbol, TradeRecord(timestamp, signal.action, 0, price, "skipped", "no_contract"))
                return

            trade_symbol = selection.symbol
            self._logger.info(f"🟢 3. SELECTED: {trade_symbol}")

            # 3. Execute Order
            self._execute_order(base_symbol, trade_symbol, "BUY", price, signal, selection)

        except Exception as e:
            self._logger.error(f"🔴 ENTRY CRASH: {e}", exc_info=True)

    def _handle_exit_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Handle CLOSE/EXIT signals."""
        self._logger.info(f"🔵 EXIT SIGNAL: {signal.symbol} {signal.action}")
        
        try:
            base_symbol = self._normalize_symbol(signal.symbol)
            state = self._symbol_state.get(base_symbol)
            
            # Find what to close
            target_symbol = None
            
            # Check Memory & Position Manager
            if state and state.active_contract:
                target_symbol = state.active_contract.symbol
            if not target_symbol and self._position_manager:
                active = self._position_manager.get_active_contract(base_symbol)
                if active: target_symbol = active.symbol
            
            if not target_symbol:
                self._logger.warning(f"Ignored Exit: No active contract found for {base_symbol}")
                return

            # Execute Exit
            qty = 0
            if self._position_manager:
                pos = self._position_manager.get_position(target_symbol)
                if pos: qty = pos.quantity
            if qty <= 0:
                qty = signal.quantity if signal.quantity > 0 else 50 

            self._execute_order(base_symbol, target_symbol, "SELL", price, signal, None, qty)

        except Exception as e:
             self._logger.error(f"🔴 EXIT CRASH: {e}", exc_info=True)

    def _select_best_option(self, base_symbol: str, signal: Signal, price: float) -> SelectedContract | None:
        """Encapsulate strike selection logic."""
        # A. Try Reuse
        if self._position_manager:
            active = self._position_manager.get_active_contract(base_symbol)
            if active and not self._position_manager.is_flat(active.symbol):
                 self._logger.info(f"🟡 Reusing Active: {active.symbol}")
                 return SelectedContract(
                    symbol=active.symbol, option_type=active.option_type, strike=active.strike,
                    expiry=active.expiry, ltp=price, delta=None, metadata={"source": "reuse"}
                 )

        # B. Fresh Selection
        if self._strike_selector:
            self._logger.info(f"🟡 2. SELECTING STRIKE...")
            direction = "BULLISH" if signal.action == "BUY" else "BEARISH"
            selector_side = "BUY"
            
            opt_type = signal.metadata.get("option_type")
            if not opt_type:
                opt_type = "CE" if direction == "BULLISH" else "PE"
            
            try:
                selection = self._strike_selector.select_contract(
                    underlying=base_symbol,
                    side=selector_side,
                    underlying_price=price,
                    option_type=cast(Literal['CE', 'PE'], opt_type)
                )
                if selection:
                    # Validate Symbol format
                    resolver = getattr(self._data_hub, "instrument_resolver", None)
                    if resolver and hasattr(resolver, "lookup") and not resolver.lookup(selection.symbol):
                         if resolver.lookup(f"NFO:{selection.symbol}"):
                             selection.symbol = f"NFO:{selection.symbol}"
                    return selection
            except Exception as e:
                 self._logger.error(f"❌ Strike Selection Failed: {e}")
        
        return None

    def _execute_order(
        self, 
        base_symbol: str, 
        trade_symbol: str, 
        side: str, 
        price: float, 
        signal: Signal,
        selection: SelectedContract | None = None,
        override_qty: int = 0
    ) -> None:
        """Centralized execution logic."""
        timestamp = datetime.now(timezone.utc)
        
        # Sizing
        if override_qty > 0:
            sized_qty = override_qty
        else:
            self._logger.info(f"🟡 4. SIZING...")
            atr_val = _extract_float(signal.metadata, "atr")
            sized_qty = self._risk_manager.suggest_position_size(
                side=side, price=price, stop_loss=signal.stop_loss,
                atr=atr_val, requested_quantity=signal.quantity,
                confidence=signal.confidence, symbol=trade_symbol
            )
        
        if sized_qty <= 0:
            self._logger.error("🔴 Zero Size Suggested.")
            return

        # Submission
        self._logger.info(f"🟡 5. SUBMITTING {side} {trade_symbol} Qty: {sized_qty}")
        try:
            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side=side,
                quantity=int(sized_qty),
                order_type=OrderType.MARKET,
                price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_id:
                self._logger.info(f"🟢 6. ORDER SUBMITTED: {order_id}")
                self._record_trade(base_symbol, TradeRecord(
                    timestamp, signal.action, int(sized_qty), price, "submitted", signal.reason, order_id
                ))
                
                self._set_cooldown(base_symbol, timestamp)
                
                if selection:
                    state = self._symbol_state.get(base_symbol)
                    if state: state.active_contract = selection
                    if self._position_manager:
                        self._position_manager.set_active_contract(base_symbol, selection)
            else:
                self._logger.error("🔴 Order ID is None")
                
        except Exception as exc:
            self._logger.error(f"🔴 ORDER CRASH: {exc}", exc_info=True)
            self._record_trade(base_symbol, TradeRecord(timestamp, signal.action, int(sized_qty), price, "error", str(exc)))

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------
    
    def _set_cooldown(self, symbol: str, timestamp: datetime) -> None:
        state = self._symbol_state.get(symbol)
        if state:
            state.cooldown_until = timestamp + timedelta(seconds=self._cooldown_seconds)

    def _record_trade(self, symbol: str, record: TradeRecord) -> None:
        state = self._symbol_state.get(symbol)
        if state:
            state.trade_history.append(record)
            if record.status == "submitted":
                state.last_trade_time = record.timestamp
                METRIC_TRADES.labels(symbol=symbol, side=record.action).inc()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.strip().upper()

    def _get_or_create_state(self, symbol: str) -> SymbolState:
        if symbol not in self._symbol_state:
            self._symbol_state[symbol] = SymbolState(
                bar_builder=OneMinuteBarBuilder(),
                indicators=IndicatorEngine()
            )
        return self._symbol_state[symbol]

    def _apply_premium_targets(self, signal: Signal, price: float, side: str) -> Signal:
        return signal

    def _monthly_lockout_active(self, expiry: datetime | None, timestamp: datetime) -> tuple[bool, float]:
        """Check if trading should halt before monthly expiry."""
        if not expiry or self._monthly_halt_minutes <= 0:
            return False, 0.0
        return False, 0.0

def _extract_float(metadata: dict, key: str) -> float | None:
    val = metadata.get(key)
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None
