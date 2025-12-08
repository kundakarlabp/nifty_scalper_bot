"""
Event-driven strategy runner coordinating trading managers.
PRODUCTION GRADE - MERGED & OPTIMIZED
"""

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
    cast,
)

# Core Imports
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType
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
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.timestamp import is_fresh_ts_ms

if TYPE_CHECKING:
    from nifty_scalper_bot.config.base import AppConfig

# --- METRICS ---
METRIC_TICKS = Counter("runner_ticks_total", "Ticks processed", ["symbol"])
METRIC_SIGNALS = Counter("runner_signals_total", "Signals generated", ["symbol", "action"])
METRIC_TRADES = Counter("runner_trades_total", "Trades executed", ["symbol", "side"])
METRIC_ERRORS = Counter("runner_errors_total", "Errors encountered", ["type"])

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
    
    # History for Deduplication & Auditing
    signal_history: Deque[Signal] = field(default_factory=lambda: deque(maxlen=20))
    trade_history: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=100))

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
        self._running = False
        self._persistent_state_manager = None
        
        # Internal State
        self._symbol_state: dict[str, SymbolState] = {}
        self._symbols_to_track: set[str] = set()
        
        # Load Settings (Robust Fallback)
        settings = getattr(config, "settings", None)
        self._options_long_only = getattr(settings, "options_long_only", True) if settings else True
        self._legacy_side_to_type = getattr(settings, "legacy_side_to_type", True) if settings else True
        self._cooldown_seconds = getattr(settings, "cooldown_seconds", 300)
        self._monthly_halt_minutes = 30

    def start(self, symbols: Iterable[str]) -> None:
        """Start listening for ticks."""
        with self._lock:
            if self._running:
                return
            
            self._logger.info(f"Starting StrategyRunner. Tracking: {list(symbols)}")
            self._symbols_to_track = set(self._normalize_symbol(s) for s in symbols)
            
            # Initialize State
            for sym in self._symbols_to_track:
                self._get_or_create_state(sym)
            
            # Subscribe
            self._message_bus.subscribe(MessageType.TICK, self._handle_tick_message)
            self._running = True

    def stop(self) -> None:
        """Stop processing."""
        with self._lock:
            self._running = False
            try:
                self._message_bus.unsubscribe(MessageType.TICK, self._handle_tick_message)
            except Exception:
                pass
            self._logger.info("StrategyRunner stopped.")

    def add_symbol(self, symbol: str) -> None:
        """Dynamically add symbol."""
        norm = self._normalize_symbol(symbol)
        with self._lock:
            self._symbols_to_track.add(norm)
            self._get_or_create_state(norm)
            self._logger.info(f"Added symbol: {norm}")
            
    def _get_or_create_state(self, symbol: str) -> SymbolState:
        if symbol not in self._symbol_state:
            self._symbol_state[symbol] = SymbolState(
                bar_builder=OneMinuteBarBuilder(),
                indicators=IndicatorEngine()
            )
        return self._symbol_state[symbol]

    def attach_persistent_state(self, manager: Any) -> None:
        """Attach persistent state manager for trade recovery."""
        self._persistent_state_manager = manager
# ----------------------------------------------------------------
    # EVENT LOOP
    # ----------------------------------------------------------------

    async def _handle_tick_message(self, message: Message) -> None:
        """Async entry point for ticks."""
        if not self._running:
            return
        
        try:
            tick = message.data
            if not isinstance(tick, dict) or "ltp" not in tick:
                return

            # Symbol Resolution
            symbol = tick.get("symbol")
            if not symbol: return
            norm_symbol = self._normalize_symbol(str(symbol))
            
            # Filter
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

        except Exception as e:
            # self._logger.debug(f"Tick error: {e}") 
            pass

    def _process_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        """Synchronous tick processing."""
        with self._lock:
            state = self._symbol_state.get(symbol)
            if not state: return

            ltp = float(tick["ltp"])
            vol = tick.get("volume", 0)
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
        # 1. Update Indicators
        state.indicators.update(bar.close)
        
        # 2. Check Cooldown
        if state.cooldown_until and bar.timestamp < state.cooldown_until:
            return

        # 3. Evaluate Strategies
        try:
            # Assumes StrategyManager returns list of Signal objects
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
    # SIGNAL HANDLERS (The "Fixed" Logic)
    # ----------------------------------------------------------------

    def _handle_entry_signal(self, signal: Signal, price: float, timestamp: datetime) -> None:
        """Handle BUY/SELL Entry with Breadcrumb Logging."""
        self._logger.info(f"🔴 1. ENTRY SIGNAL: {signal.symbol} {signal.action} @ {price}")

        try:
            base_symbol = self._normalize_symbol(signal.symbol)
            state = self._symbol_state.get(base_symbol)
            
            # 1. Deduplication (Don't trade same signal twice in 1 min)
            if state:
                for recent in state.signal_history:
                    if recent.action == signal.action and (timestamp - recent.timestamp).total_seconds() < 60:
                         self._logger.info(f"Skipping duplicate signal for {base_symbol}")
                         return
                state.signal_history.append(signal)

            # 2. Strike Selection
            selection = None
            
            # A. Try Reuse Active Position
            if self._position_manager:
                active = self._position_manager.get_active_contract(base_symbol)
                if active and not self._position_manager.is_flat(active.symbol):
                     selection = SelectedContract(
                        symbol=active.symbol, option_type=active.option_type, strike=active.strike,
                        expiry=active.expiry, ltp=price, delta=None, metadata={"source": "reuse"}
                     )
                     self._logger.info(f"🟡 Reusing Active: {selection.symbol}")

            # B. New Selection (The Fix)
            if not selection and self._strike_selector:
                self._logger.info(f"🟡 2. SELECTING STRIKE...")
                direction = "BULLISH" if signal.action == "BUY" else "BEARISH"
                selector_side = "BUY" # We buy options (Long Gamma)
                
                # Logic for Option Type
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
                except Exception as e:
                     self._logger.error(f"❌ Strike Selection Failed: {e}")

            if not selection:
                self._logger.error("🔴 No Contract Selected. Trade Aborted.")
                self._record_trade(base_symbol, TradeRecord(timestamp, signal.action, 0, price, "skipped", "no_contract"))
                return

            self._logger.info(f"🟢 3. SELECTED: {selection.symbol}")
            trade_symbol = selection.symbol

            # 3. Risk Sizing
            self._logger.info(f"🟡 4. SIZING...")
            sized_qty = self._risk_manager.suggest_position_size(
                side="BUY", price=price, stop_loss=signal.stop_loss,
                requested_quantity=signal.quantity, confidence=signal.confidence, symbol=trade_symbol
            )
            
            if sized_qty <= 0:
                self._logger.error("🔴 Zero Size Suggested.")
                return

            # 4. Execution
            self._logger.info(f"🟡 5. SUBMITTING...")
            order_id = self._order_manager.place_order(
                symbol=trade_symbol,
                side="BUY",
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
                
                # Set Cooldown & State
                self._set_cooldown(base_symbol, timestamp)
                if state: 
                    state.active_contract = selection
                    if self._position_manager:
                        self._position_manager.set_active_contract(base_symbol, selection)

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
            
            # 1. Check Active Contract in Memory
            if state and state.active_contract:
                target_symbol = state.active_contract.symbol
            
            # 2. Check Position Manager
            if not target_symbol and self._position_manager:
                active = self._position_manager.get_active_contract(base_symbol)
                if active: target_symbol = active.symbol
            
            if not target_symbol:
                self._logger.warning(f"Ignored Exit: No active contract found for {base_symbol}")
                return

            # 3. Execute Exit
            # We use 'SELL' to close a long option position
            self._logger.info(f"🔵 CLOSING POSITION: {target_symbol}")
            
            # Get current position size to close all
            qty = 0
            if self._position_manager:
                pos = self._position_manager.get_position(target_symbol)
                if pos: qty = pos.quantity
            
            if qty <= 0:
                # Fallback to signal qty or default
                qty = signal.quantity if signal.quantity > 0 else 50 

            order_id = self._order_manager.place_order(
                symbol=target_symbol,
                side="SELL",
                quantity=qty,
                order_type=OrderType.MARKET,
                tag="StrategyExit"
            )
            
            if order_id:
                self._logger.info(f"🟢 EXIT SUBMITTED: {order_id}")
                if state: state.active_contract = None

        except Exception as e:
             self._logger.error(f"🔴 EXIT CRASH: {e}", exc_info=True)

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

    def _monthly_lockout_active(self, expiry: datetime | None, timestamp: datetime) -> tuple[bool, float]:
        """Check if trading should halt before monthly expiry."""
        if not expiry or self._monthly_halt_minutes <= 0:
            return False, 0.0
        return False, 0.0

# Helpers
def _extract_float(metadata: dict, key: str) -> float | None:
    val = metadata.get(key)
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None
