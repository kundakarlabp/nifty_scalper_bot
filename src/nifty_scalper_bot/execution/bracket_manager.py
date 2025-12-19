"""
Thread-safe Bracket Manager with VIRTUAL Execution capabilities.
Production-Grade: Replaces legacy broker-side brackets with high-speed internal monitoring.
Enhanced with ATR Trailing, Multi-Target (TP1/TP2), and Partial Scaling.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, cast, runtime_checkable

from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from nifty_scalper_bot.infra.metrics import MetricsCollector

# --------------------------------------------------------------------------
# METRICS INTEGRATION
# --------------------------------------------------------------------------
try:
    from nifty_scalper_bot.infra.metrics import METRICS as GLOBAL_METRICS
    METRICS_AVAILABLE = True
    METRICS = cast("MetricsCollector | None", GLOBAL_METRICS)
except ImportError:
    METRICS_AVAILABLE = False
    METRICS = cast("MetricsCollector | None", None)

LOGGER = get_logger(__name__)
_FILLED_STATUSES = {"FILLED", "COMPLETE", "COMPLETED"}
_CANCELLED_STATUSES = {"CANCELLED", "REJECTED", "CANCELED"}

# --------------------------------------------------------------------------
# PROTOCOLS
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsCancelOrder(Protocol):
    """Protocol representing broker cancel capability."""
    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> Any:
        ...

@runtime_checkable
class SupportsModifyOrder(Protocol):
    """Protocol representing broker order modification capability."""
    def modify_order(self, order_id: str, **kwargs: Any) -> Any:
        ...

# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class TargetLevel:
    """Represents a partial profit target."""
    price: float
    quantity: int
    executed: bool = False
    name: str = "TP"

@dataclass
class BracketState:
    """
    State container for a managed trade exit.
    Held in memory; survives restarts if persisted via TradeStore.
    """
    entry_order_id: str
    symbol: str
    side: str          # Entry Side (BUY/SELL)
    quantity: int      # Original Quantity
    entry_price: float
    
    # Execution Triggers
    sl_trigger_price: float
    tp_trigger_price: float # Ultimate TP (TP2/Final)
    
    # Multi-Target & Scaling State
    tp_levels: List[TargetLevel] = field(default_factory=list)
    remaining_quantity: int = 0
    
    # Trailing & Logic State
    is_virtual: bool = True
    active: bool = True
    trailing_enabled: bool = True
    trailing_config: Dict[str, Any] = field(default_factory=dict) # e.g., {'mode': 'ATR', 'mult': 2.0}
    
    # Market Data Tracking
    highest_ltp: float = 0.0  # High water mark since entry (for BUY)
    lowest_ltp: float = float('inf') # Low water mark since entry (for SELL)
    
    # Metadata
    tag: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
        # Initialize high/low water marks with entry price
        if self.highest_ltp == 0.0:
            self.highest_ltp = self.entry_price
        if self.lowest_ltp == float('inf'):
            self.lowest_ltp = self.entry_price


class BracketManager:
    """
    The 'Sniper' Engine.
    Monitors LTP/ATR internally and fires immediate MARKET exits when levels are hit.
    Supports TP1/TP2 scaling and ATR-based Trailing.
    """

    def __init__(self, order_manager: Any):
        self.order_manager = order_manager
        self._brackets: Dict[str, BracketState] = {}
        # Reverse Index: Map broker order IDs to entry IDs (for sync)
        self._order_to_entry: Dict[str, str] = {}
        
        # Real-time Data Cache
        self._current_atr: Dict[str, float] = {} # Symbol -> ATR Value
        
        self._lock = threading.RLock()
        self._running = True
        
        # Configuration
        self._auto_reduce_sl = True
        self._stale_cleanup_age = 86400  # 24 hours

    # --------------------------------------------------------------------------
    # 1. CORE API & REGISTRATION
    # --------------------------------------------------------------------------

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        tag: str = "auto_bracket",
        entry_order_id: str | None = None,
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None
    ) -> None:
        """
        Main Entry Point.
        Internally converts request to a Virtual Bracket with optional TP1/TP2 scaling.
        """
        if not entry_order_id:
            entry_order_id = f"virt_{int(time.time())}_{symbol}"

        self.register_virtual_bracket(
            order_id=entry_order_id,
            symbol=symbol,
            side=side,
            qty=quantity,
            price=price,
            sl=stop_loss,
            tp=take_profit,
            tag=tag,
            tp1_price=tp1_price,
            tp1_qty=tp1_qty,
            trailing_atr_mult=trailing_atr_mult
        )

    def register_virtual_bracket(
        self, 
        order_id: str, 
        symbol: str, 
        side: str, 
        qty: int, 
        price: float, 
        sl: float, 
        tp: float,
        tag: str = "virtual",
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None
    ) -> None:
        """
        Register a position for monitoring.
        Supports advanced configuration for TP1 scaling and ATR trailing.
        """
        with self._lock:
            # Deduplication
            if order_id in self._brackets:
                LOGGER.warning(f"Bracket {order_id} exists. Overwriting triggers.")
                existing = self._brackets[order_id]
                existing.sl_trigger_price = sl
                existing.tp_trigger_price = tp
                existing.quantity = qty
                existing.remaining_quantity = qty
                return

            # setup trailing config
            t_config = {}
            if trailing_atr_mult:
                t_config = {'mode': 'ATR', 'mult': trailing_atr_mult}

            # Setup TP Levels
            targets = []
            if tp1_price and tp1_qty and tp1_qty < qty:
                targets.append(TargetLevel(price=tp1_price, quantity=tp1_qty, name="TP1"))
                LOGGER.info(f"🔹 Configured TP1: {tp1_price} (Qty: {tp1_qty})")

            state = BracketState(
                entry_order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                remaining_quantity=abs(qty),
                entry_price=price,
                sl_trigger_price=sl,
                tp_trigger_price=tp, # This is effectively Final TP
                tp_levels=targets,
                is_virtual=True,
                tag=tag,
                trailing_config=t_config
            )
            
            self._brackets[order_id] = state
            self._order_to_entry[order_id] = order_id
            
            LOGGER.info(
                f"🛡️ Bracket Active for {symbol} (Qty: {qty}): "
                f"Entry={price} | SL={sl} | TP={tp} | Trail={t_config.get('mode', 'Standard')}"
            )

            if METRICS_AVAILABLE and METRICS:
                METRICS.brackets_created.inc()

    # --------------------------------------------------------------------------
    # 2. MARKET DATA INGESTION
    # --------------------------------------------------------------------------

    def update_market_stats(self, symbol: str, atr: float = 0.0, volume: float = 0.0) -> None:
        """
        Feed external calculations (ATR/Volume) into the manager for trailing logic.
        """
        with self._lock:
            if atr > 0:
                self._current_atr[symbol] = atr

    # --------------------------------------------------------------------------
    # 3. EXECUTION LOGIC (The "Sniper")
    # --------------------------------------------------------------------------

    def on_tick(self, symbol: str, ltp: float) -> None:
        """
        CRITICAL: The Heartbeat of Virtual Execution.
        Checks every incoming tick against active Virtual Levels.
        """
        if not self._brackets:
            return

        with self._lock:
            candidates = [
                b for b in self._brackets.values() 
                if b.symbol == symbol and b.active
            ]

        for bracket in candidates:
            self._process_bracket(bracket, ltp)

    def _process_bracket(self, bracket: BracketState, ltp: float) -> None:
        """Evaluate logic for a single bracket: Exits, Partials, and Trailing."""
        
        # 1. UPDATE HIGH/LOW WATER MARKS
        if bracket.side == "BUY":
            if ltp > bracket.highest_ltp:
                bracket.highest_ltp = ltp
                self._process_trailing(bracket, ltp)
        else:
            if ltp < bracket.lowest_ltp:
                bracket.lowest_ltp = ltp
                self._process_trailing(bracket, ltp)

        # 2. CHECK STOP LOSS
        if self._check_stop_loss(bracket, ltp):
            return

        # 3. CHECK PARTIAL TARGETS (TP1)
        self._check_partial_targets(bracket, ltp)

        # 4. CHECK FINAL TARGET (TP2/Final)
        self._check_final_target(bracket, ltp)

    def _check_stop_loss(self, bracket: BracketState, ltp: float) -> bool:
        """Returns True if SL hit and exit fired."""
        triggered = False
        reason = ""

        if bracket.side == "BUY":
            if ltp <= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} <= {bracket.sl_trigger_price})"
        else: # SELL
            if ltp >= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} >= {bracket.sl_trigger_price})"

        if triggered:
            LOGGER.warning(f"🛑 STOP LOSS TRIGGERED for {bracket.symbol} | {reason}")
            self._execute_exit(bracket, bracket.remaining_quantity, reason, is_partial=False)
            return True
        return False

    def _check_partial_targets(self, bracket: BracketState, ltp: float) -> None:
        """Checks TP1/Intermediate levels."""
        for target in bracket.tp_levels:
            if target.executed:
                continue
            
            triggered = False
            if bracket.side == "BUY":
                if ltp >= target.price:
                    triggered = True
            else: # SELL
                if ltp <= target.price:
                    triggered = True
            
            if triggered:
                reason = f"{target.name} Hit ({ltp})"
                qty_to_close = min(target.quantity, bracket.remaining_quantity)
                
                # Execute Partial
                success = self._execute_exit(bracket, qty_to_close, reason, is_partial=True)
                
                if success:
                    target.executed = True
                    # AUTO-ADJUST: Move SL to Breakeven after TP1
                    if target.name == "TP1":
                        self._move_sl_to_breakeven(bracket)

    def _check_final_target(self, bracket: BracketState, ltp: float) -> None:
        """Checks Final TP."""
        triggered = False
        if bracket.side == "BUY":
            if ltp >= bracket.tp_trigger_price:
                triggered = True
        else: # SELL
            if ltp <= bracket.tp_trigger_price:
                triggered = True

        if triggered:
            reason = f"FINAL TP Hit ({ltp})"
            self._execute_exit(bracket, bracket.remaining_quantity, reason, is_partial=False)

    def _process_trailing(self, bracket: BracketState, ltp: float) -> None:
        """Calculates and updates Trailing SL based on ATR or Price movement."""
        if not bracket.trailing_enabled:
            return

        t_config = bracket.trailing_config
        
        # ATR BASED TRAILING
        if t_config.get('mode') == 'ATR':
            atr = self._current_atr.get(bracket.symbol, 0.0)
            if atr <= 0:
                return # No ATR data yet
            
            mult = t_config.get('mult', 1.5)
            buffer = atr * mult
            
            with self._lock:
                if bracket.side == "BUY":
                    # SL should be High - ATR Buffer
                    potential_sl = bracket.highest_ltp - buffer
                    # ONLY Move SL UP
                    if potential_sl > bracket.sl_trigger_price:
                        bracket.sl_trigger_price = potential_sl
                        LOGGER.debug(f"📈 ATR Trail {bracket.symbol}: SL -> {potential_sl:.2f} (High: {bracket.highest_ltp})")
                
                else: # SELL
                    # SL should be Low + ATR Buffer
                    potential_sl = bracket.lowest_ltp + buffer
                    # ONLY Move SL DOWN
                    if potential_sl < bracket.sl_trigger_price:
                        bracket.sl_trigger_price = potential_sl
                        LOGGER.debug(f"📉 ATR Trail {bracket.symbol}: SL -> {potential_sl:.2f} (Low: {bracket.lowest_ltp})")

    def _move_sl_to_breakeven(self, bracket: BracketState) -> None:
        """Moves SL to Entry Price (Cost)."""
        with self._lock:
            # Add a tiny buffer for fees if needed, here we use flat entry
            if bracket.side == "BUY":
                if bracket.entry_price > bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")
            else:
                if bracket.entry_price < bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")

    # --------------------------------------------------------------------------
    # 4. EXECUTION HANDLER
    # --------------------------------------------------------------------------

    def _execute_exit(self, bracket: BracketState, qty: int, reason: str, is_partial: bool) -> bool:
        """
        Send Market Order to Broker.
        Returns True if successful.
        """
        if qty <= 0:
            return False

        with self._lock:
            # Double check if we still have quantity
            if bracket.remaining_quantity <= 0:
                bracket.active = False
                return False
            
            if not is_partial:
                bracket.active = False # Full Exit
            
            bracket.remaining_quantity -= qty

        LOGGER.info(f"⚡ EXECUTING EXIT: {bracket.symbol} | Qty: {qty} | Reason: {reason}")

        try:
            exit_side = "SELL" if bracket.side == "BUY" else "BUY"
            
            # Place Order
            self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=qty,
                order_type="MARKET", 
                tag=f"virt_exit_{bracket.tag[:5]}",
                check_risk=False,
                product="MIS"
            )
            
            # Metrics
            if METRICS_AVAILABLE and METRICS:
                METRICS.brackets_triggered.inc()
            
            # If full exit, clean up
            if not is_partial and bracket.remaining_quantity <= 0:
                self.unregister_bracket(bracket.entry_order_id)
            
            return True

        except Exception as e:
            LOGGER.critical(f"🛑 EXIT FAILED for {bracket.symbol}: {e}", exc_info=True)
            # Critical Logic: If partial failed, we keep it active. 
            # If full exit failed, we re-activate to try again next tick.
            with self._lock:
                if not is_partial:
                    bracket.active = True
                    bracket.remaining_quantity += qty # Revert deduction
            return False

    # --------------------------------------------------------------------------
    # 5. SYNC & HOUSEKEEPING
    # --------------------------------------------------------------------------

    def sync_order_status(self, broker_order_id: str, status: str, filled_qty: int) -> None:
        """
        Called by Order Update Stream.
        Detects if an Exit order initiated externally or by this manager has filled.
        """
        # If this order_id maps to a managed exit, we might need to update state.
        # But crucially: If the USER manually closes the position on the broker:
        # We need to know which symbol/side and reduce quantity.
        
        if status not in _FILLED_STATUSES:
            return

        # Simple Logic: If we are managing a symbol, and a random SELL order fills,
        # we should probably check if we need to reduce our tracked quantity.
        # Note: Implementing robust OCO matching is complex. 
        # Here we rely on the bot primarily initiating exits.
        pass

    def manual_override_close(self, symbol: str) -> None:
        """Force close all brackets for a symbol (Panic Button)."""
        with self._lock:
            targets = [b for b in self._brackets.values() if b.symbol == symbol]
            for b in targets:
                self._execute_exit(b, b.remaining_quantity, "Manual Override", is_partial=False)

    def update_trailing_sl(self, symbol: str, new_sl: float) -> None:
        """Manual/Algo external update of SL."""
        with self._lock:
            targets = [b for b in self._brackets.values() if b.symbol == symbol and b.active]
            for bracket in targets:
                updated = False
                if bracket.side == "BUY":
                    if new_sl > bracket.sl_trigger_price:
                        bracket.sl_trigger_price = new_sl
                        updated = True
                else: # SELL
                    if new_sl < bracket.sl_trigger_price:
                        bracket.sl_trigger_price = new_sl
                        updated = True
                
                if updated:
                    bracket.updated_at = time.time()
                    LOGGER.info(f"📈 Manual SL Update {symbol}: SL -> {new_sl:.2f}")

    def get_bracket(self, entry_id: str) -> Optional[BracketState]:
        with self._lock:
            return self._brackets.get(entry_id)

    def unregister_bracket(self, entry_id: str) -> None:
        """Remove a bracket from memory."""
        with self._lock:
            if entry_id in self._brackets:
                del self._brackets[entry_id]
            if entry_id in self._order_to_entry:
                del self._order_to_entry[entry_id]

    def cleanup_stale_brackets(self, max_age_seconds: int = 86400) -> int:
        """Remove old inactive brackets."""
        now = time.time()
        with self._lock:
            to_remove = [
                eid for eid, b in self._brackets.items()
                if (now - b.created_at) > max_age_seconds
            ]
            for eid in to_remove:
                self.unregister_bracket(eid)
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Diagnostic stats."""
        with self._lock:
            return {
                "active_brackets": len(self._brackets),
                "symbols_managed": len({b.symbol for b in self._brackets.values()}),
                "atr_tracked_symbols": len(self._current_atr),
            }
