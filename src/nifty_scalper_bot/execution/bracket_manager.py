"""
Thread-safe Bracket Manager with VIRTUAL Execution capabilities.
Production-Grade: Replaces legacy broker-side brackets with high-speed internal monitoring.
Enhanced with ATR Trailing, Multi-Target (TP1/TP2), Partial Scaling, and Orphan Sync.
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
    """Represents a partial profit target level."""
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
    tp_trigger_price: float  # Final/Ultimate TP
    
    # Multi-Target & Scaling State (NEW)
    remaining_quantity: int = 0
    tp_levels: List[TargetLevel] = field(default_factory=list)
    
    # Trailing & Logic State (NEW)
    is_virtual: bool = True
    active: bool = True  # If False, waits for confirmation or is finished
    trailing_enabled: bool = True
    trailing_config: Dict[str, Any] = field(default_factory=dict) # e.g. {'mode': 'ATR', 'mult': 1.5}
    
    # Market Data Tracking (NEW)
    highest_ltp: float = 0.0  # High water mark since entry (for BUY)
    lowest_ltp: float = float('inf')  # Low water mark since entry (for SELL)
    
    # Metadata
    tag: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        # Auto-initialize state fields if not set
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
        
        # Initialize High/Low water marks with entry price
        if self.highest_ltp == 0.0 or self.highest_ltp < self.entry_price:
            self.highest_ltp = self.entry_price
        
        if self.lowest_ltp == float('inf') or self.lowest_ltp > self.entry_price:
            self.lowest_ltp = self.entry_price


class BracketManager:
    """
    The 'Sniper' Engine.
    Monitors LTP/ATR internally and fires immediate MARKET exits when levels are hit.
    Supports TP1/TP2 scaling, ATR-based Trailing, and Broker Sync.
    """

    def __init__(self, order_manager: Any):
        self.order_manager = order_manager
        self._brackets: Dict[str, BracketState] = {}
        # Reverse Index: Map broker order IDs/Symbol to entry IDs
        self._order_to_entry: Dict[str, str] = {}
        self._symbol_map: Dict[str, List[str]] = {}  # Fast lookup: Symbol -> [Entry IDs]
        
        # Real-time Data Cache (NEW)
        self._current_atr: Dict[str, float] = {}
        
        self._lock = threading.RLock()
        self._running = True
        
        # Configuration
        self._auto_reduce_sl = True
        self._stale_cleanup_age = 86400  # 24 hours

    # --------------------------------------------------------------------------
    # 1. CORE API (Backward Compatible & Enhanced)
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
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None
    ) -> None:
        """
        Legacy Bridge: Allows old code to call this method.
        Internally converts request to a Virtual Bracket.
        """
        if not entry_order_id:
            # Generate a synthetic ID if caller didn't provide one
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
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
        activate_immediately: bool = True
    ) -> None:
        """Register a position for monitoring with full logic."""
        with self._lock:
            # Deduplication
            if order_id in self._brackets:
                LOGGER.warning(f"Bracket {order_id} exists. Updating triggers.")
                existing = self._brackets[order_id]
                existing.sl_trigger_price = sl
                existing.tp_trigger_price = tp
                # Reset quantity if re-registering
                existing.quantity = qty
                existing.remaining_quantity = qty
                return

            # Setup Trailing Config
            t_config = {}
            if trailing_atr_mult:
                t_config = {'mode': 'ATR', 'mult': trailing_atr_mult}
            elif self._auto_reduce_sl:
                 # Default logic if enabled but no explicit ATR
                 t_config = {'mode': 'STANDARD'}

            # Setup TP Levels (Partial Exits)
            targets = []
            if tp1_price and tp1_qty and tp1_qty < qty:
                targets.append(TargetLevel(price=tp1_price, quantity=tp1_qty, name="TP1"))
                LOGGER.info(f"🔹 Configured TP1 for {symbol}: {tp1_price} (Qty: {tp1_qty})")

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
                active=activate_immediately,
                tag=tag,
                trailing_config=t_config
            )
            
            self._brackets[order_id] = state
            
            # Populate Indices
            self._order_to_entry[order_id] = order_id
            if symbol not in self._symbol_map:
                self._symbol_map[symbol] = []
            self._symbol_map[symbol].append(order_id)
            
            trail_msg = f"| Trail={t_config.get('mode', 'None')}"
            LOGGER.info(
                f"🛡️ Bracket Active for {symbol} (Qty: {qty}): "
                f"Entry={price} | SL={sl} | TP={tp} {trail_msg}"
            )

            # Record metric
            if METRICS_AVAILABLE and METRICS:
                METRICS.brackets_created.inc()

    def confirm_entry_fill(self, order_id: str, fill_price: float = 0.0) -> None:
        """Called externally when the entry order fills."""
        with self._lock:
            bracket = self._brackets.get(order_id)
            if bracket:
                if not bracket.active:
                    bracket.active = True
                    LOGGER.info(f"✅ Bracket ACTIVATED for {bracket.symbol} (Order {order_id})")
                
                if fill_price > 0:
                    bracket.entry_price = fill_price
                    # Reset high/low water marks to fill price
                    bracket.highest_ltp = fill_price
                    bracket.lowest_ltp = fill_price

    # --------------------------------------------------------------------------
    # 2. MARKET DATA INGESTION (NEW)
    # --------------------------------------------------------------------------

    def update_market_stats(self, symbol: str, atr: float = 0.0, volume: float = 0.0) -> None:
        """Feed external calculations (ATR) into the manager."""
        if atr > 0:
            # No lock needed for simple dict assignment in Python (atomic-ish)
            self._current_atr[symbol] = atr

    # --------------------------------------------------------------------------
    # 3. EXECUTION LOGIC (The "Sniper")
    # --------------------------------------------------------------------------

    def on_tick(self, symbol: str, ltp: float) -> None:
        """
        CRITICAL: The Heartbeat of Virtual Execution.
        Checks every incoming tick against active Virtual Levels.
        """
        # Fast check without lock
        if not self._brackets:
            return

        # Fast Lookup Strategy
        relevant_ids = self._symbol_map.get(symbol)
        if not relevant_ids:
            return

        # Snapshot active brackets for this symbol safely
        with self._lock:
            candidates = []
            for eid in relevant_ids:
                b = self._brackets.get(eid)
                if b and b.active:
                    candidates.append(b)

        # Process without holding lock for too long (logic only)
        for bracket in candidates:
            self._check_and_fire(bracket, ltp)

    def _check_and_fire(self, bracket: BracketState, ltp: float) -> None:
        """Evaluate logic for a single bracket: Exits, Partials, and Trailing."""
        
        # 1. UPDATE TRAILING STATE (High/Low Water Marks)
        self._process_trailing_logic(bracket, ltp)

        # 2. STOP LOSS CHECK
        if self._check_stop_loss(bracket, ltp):
            return  # SL hit, stop processing

        # 3. PARTIAL TARGETS (TP1)
        self._check_partial_targets(bracket, ltp)

        # 4. FINAL TARGET (TP2)
        self._check_final_target(bracket, ltp)

    def _process_trailing_logic(self, bracket: BracketState, ltp: float) -> None:
        """Updates High/Low marks and adjusts SL if Trailing is enabled."""
        # A. Update Water Marks
        if bracket.side == "BUY":
            if ltp > bracket.highest_ltp:
                bracket.highest_ltp = ltp
                self._apply_trailing_math(bracket)
        else: # SELL
            if ltp < bracket.lowest_ltp:
                bracket.lowest_ltp = ltp
                self._apply_trailing_math(bracket)

    def _apply_trailing_math(self, bracket: BracketState) -> None:
        """Calculates new SL based on ATR or Fixed points."""
        if not bracket.trailing_config:
            return
            
        mode = bracket.trailing_config.get('mode')
        
        # ATR Trailing
        if mode == 'ATR':
            atr = self._current_atr.get(bracket.symbol, 0.0)
            if atr <= 0: return # No ATR available yet
            
            mult = bracket.trailing_config.get('mult', 1.5)
            buffer = atr * mult
            
            with self._lock:
                if bracket.side == "BUY":
                    potential_sl = bracket.highest_ltp - buffer
                    # Ratchet UP only
                    if potential_sl > bracket.sl_trigger_price:
                        bracket.sl_trigger_price = potential_sl
                        LOGGER.debug(f"📈 ATR Trail {bracket.symbol}: SL -> {potential_sl:.2f}")
                else: # SELL
                    potential_sl = bracket.lowest_ltp + buffer
                    # Ratchet DOWN only
                    if potential_sl < bracket.sl_trigger_price:
                        bracket.sl_trigger_price = potential_sl
                        LOGGER.debug(f"📉 ATR Trail {bracket.symbol}: SL -> {potential_sl:.2f}")

    def _check_stop_loss(self, bracket: BracketState, ltp: float) -> bool:
        """Returns True if SL hit and exit fired."""
        triggered = False
        reason = ""

        if bracket.side == "BUY":
            if ltp <= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} <= {bracket.sl_trigger_price:.2f})"
        else: # SELL
            if ltp >= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} >= {bracket.sl_trigger_price:.2f})"

        if triggered:
            LOGGER.warning(f"🛑 STOP LOSS TRIGGERED for {bracket.symbol} | {reason}")
            # Exit full remaining quantity
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

    def _move_sl_to_breakeven(self, bracket: BracketState) -> None:
        """Moves SL to Entry Price (Cost)."""
        with self._lock:
            if bracket.side == "BUY":
                if bracket.entry_price > bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")
            else:
                if bracket.entry_price < bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")

    def _execute_exit(self, bracket: BracketState, qty: int, reason: str, is_partial: bool) -> bool:
        """
        Send Market Order to Broker.
        Returns True if successful.
        """
        if qty <= 0:
            return False

        with self._lock:
            # Double check if we still have quantity
            if bracket.remaining_quantity <= 0 or not bracket.active:
                return False
            
            # State Update *Before* Order to prevent double firing
            bracket.remaining_quantity -= qty
            
            if not is_partial or bracket.remaining_quantity <= 0:
                bracket.active = False # Deactivate monitoring
            
        LOGGER.warning(f"⚡ EXECUTING EXIT: {bracket.symbol} | Qty: {qty} | Reason: {reason}")

        try:
            exit_side = "SELL" if bracket.side == "BUY" else "BUY"
            
            # Use OrderManager to place immediate market order
            self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=qty,
                order_type="MARKET", 
                tag=f"virt_exit_{bracket.tag[:5]}",
                check_risk=False, # Force exit
                product="MIS"
            )
            
            # Metrics
            if METRICS_AVAILABLE and METRICS:
                METRICS.brackets_triggered.inc()
                
            # Cleanup if full exit
            if not is_partial and bracket.remaining_quantity <= 0:
                self.unregister_bracket(bracket.entry_order_id)

            return True

        except Exception as e:
            LOGGER.critical(f"🛑 EXIT FAILED for {bracket.symbol}: {e}", exc_info=True)
            # Revert State on Failure
            with self._lock:
                bracket.remaining_quantity += qty
                if not is_partial:
                    bracket.active = True 
            return False

    # --------------------------------------------------------------------------
    # 4. SYNC & MANUAL INTERVENTION (World Class)
    # --------------------------------------------------------------------------

    def sync_manual_exit(self, symbol: str, quantity_left: int) -> None:
        """
        Called by OrderManager/PositionManager when position size changes externally.
        ORPHAN HANDLING: If qty goes to 0, kill all brackets for symbol.
        """
        if quantity_left <= 0:
            self.manual_override_close(symbol, reason="External/Manual Exit Detected")
        else:
            # Logic for partial manual exit can be added here
            # For now, we assume if some qty remains, we keep brackets active
            pass

    def manual_override_close(self, symbol: str, reason: str = "Manual Override") -> None:
        """Force close/remove all brackets for a symbol."""
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids:
                return
            
            count = 0
            for eid in list(relevant_ids):
                if eid in self._brackets:
                    # We strictly unregister, assuming the position is already gone/closing
                    self.unregister_bracket(eid)
                    count += 1
            
            if count > 0:
                LOGGER.info(f"🧹 Cleaned up {count} brackets for {symbol} due to: {reason}")

    def sync_order_status(self, broker_order_id: str, status: str, filled_qty: int) -> None:
        """
        Detects if an Exit order initiated externally has filled.
        Used to keep internal state consistent.
        """
        if status not in _FILLED_STATUSES:
            return
            
        # This is a hook for future expansion where we map every broker order back to a bracket.
        # Currently handled via sync_manual_exit based on net position.
        pass

    # --------------------------------------------------------------------------
    # 5. DYNAMIC UPDATES (Trailing & Utils)
    # --------------------------------------------------------------------------

    def update_trailing_sl(self, symbol: str, new_sl: float) -> None:
        """Update SL price manually for all active brackets on a symbol."""
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids: return

            for eid in relevant_ids:
                bracket = self._brackets.get(eid)
                if not bracket or not bracket.active: continue
                
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

    # --------------------------------------------------------------------------
    # 6. HOUSEKEEPING & UTILS
    # --------------------------------------------------------------------------

    def is_symbol_managed(self, symbol: str) -> bool:
        """
        Check if symbol has active bracket protection.
        Used during position reconciliation to avoid duplicate brackets.
        """
        with self._lock:
            # 1. Fast check: symbol not in tracking map
            if symbol not in self._symbol_map:
                return False
            
            # 2. Deep check: Are any linked brackets actually active?
            # We iterate through all order IDs associated with this symbol
            entry_ids = self._symbol_map.get(symbol, [])
            for eid in entry_ids:
                bracket = self._brackets.get(eid)
                # It's managed if at least one bracket is Active and has Quantity remaining
                if bracket and bracket.active and bracket.remaining_quantity > 0:
                    return True
            
            return False
            
    def get_bracket(self, entry_id: str) -> Optional[BracketState]:
        with self._lock:
            return self._brackets.get(entry_id)

    def unregister_bracket(self, entry_id: str) -> None:
        """Remove a bracket from memory and indices."""
        with self._lock:
            if entry_id in self._brackets:
                bracket = self._brackets[entry_id]
                symbol = bracket.symbol
                
                # Cleanup Main Dict
                del self._brackets[entry_id]
                
                # Cleanup Symbol Map
                if symbol in self._symbol_map:
                    if entry_id in self._symbol_map[symbol]:
                        self._symbol_map[symbol].remove(entry_id)
                    if not self._symbol_map[symbol]:
                        del self._symbol_map[symbol]

            # Cleanup reverse index
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
            
            if to_remove:
                LOGGER.info(f"🧹 Cleaned up {len(to_remove)} stale brackets.")
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Diagnostic stats."""
        with self._lock:
            return {
                "active_brackets": len(self._brackets),
                "symbols_managed": len(self._symbol_map),
                "atr_tracked_symbols": len(self._current_atr),
            }
