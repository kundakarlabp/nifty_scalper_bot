"""
Thread-safe Bracket Manager with VIRTUAL Execution capabilities.
Production-Grade: Replaces legacy broker-side brackets with high-speed internal monitoring.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, cast, runtime_checkable

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


# --------------------------------------------------------------------------
# ✅ CRITICAL RESTORE: Protocols required by app.py
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


@dataclass
class BracketState:
    """
    State container for a managed trade exit.
    Held in memory; survives restarts if persisted via TradeStore.
    """
    entry_order_id: str
    symbol: str
    side: str          # Entry Side (BUY/SELL)
    quantity: int
    entry_price: float
    
    # Execution Triggers
    sl_trigger_price: float
    tp_trigger_price: float
    
    # State flags
    is_virtual: bool = True
    active: bool = True
    trailing_enabled: bool = True
    
    # Metadata
    tag: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class BracketManager:
    """
    The 'Sniper' Engine.
    Monitors LTP internally and fires immediate MARKET exits when levels are hit.
    """

    def __init__(self, order_manager: Any):
        self.order_manager = order_manager
        self._brackets: Dict[str, BracketState] = {}
        self._lock = threading.RLock()
        self._running = True
        
        # Configuration
        self._auto_reduce_sl = True
        self._stale_cleanup_age = 86400  # 24 hours

    # --------------------------------------------------------------------------
    # 1. CORE API (Backward Compatible)
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
        entry_order_id: str | None = None
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
            tag=tag
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
        tag: str = "virtual"
    ) -> None:
        """Register a position for monitoring."""
        with self._lock:
            # Deduplication
            if order_id in self._brackets:
                LOGGER.warning(f"Bracket {order_id} exists. Overwriting triggers.")
                existing = self._brackets[order_id]
                existing.sl_trigger_price = sl
                existing.tp_trigger_price = tp
                return

            state = BracketState(
                entry_order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                entry_price=price,
                sl_trigger_price=sl,
                tp_trigger_price=tp,
                is_virtual=True,
                tag=tag
            )
            self._brackets[order_id] = state
            
            LOGGER.info(
                f"🛡️ Bracket Active for {symbol} (Qty: {qty}): "
                f"Entry={price} | SL={sl} | TP={tp} (Virtual Mode)"
            )

    # --------------------------------------------------------------------------
    # 2. EXECUTION LOGIC (The "Sniper")
    # --------------------------------------------------------------------------

    def on_tick(self, symbol: str, ltp: float) -> None:
        """
        CRITICAL: The Heartbeat of Virtual Execution.
        Checks every incoming tick against active Virtual Levels.
        """
        # Fast check without lock
        if not self._brackets:
            return

        # Snapshot active brackets for this symbol
        with self._lock:
            candidates = [
                b for b in self._brackets.values() 
                if b.symbol == symbol and b.active
            ]

        for bracket in candidates:
            self._check_and_fire(bracket, ltp)

    def _check_and_fire(self, bracket: BracketState, ltp: float) -> None:
        """Evaluate logic for a single bracket."""
        fire_exit = False
        reason = ""

        # 1. STOP LOSS CHECK
        if bracket.side == "BUY":  # Long Position
            if ltp <= bracket.sl_trigger_price:
                fire_exit = True
                reason = f"SL Hit ({ltp} <= {bracket.sl_trigger_price})"
        else:  # Short Position
            if ltp >= bracket.sl_trigger_price:
                fire_exit = True
                reason = f"SL Hit ({ltp} >= {bracket.sl_trigger_price})"

        # 2. TAKE PROFIT CHECK
        if not fire_exit and bracket.tp_trigger_price > 0:
            if bracket.side == "BUY":
                if ltp >= bracket.tp_trigger_price:
                    fire_exit = True
                    reason = f"TP Hit ({ltp} >= {bracket.tp_trigger_price})"
            else:
                if ltp <= bracket.tp_trigger_price:
                    fire_exit = True
                    reason = f"TP Hit ({ltp} <= {bracket.tp_trigger_price})"

        # 3. EXECUTE
        if fire_exit:
            self._execute_exit(bracket, reason)

    def _execute_exit(self, bracket: BracketState, reason: str) -> None:
        """Send Market Order to Broker."""
        with self._lock:
            if not bracket.active: 
                return # Already fired
            bracket.active = False

        LOGGER.warning(f"⚡ EXECUTING EXIT: {bracket.symbol} | Reason: {reason}")

        try:
            exit_side = "SELL" if bracket.side == "BUY" else "BUY"
            
            # Use OrderManager to place immediate market order
            # Note: We skip risk checks for exits to ensure they go through
            self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=bracket.quantity,
                order_type="MARKET", 
                tag=f"virt_exit_{bracket.tag[:5]}",
                check_risk=False,
                product="MIS"
            )
            
            # Cleanup
            self.unregister_bracket(bracket.entry_order_id)

        except Exception as e:
            LOGGER.critical(f"🛑 EXIT FAILED for {bracket.symbol}: {e}", exc_info=True)
            # Re-activate to retry on next tick?
            with self._lock:
                bracket.active = True 

    # --------------------------------------------------------------------------
    # 3. DYNAMIC UPDATES (Trailing)
    # --------------------------------------------------------------------------

    def update_trailing_sl(self, symbol: str, new_sl: float) -> None:
        """Update SL price for all active brackets on a symbol."""
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
                    LOGGER.info(f"📈 Trailing Update {symbol}: SL -> {new_sl:.2f}")

    # --------------------------------------------------------------------------
    # 4. HOUSEKEEPING & UTILS
    # --------------------------------------------------------------------------

    def get_bracket(self, entry_id: str) -> Optional[BracketState]:
        with self._lock:
            return self._brackets.get(entry_id)

    def unregister_bracket(self, entry_id: str) -> None:
        """Remove a bracket from memory."""
        with self._lock:
            if entry_id in self._brackets:
                del self._brackets[entry_id]

    def cleanup_stale_brackets(self, max_age_seconds: int = 86400) -> int:
        """Remove old inactive brackets."""
        now = time.time()
        with self._lock:
            to_remove = [
                eid for eid, b in self._brackets.items()
                if (now - b.created_at) > max_age_seconds
            ]
            for eid in to_remove:
                del self._brackets[eid]
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Diagnostic stats."""
        with self._lock:
            return {
                "active_brackets": len(self._brackets),
                "symbols_managed": len({b.symbol for b in self._brackets.values()}),
            }
