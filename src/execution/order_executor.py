# src/execution/order_executor.py
"""
Order execution module.

The `OrderExecutor` encapsulates interaction with the Zerodha Kite API
for placing entry orders and setting up Good-Till-Triggered (GTT) stop
loss and target orders. To support backtesting or dry runs, the
executor can operate in a simulated mode where it generates dummy order
IDs and records order information internally without contacting the
broker.

Trailing stop logic can be applied via the `update_trailing_stop`
method: as the trade moves in favour of the position, the stop loss
level is moved up (for long trades) or down (for short trades) by a
multiple of the Average True Range (ATR). This lock-in of profits is
an optional enhancement that the caller can perform periodically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

# ✅ Corrected import path based on typical project structure
from src.config import Config

logger = logging.getLogger(__name__)

@dataclass
class OrderRecord:
    """Internal representation of an open trade."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    # ✅ Use ATR multiplier for trailing step if ATR is provided
    trailing_step_atr_multiplier: float # Multiplier for ATR to get step size
    is_open: bool = True

class OrderExecutor:
    """Simple wrapper around order placement and GTT management.

    If `kite` is `None` then orders are simulated and no API calls
    are made. Otherwise the given `KiteConnect` instance is used for
    placing market orders and creating GTTs.
    """

    def __init__(self, kite: Optional[Any] = None) -> None: # Use Any for kite object type
        self.kite = kite
        self.orders: Dict[str, OrderRecord] = {}

    def _generate_order_id(self) -> str:
        """Generate a unique identifier for a simulated order."""
        return str(uuid.uuid4())

    def place_entry_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = Config.DEFAULT_PRODUCT,
        order_type: str = Config.DEFAULT_ORDER_TYPE,
        validity: str = Config.DEFAULT_VALIDITY,
    ) -> Optional[str]:
        """Place the initial market order."""
        try:
            if quantity <= 0:
                logger.warning("Attempted to place order with non-positive quantity: %s", quantity)
                return None
            if self.kite:
                # Place order via Kite API
                order_id = self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product=product,
                    order_type=order_type,
                    variety="regular", # Standard order variety
                    validity=validity,
                )
                logger.info("✅ Order placed via Kite. ID: %s, Symbol: %s, Qty: %d, Type: %s",
                            order_id, symbol, quantity, transaction_type)
                return order_id
            else:
                # Simulate order placement
                order_id = self._generate_order_id()
                logger.info("🧪 Simulated entry order placed. ID: %s, Symbol: %s, Qty: %d, Type: %s",
                            order_id, symbol, quantity, transaction_type)
                return order_id
        except Exception as exc:
            logger.error("💥 Order placement failed for %s: %s", symbol, exc, exc_info=True)
            return None

    def setup_gtt_orders(
        self,
        entry_order_id: str,
        entry_price: float,
        stop_loss_price: float,
        target_price: float,
        symbol: str,
        exchange: str,
        quantity: int,
        transaction_type: str,
        # ✅ Clarify trailing_atr usage or remove if not standard
        # Assuming trailing_atr is the step size derived from ATR and a multiplier
        # If you want to derive step from ATR here, pass ATR and use Config.ATR_SL_MULTIPLIER
        # For now, let's assume trailing_step_atr_multiplier is passed or derived elsewhere
        # Let's remove trailing_atr from args for clarity based on OrderRecord
        # trailing_atr: float = 0.0,
    ) -> bool:
        """Create stop loss and target orders (GTT OCO) after the entry fills."""
        try:
            # Determine opposite transaction type for exit orders
            exit_transaction_type = "SELL" if transaction_type.upper() == "BUY" else "BUY"

            if self.kite:
                try:
                    # Prepare GTT order legs
                    # Note: price is typically for LIMIT orders within GTT.
                    # For MARKET, Kite might use trigger_values directly.
                    # Check KiteConnect documentation for exact GTT OCO format.
                    gtt_orders: List[Dict[str, Any]] = [
                        {
                            "transaction_type": exit_transaction_type,
                            "quantity": quantity,
                            "product": Config.DEFAULT_PRODUCT,
                            "order_type": Config.DEFAULT_ORDER_TYPE,
                            # Price might be needed for LIMIT, optional/ignored for MARKET GTTs
                            "price": stop_loss_price,
                        },
                        {
                            "transaction_type": exit_transaction_type,
                            "quantity": quantity,
                            "product": Config.DEFAULT_PRODUCT,
                            "order_type": Config.DEFAULT_ORDER_TYPE,
                            # Price might be needed for LIMIT, optional/ignored for MARKET GTTs
                            "price": target_price,
                        },
                    ]

                    # Place GTT OCO order
                    gtt_result = self.kite.place_gtt(
                        trigger_type=self.kite.GTT_TYPE_OCO,
                        tradingsymbol=symbol,
                        exchange=exchange,
                        trigger_values=[stop_loss_price, target_price], # Key trigger prices
                        last_price=entry_price, # Last traded price for validation
                        orders=gtt_orders,
                    )
                    logger.info("✅ GTT OCO orders placed via Kite for entry order %s. GTT ID: %s",
                                entry_order_id, gtt_result.get('trigger_id', 'N/A'))
                except AttributeError:
                    logger.error("❌ GTT placement not supported by this Kite instance or method not found.")
                    return False
                except Exception as gtt_exc:
                     logger.error("💥 GTT placement failed for order %s: %s", entry_order_id, gtt_exc, exc_info=True)
                     return False # Consider this a failure to setup GTT

            # --- Internal Record Keeping (for simulation or tracking) ---
            # Calculate trailing step multiplier (assuming it's derived from ATR and config)
            # Placeholder: If ATR was passed, you'd use it here. For now, assume 0 or derived elsewhere.
            # Let's assume trailing logic uses Config.ATR_SL_MULTIPLIER if ATR data becomes available later.
            # For record keeping, store the multiplier used conceptually.
            # If trailing_atr was passed as step size, you could store it.
            # For simplicity, storing the multiplier concept.
            trailing_step_multiplier = getattr(Config, 'ATR_SL_MULTIPLIER', 1.5) # Default fallback

            # Store the order details internally
            self.orders[entry_order_id] = OrderRecord(
                order_id=entry_order_id,
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss_price,
                target=target_price,
                trailing_step_atr_multiplier=trailing_step_multiplier, # Store for future trailing updates
            )
            logger.debug("📝 Internal order record created for %s", entry_order_id)
            return True
        except Exception as exc:
            logger.error("💥 Failed to set up internal GTT tracking or simulation for order %s: %s", entry_order_id, exc, exc_info=True)
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
        """Update stop loss based on favorable movement using ATR."""
        order = self.orders.get(order_id)
        if not order or not order.is_open:
            logger.debug("📉 Trailing stop update skipped: Order %s not found or closed.", order_id)
            return
        try:
            # ✅ Use Config.ATR_SL_MULTIPLIER to calculate step size from ATR
            step_size = atr * order.trailing_step_atr_multiplier # Use the multiplier stored or from config
            if step_size <= 0:
                logger.debug("📉 Trailing stop update skipped: Calculated step size <= 0 for order %s", order_id)
                return

            new_sl = order.stop_loss
            if order.transaction_type.upper() == "BUY":
                # For long position, move SL up
                potential_new_sl = current_price - step_size
                if potential_new_sl > order.stop_loss:
                    # Optional: Add minimum move threshold check here
                    # if potential_new_sl - order.stop_loss >= min_move_threshold:
                    new_sl = potential_new_sl
            elif order.transaction_type.upper() == "SELL":
                 # For short position, move SL down
                potential_new_sl = current_price + step_size
                if potential_new_sl < order.stop_loss:
                    # Optional: Add minimum move threshold check here
                    # if order.stop_loss - potential_new_sl >= min_move_threshold:
                    new_sl = potential_new_sl

            # Update SL if it changed
            if new_sl != order.stop_loss:
                 logger.info("↗️ Trailing stop updated for %s (BUY): %.2f -> %.2f (Current Price: %.2f, ATR: %.2f)",
                             order_id, order.stop_loss, new_sl, current_price, atr) if order.transaction_type.upper() == "BUY" else \
                 logger.info("↘️ Trailing stop updated for %s (SELL): %.2f -> %.2f (Current Price: %.2f, ATR: %.2f)",
                             order_id, order.stop_loss, new_sl, current_price, atr)
                 order.stop_loss = new_sl
            else:
                 logger.debug("➡️ Trailing stop checked for %s, no update needed. Current SL: %.2f", order_id, order.stop_loss)

        except Exception as exc:
            logger.error("💥 Error updating trailing stop for order %s: %s", order_id, exc, exc_info=True)

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """Mark an order as closed and remove it from the active registry."""
        order = self.orders.get(order_id)
        if not order:
            logger.warning("⚠️ Attempted to exit non-existent order: %s", order_id)
            return
        if not order.is_open:
             logger.debug("ℹ️ Order %s already marked as closed.", order_id)
             return

        order.is_open = False
        logger.info("⏹️ Order %s marked as exited (%s). Symbol: %s, Qty: %d",
                    order_id, exit_reason, order.symbol, order.quantity)
        # Note: In live mode, this function currently only updates internal state.
        # To fully exit a live position, you would need to:
        # 1. Cancel the existing GTT order associated with this trade (requires GTT ID).
        # 2. Place an exit market order.
        # This requires storing GTT IDs and interacting with more Kite APIs.
        # This simplified version focuses on internal tracking.

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        """Return a dictionary of currently open orders."""
        active = {oid: o for oid, o in self.orders.items() if o.is_open}
        logger.debug("📊 Retrieved %d active orders.", len(active))
        return active
