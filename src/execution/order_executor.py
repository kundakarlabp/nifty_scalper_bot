"""
Order execution module.

The ``OrderExecutor`` encapsulates interaction with the Zerodha Kite API
for placing entry orders and setting up Goodâ€‘Tillâ€‘Triggered (GTT) stop
loss and target orders.  To support backtesting or dry runs, the
executor can operate in a simulated mode where it generates dummy order
IDs and records order information internally without contacting the
broker.

Trailing stop logic can be applied via the ``update_trailing_stop``
method: as the trade moves in favour of the position, the stop loss
level is moved up (for long trades) or down (for short trades) by a
multiple of the Average True Range (ATR).  This lockâ€‘in of profits is
an optional enhancement that the caller can perform periodically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import Config

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
    trailing_step: float
    is_open: bool = True


class OrderExecutor:
    """Simple wrapper around order placement and GTT management.

    If ``kite`` is ``None`` then orders are simulated and no API calls
    are made.  Otherwise the given ``KiteConnect`` instance is used for
    placing market orders and creating GTTs.  In both cases the
    ``OrderExecutor`` maintains an inâ€‘memory registry of open orders to
    support trailing stop updates and P&L computation.
    """

    def __init__(self, kite: Optional[object] = None) -> None:
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
        """Place the initial market order.

        In live mode this calls the broker API.  In simulation mode it
        generates a UUID.  Returns the broker order ID on success.
        """
        try:
            if quantity <= 0:
                logger.warning("Attempted to place order with nonâ€‘positive quantity: %s", quantity)
                return None
            if self.kite:
                # Attempt to place order via Kite Connect
                order_id = self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product=product,
                    order_type=order_type,
                    variety="regular",
                    validity=validity,
                )
                logger.info("Order placed via Kite. ID: %s", order_id)
                return order_id
            # Simulate order placement
            order_id = self._generate_order_id()
            logger.info("Simulated entry order placed. ID: %s", order_id)
            return order_id
        except Exception as exc:
            logger.error("Order placement failed: %s", exc, exc_info=True)
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
        trailing_atr: float = 0.0,
    ) -> bool:
        """Create stop loss and target orders after the entry fills.

        Returns ``True`` if the orders were registered successfully.  In
        simulation mode the order is stored internally.  The caller can
        pass an ``atr`` value to preâ€‘define the trailing stop step.  If
        omitted, trailing stops will be updated explicitly later.
        """
        try:
            # In live mode we call the GTT API once to create both stop loss and target
            if self.kite:
                try:
                    # Create a GTT type order.  Note: the actual API call may differ.
                    self.kite.place_gtt(
                        trigger_type=self.kite.GTT_TYPE_OCO,
                        tradingsymbol=symbol,
                        exchange=exchange,
                        trigger_values=[stop_loss_price, target_price],
                        last_price=entry_price,
                        orders=[
                            {
                                "transaction_type": "SELL" if transaction_type == "BUY" else "BUY",
                                "quantity": quantity,
                                "product": Config.DEFAULT_PRODUCT,
                                "order_type": Config.DEFAULT_ORDER_TYPE,
                                "price": stop_loss_price,
                            },
                            {
                                "transaction_type": "SELL" if transaction_type == "BUY" else "BUY",
                                "quantity": quantity,
                                "product": Config.DEFAULT_PRODUCT,
                                "order_type": Config.DEFAULT_ORDER_TYPE,
                                "price": target_price,
                            },
                        ],
                    )
                    logger.info("GTT orders placed via Kite for order %s", entry_order_id)
                except AttributeError:
                    logger.error("GTT placement not supported by Kite instance.")
                    return False
            # Register internally for simulation
            trailing_step = trailing_atr * Config.BASE_STOP_LOSS_POINTS if trailing_atr > 0 else 0.0
            self.orders[entry_order_id] = OrderRecord(
                order_id=entry_order_id,
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss_price,
                target=target_price,
                trailing_step=trailing_step,
            )
            return True
        except Exception as exc:
            logger.error("Failed to set up GTT orders: %s", exc, exc_info=True)
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
        """Move the stop loss closer to the current price as the trade moves in favour.

        For a long trade the stop loss is trailed up; for a short trade it is
        trailed down.  The step size is computed as a fraction of the ATR.
        Only simulated orders are affected.  Live trailing would need to
        cancel and recreate GTTs via the broker API.
        """
        order = self.orders.get(order_id)
        if not order or not order.is_open:
            return
        try:
            step = atr * Config.BASE_STOP_LOSS_POINTS
            if step <= 0:
                return
            if order.transaction_type.upper() == "BUY":
                # Trail stop up for long positions
                new_sl = max(order.stop_loss, current_price - step)
                if new_sl > order.stop_loss:
                    logger.info("Trailing stop from %.2f to %.2f for order %s", order.stop_loss, new_sl, order_id)
                    order.stop_loss = new_sl
            else:
                # Trail stop down for short positions
                new_sl = min(order.stop_loss, current_price + step)
                if new_sl < order.stop_loss:
                    logger.info("Trailing stop from %.2f to %.2f for order %s", order.stop_loss, new_sl, order_id)
                    order.stop_loss = new_sl
        except Exception as exc:
            logger.error("Error updating trailing stop: %s", exc, exc_info=True)

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """Mark an order as closed and remove it from the registry."""
        order = self.orders.get(order_id)
        if not order:
            return
        order.is_open = False
        logger.info("Order %s exited (%s)", order_id, exit_reason)
        # In a real implementation you may cancel the associated GTT here

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        """Return a dictionary of currently open orders."""
        return {oid: o for oid, o in self.orders.items() if o.is_open}