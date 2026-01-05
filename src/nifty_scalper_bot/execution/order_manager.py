
import logging
import time
from typing import Dict, Optional, List, Any
import threading

# Configure logging
logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages the lifecycle of orders, tracks their status, and handles order updates.
    """
    def __init__(self, trading_client):
        self.trading_client = trading_client
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.logger = logger
        self.active_orders: Dict[str, Dict[str, Any]] = {}  # Orders that are open/pending

    def register_order(self, order_details: Dict[str, Any]) -> None:
        """
        Register a new order to be tracked.
        """
        with self.lock:
            order_id = str(order_details.get('order_id', ''))
            if not order_id:
                self.logger.error("Attempted to register order without ID")
                return

            self.orders[order_id] = order_details
            
            # If the order is active, add to active_orders
            status = order_details.get('status', '').upper()
            if status in ['OPEN', 'PENDING', 'TRIGGER PENDING', 'GMO']:
                self.active_orders[order_id] = order_details
                
            self.logger.info(f"Registered order {order_id} for {order_details.get('tradingsymbol', 'UNKNOWN')}")

    def update_order_status(self, order_id: str, status: str, fill_price: float = None, payload: dict = None) -> None:
        """
        Update the status of an existing order or adopt an orphan order.
        """
        with self.lock:
            order_id = str(order_id)
            order = self.orders.get(order_id)
            
            # Handle Orphan Orders - ADOPTION LOGIC
            if order is None and payload:
                self.logger.warning(f"⚠️ Adopting orphan order {order_id} from broker update")
                try:
                    adopted_order = self._parse_broker_order(payload)
                    if adopted_order:
                        self.register_order(adopted_order)
                        order = adopted_order
                        self.logger.info(f"✅ Successfully adopted orphan order {order_id}")
                    else:
                        self.logger.error(f"❌ Failed to parse orphan order {order_id}")
                except Exception as e:
                    self.logger.error(f"❌ Error adopting orphan order {order_id}: {e}")
            
            if order is None:
                self.logger.warning(f"⚠️ Cannot update unknown order {order_id} (No payload provided for adoption)")
                return

            # Update status
            previous_status = order.get('status')
            order['status'] = status
            order['last_update_time'] = time.time()
            
            if fill_price is not None:
                order['average_price'] = fill_price
                
            # Update active orders list
            status_upper = status.upper()
            if status_upper in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
            elif status_upper in ['OPEN', 'PENDING', 'TRIGGER PENDING']:
                self.active_orders[order_id] = order
                
            self.logger.info(f"Order {order_id} status updated: {previous_status} -> {status}")

    def _parse_broker_order(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse raw broker payload into standard order dictionary.
        This handles Zerodha/Kite Connect specific fields.
        """
        try:
            # Check if this is a standard Kite Connect order structure
            if 'order_id' in payload and 'tradingsymbol' in payload:
                return {
                    'order_id': str(payload.get('order_id')),
                    'parent_order_id': str(payload.get('parent_order_id')) if payload.get('parent_order_id') else None,
                    'exchange_order_id': payload.get('exchange_order_id'),
                    'placed_by': payload.get('placed_by'),
                    'variety': payload.get('variety'),
                    'status': payload.get('status'),
                    'tradingsymbol': payload.get('tradingsymbol'),
                    'exchange': payload.get('exchange'),
                    'instrument_token': payload.get('instrument_token'),
                    'transaction_type': payload.get('transaction_type'),
                    'order_type': payload.get('order_type'),
                    'product': payload.get('product'),
                    'validity': payload.get('validity'),
                    'price': payload.get('price', 0.0),
                    'quantity': payload.get('quantity', 0),
                    'trigger_price': payload.get('trigger_price', 0.0),
                    'average_price': payload.get('average_price', 0.0),
                    'filled_quantity': payload.get('filled_quantity', 0),
                    'pending_quantity': payload.get('pending_quantity', 0),
                    'cancelled_quantity': payload.get('cancelled_quantity', 0),
                    'order_timestamp': payload.get('order_timestamp'),
                    'tag': payload.get('tag')
                }
            return None
        except Exception as e:
            self.logger.error(f"Error parsing broker order: {e}")
            return None

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.orders.get(str(order_id))
            
    def get_active_orders(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.active_orders.values())
            
    def get_all_orders(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.orders.values())
