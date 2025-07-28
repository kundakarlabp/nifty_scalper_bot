import logging
import time
from typing import Dict, List, Optional
from kiteconnect import KiteConnect
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN

logger = logging.getLogger(__name__)

class OrderExecutor:
    def __init__(self, kite_connect: KiteConnect = None):
        self.kite = kite_connect or self._initialize_kite()
        self.active_orders = {}
        self.order_history = []
        self.max_positions = 1
        self.current_positions = 0
        
    def _initialize_kite(self) -> KiteConnect:
        """Initialize Kite Connect with credentials"""
        try:
            kite = KiteConnect(api_key=ZERODHA_API_KEY)
            kite.set_access_token(ZERODHA_ACCESS_TOKEN)
            logger.info("✅ Kite Connect initialized for order execution")
            return kite
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kite Connect: {e}")
            raise
    
    def place_order(self, order_params: Dict) -> Optional[str]:
        """Place an order with given parameters"""
        try:
            # Validate order parameters
            required_fields = ['tradingsymbol', 'transaction_type', 'quantity', 'product', 'order_type']
            for field in required_fields:
                if field not in order_params:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check position limits
            if self.current_positions >= self.max_positions:
                logger.warning("⚠️  Maximum positions limit reached")
                return None
            
            # Place the order
            order_id = self.kite.place_order(
                variety=order_params.get('variety', self.kite.VARIETY_REGULAR),
                exchange=order_params.get('exchange', self.kite.EXCHANGE_NSE),
                tradingsymbol=order_params['tradingsymbol'],
                transaction_type=order_params['transaction_type'],
                quantity=order_params['quantity'],
                product=order_params['product'],
                order_type=order_params['order_type'],
                price=order_params.get('price'),
                trigger_price=order_params.get('trigger_price'),
                validity=order_params.get('validity', self.kite.VALIDITY_DAY),
                disclosed_quantity=order_params.get('disclosed_quantity', 0)
            )
            
            # Store order information
            order_info = {
                'order_id': order_id,
                'params': order_params,
                'status': 'placed',
                'timestamp': time.time(),
                'filled_quantity': 0,
                'average_price': 0
            }
            
            self.active_orders[order_id] = order_info
            self.current_positions += 1
            
            logger.info(f"✅ Order placed successfully. Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return None
    
    def place_nifty_order(self, signal: Dict, position_info: Dict) -> Optional[str]:
        """Place order for Nifty based on signal"""
        try:
            transaction_type = self.kite.TRANSACTION_TYPE_BUY if signal['signal'] == 'BUY' else self.kite.TRANSACTION_TYPE_SELL
            
            order_params = {
                'tradingsymbol': 'NIFTY 50',
                'transaction_type': transaction_type,
                'quantity': position_info['quantity'],
                'product': self.kite.PRODUCT_MIS,
                'order_type': self.kite.ORDER_TYPE_MARKET,
                'variety': self.kite.VARIETY_REGULAR,
                'exchange': self.kite.EXCHANGE_NSE
            }
            
            # For limit orders, you can use:
            # 'order_type': self.kite.ORDER_TYPE_LIMIT,
            # 'price': signal['entry_price']
            
            order_id = self.place_order(order_params)
            
            if order_id:
                logger.info(f"✅ Nifty order placed: {signal['signal']} {position_info['quantity']} qty")
                
                # Send Telegram alert
                self._send_order_alert(signal, position_info, order_id)
            
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Error placing Nifty order: {e}")
            return None
    
    def _send_order_alert(self, signal: Dict, position_info: Dict, order_id: str):
        """Send order execution alert via Telegram"""
        try:
            from src.notifications.telegram_bot import telegram_bot
            
            order_message = f"""
            �� TRADE EXECUTED
            
            📊 Symbol: NIFTY 50
            📈 Direction: {signal['signal']}
            💰 Entry Price: {signal['entry_price']:.2f}
            🛑 Stop Loss: {signal['stop_loss']:.2f}
            ✅ Target: {signal['target']:.2f}
            📦 Quantity: {position_info['quantity']} ({position_info['lots']} lots)
            🔥 Confidence: {signal['confidence']*100:.1f}%
            🌊 Volatility: {signal['market_volatility']:.2f}
            🆔 Order ID: {order_id}
            
            📝 Reasons: {', '.join(signal['reasons'][:3])}
            """
            
            telegram_bot.send_signal_alert(order_message)
            
        except Exception as e:
            logger.error(f"❌ Error sending order alert: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order ID {order_id} not found in active orders")
                return None
            
            # Get order history from Zerodha
            order_history = self.kite.order_history(order_id)
            
            if order_history:
                latest_status = order_history[-1]
                
                # Update local order info
                self.active_orders[order_id]['status'] = latest_status['status']
                self.active_orders[order_id]['filled_quantity'] = latest_status.get('filled_quantity', 0)
                self.active_orders[order_id]['average_price'] = latest_status.get('average_price', 0)
                
                return latest_status
            else:
                return self.active_orders[order_id]
                
        except Exception as e:
            logger.error(f"❌ Error getting order status for {order_id}: {e}")
            return self.active_orders.get(order_id)
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"❌ Error fetching positions: {e}")
            return {}
    
    def get_holdings(self) -> List:
        """Get current holdings"""
        try:
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            logger.error(f"❌ Error fetching holdings: {e}")
            return []
    
    def close_position(self, order_id: str) -> bool:
        """Close a position by placing exit order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order ID {order_id} not found")
                return False
            
            order_info = self.active_orders[order_id]
            original_params = order_info['params']
            
            # Determine exit transaction type
            exit_transaction_type = (
                self.kite.TRANSACTION_TYPE_SELL 
                if original_params['transaction_type'] == self.kite.TRANSACTION_TYPE_BUY 
                else self.kite.TRANSACTION_TYPE_BUY
            )
            
            # Place exit order
            exit_params = original_params.copy()
            exit_params['transaction_type'] = exit_transaction_type
            
            exit_order_id = self.place_order(exit_params)
            
            if exit_order_id:
                logger.info(f"✅ Exit order placed for {order_id}. Exit order ID: {exit_order_id}")
                
                # Update position count
                self.current_positions = max(0, self.current_positions - 1)
                
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Error closing position {order_id}: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.kite.cancel_order(
                variety=self.active_orders[order_id]['params'].get('variety', self.kite.VARIETY_REGULAR),
                order_id=order_id
            )
            
            # Update local order status
            self.active_orders[order_id]['status'] = 'CANCELLED'
            self.current_positions = max(0, self.current_positions - 1)
            
            logger.info(f"✅ Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling order {order_id}: {e}")
            return False
    
    def get_active_orders(self) -> Dict:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_order_history(self) -> List:
        """Get order history"""
        return self.order_history.copy()
    
    def update_order_status(self):
        """Update status of all active orders"""
        try:
            for order_id in list(self.active_orders.keys()):
                self.get_order_status(order_id)
        except Exception as e:
            logger.error(f"❌ Error updating order statuses: {e}")
    
    def get_trading_limits(self) -> Dict:
        """Get trading limits and margins"""
        try:
            margins = self.kite.margins()
            return margins
        except Exception as e:
            logger.error(f"❌ Error fetching trading limits: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    try:
        executor = OrderExecutor()
        print("✅ Order Executor initialized successfully")
        
        # Test getting positions
        positions = executor.get_positions()
        print(f"Positions: {positions}")
        
        # Test getting holdings
        holdings = executor.get_holdings()
        print(f"Holdings count: {len(holdings)}")
        
        # Test getting limits
        limits = executor.get_trading_limits()
        print(f"Trading limits available: {'Yes' if limits else 'No'}")
        
    except Exception as e:
        print(f"❌ Order Executor test failed: {e}")
