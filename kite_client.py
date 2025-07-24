import logging
from typing import Dict, List, Optional, Any
from kiteconnect import KiteConnect
from config import Config
from utils import safe_float, safe_int, format_price

logger = logging.getLogger(__name__)

class KiteClient:
    """Wrapper for Kite Connect API with error handling"""
    
    def __init__(self):
        self.kite = None
        self.is_connected = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Kite Connect client"""
        try:
            if not Config.ZERODHA_API_KEY:
                logger.error("Zerodha API key not provided")
                return
            
            self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
            
            if Config.ZERODHA_ACCESS_TOKEN:
                self.kite.set_access_token(Config.ZERODHA_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("Kite client initialized successfully")
            else:
                logger.warning("No access token provided - manual authentication required")
                
        except Exception as e:
            logger.error(f"Failed to initialize Kite client: {e}")
            self.is_connected = False
    
    def get_ltp(self, symbol: str, exchange: str = "NFO") -> Optional[float]:
        """Get Last Traded Price"""
        if not self.is_connected:
            return None
        
        try:
            instrument_key = f"{exchange}:{symbol}"
            quote = self.kite.ltp(instrument_key)
            return safe_float(quote[instrument_key]['last_price'])
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str, exchange: str = "NFO") -> Optional[Dict[str, Any]]:
        """Get full quote data"""
        if not self.is_connected:
            return None
        
        try:
            instrument_key = f"{exchange}:{symbol}"
            quote = self.kite.quote(instrument_key)
            return quote[instrument_key]
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def place_order(self, symbol: str, transaction_type: str, quantity: int,
                   order_type: str = "MARKET", price: Optional[float] = None,
                   trigger_price: Optional[float] = None, 
                   exchange: str = "NFO", product: str = "MIS") -> Optional[str]:
        """Place an order"""
        if not self.is_connected:
            logger.error("Kite client not connected")
            return None
        
        if Config.DRY_RUN:
            logger.info(f"DRY RUN - Order: {transaction_type} {quantity} {symbol} @ {price or 'MARKET'}")
            return "DRY_RUN_ORDER_ID"
        
        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'variety': 'regular'
            }
            
            if price:
                order_params['price'] = format_price(price)
            
            if trigger_price:
                order_params['trigger_price'] = format_price(trigger_price)
            
            order_id = self.kite.place_order(**order_params)
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def modify_order(self, order_id: str, price: Optional[float] = None,
                    quantity: Optional[int] = None, order_type: Optional[str] = None) -> bool:
        """Modify an existing order"""
        if not self.is_connected:
            return False
        
        try:
            modify_params = {'order_id': order_id, 'variety': 'regular'}
            
            if price:
                modify_params['price'] = format_price(price)
            if quantity:
                modify_params['quantity'] = quantity
            if order_type:
                modify_params['order_type'] = order_type
            
            self.kite.modify_order(**modify_params)
            logger.info(f"Order {order_id} modified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            return False
        
        try:
            self.kite.cancel_order(order_id=order_id, variety='regular')
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        if not self.is_connected:
            return []
        
        try:
            orders = self.kite.orders()
            return orders
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all positions"""
        if not self.is_connected:
            return {'net': [], 'day': []}
        
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {'net': [], 'day': []}
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings"""
        if not self.is_connected:
            return []
        
        try:
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []
    
    def get_margins(self) -> Dict[str, Any]:
        """Get margin information"""
        if not self.is_connected:
            return {}
        
        try:
            margins = self.kite.margins()
            return margins
        except Exception as e:
            logger.error(f"Error fetching margins: {e}")
            return {}
    
    def get_historical_data(self, instrument_token: int, from_date: str, 
                           to_date: str, interval: str = "minute") -> List[Dict[str, Any]]:
        """Get historical data"""
        if not self.is_connected:
            return []
        
        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def get_instruments(self, exchange: str = "NFO") -> List[Dict[str, Any]]:
        """Get instrument list"""
        if not self.is_connected:
            return []
        
        try:
            instruments = self.kite.instruments(exchange)
            return instruments
        except Exception as e:
            logger.error(f"Error fetching instruments for {exchange}: {e}")
            return []
    
    def place_bracket_order(self, symbol: str, transaction_type: str, quantity: int,
                           price: float, squareoff: float, stoploss: float,
                           exchange: str = "NFO") -> Optional[str]:
        """Place bracket order"""
        if not self.is_connected:
            return None
        
        if Config.DRY_RUN:
            logger.info(f"DRY RUN - Bracket Order: {transaction_type} {quantity} {symbol}")
            return "DRY_RUN_BRACKET_ORDER_ID"
        
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="LIMIT",
                price=format_price(price),
                product="BO",
                variety="bo",
                squareoff=format_price(squareoff),
                stoploss=format_price(stoploss)
            )
            logger.info(f"Bracket order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None
    
    def place_cover_order(self, symbol: str, transaction_type: str, quantity: int,
                         price: float, trigger_price: float, 
                         exchange: str = "NFO") -> Optional[str]:
        """Place cover order"""
        if not self.is_connected:
            return None
        
        if Config.DRY_RUN:
            logger.info(f"DRY RUN - Cover Order: {transaction_type} {quantity} {symbol}")
            return "DRY_RUN_COVER_ORDER_ID"
        
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="MARKET",
                price=format_price(price),
                trigger_price=format_price(trigger_price),
                product="CO",
                variety="co"
            )
            logger.info(f"Cover order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing cover order: {e}")
            return None