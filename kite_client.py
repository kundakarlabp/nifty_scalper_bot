#!/usr/bin/env python3
"""
Wrapper for Kite Connect API with robust error handling and utility methods.
"""
import logging
from typing import Dict, List, Optional, Any
from kiteconnect import KiteConnect
from config import Config
from utils import safe_float, safe_int, format_price

logger = logging.getLogger(__name__)

class KiteClient:
    """Kite Connect API client wrapper."""

    def __init__(self):
        self.kite: Optional[KiteConnect] = None
        self.is_connected: bool = False
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the KiteConnect client with API key and access token."""
        try:
            if not Config.KITE_API_KEY:
                logger.error("Kite API key not provided in Config.KITE_API_KEY")
                return

            self.kite = KiteConnect(api_key=Config.KITE_API_KEY)

            if Config.KITE_ACCESS_TOKEN:
                self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("Kite client connected successfully")
            else:
                logger.warning("Kite access token not provided in Config.KITE_ACCESS_TOKEN")
        except Exception as e:
            logger.error(f"Failed to initialize Kite client: {e}")
            self.is_connected = False

    def get_ltp(self, symbol: str, exchange: str = "NFO") -> Optional[float]:
        """Fetch Last Traded Price for a given symbol."""
        if not self.is_connected or not self.kite:
            return None
        try:
            key = f"{exchange}:{symbol}"
            quote = self.kite.ltp(key)
            return safe_float(quote[key]['last_price'])
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str, exchange: str = "NFO") -> Optional[Dict[str, Any]]:
        """Fetch full quote data for a given symbol."""
        if not self.is_connected or not self.kite:
            return None
        try:
            key = f"{exchange}:{symbol}"
            quote = self.kite.quote(key)
            return quote.get(key)
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        exchange: str = "NFO",
        product: str = "MIS",
        variety: str = "regular"
    ) -> Optional[str]:
        """Place an order and return its order_id."""
        if not self.is_connected or not self.kite:
            logger.error("Kite client not connected")
            return None
        try:
            params: Dict[str, Any] = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'variety': variety
            }
            if price is not None:
                params['price'] = format_price(price)
            if trigger_price is not None:
                params['trigger_price'] = format_price(trigger_price)

            order_id = self.kite.place_order(**params)
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        order_type: Optional[str] = None,
        variety: str = "regular"
    ) -> bool:
        """Modify an existing order."""
        if not self.is_connected or not self.kite:
            return False
        try:
            params: Dict[str, Any] = {
                'order_id': order_id,
                'variety': variety
            }
            if price is not None:
                params['price'] = format_price(price)
            if quantity is not None:
                params['quantity'] = quantity
            if order_type is not None:
                params['order_type'] = order_type

            self.kite.modify_order(**params)
            logger.info(f"Order {order_id} modified successfully")
            return True
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        """Cancel an existing order."""
        if not self.is_connected or not self.kite:
            return False
        try:
            self.kite.cancel_order(order_id=order_id, variety=variety)
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_orders(self) -> List[Dict[str, Any]]:
        """Fetch all orders."""
        if not self.is_connected or not self.kite:
            return []
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all positions."""
        if not self.is_connected or not self.kite:
            return {'net': [], 'day': []}
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {'net': [], 'day': []}

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Fetch all holdings."""
        if not self.is_connected or not self.kite:
            return []
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []

    def get_margins(self) -> Dict[str, Any]:
        """Fetch margin information."""
        if not self.is_connected or not self.kite:
            return {}
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Error fetching margins: {e}")
            return {}

    def get_historical_data(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str = "minute"
    ) -> List[Dict[str, Any]]:
        """Fetch historical data."""
        if not self.is_connected or not self.kite:
            return []
        try:
            return self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []

    def get_instruments(self, exchange: str = "NFO") -> List[Dict[str, Any]]:
        """Fetch list of instruments for a given exchange."""
        if not self.is_connected or not self.kite:
            return []
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            logger.error(f"Error fetching instruments for {exchange}: {e}")
            return []

    def place_bracket_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float,
        squareoff: float,
        stoploss: float,
        exchange: str = "NFO",
        product: str = "BO",
        variety: str = "bo"
    ) -> Optional[str]:
        """Place a bracket order."""
        if not self.is_connected or not self.kite:
            return None
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="LIMIT",
                price=format_price(price),
                product=product,
                variety=variety,
                squareoff=format_price(squareoff),
                stoploss=format_price(stoploss)
            )
            logger.info(f"Bracket order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None

    def place_cover_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float,
        trigger_price: float,
        exchange: str = "NFO",
        product: str = "CO",
        variety: str = "co"
    ) -> Optional[str]:
        """Place a cover order."""
        if not self.is_connected or not self.kite:
            return None
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="MARKET",
                price=format_price(price),
                trigger_price=format_price(trigger_price),
                product=product,
                variety=variety
            )
            logger.info(f"Cover order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing cover order: {e}")
            return None
