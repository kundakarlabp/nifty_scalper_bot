#!/usr/bin/env python3
"""
Kite Client Wrapper for Nifty Scalper Bot
Handles Kite Connect API integration and authentication
"""
# Ensure all necessary types are imported
from typing import Dict, List, Optional, Any
import logging
# Import KiteConnect library
from kiteconnect import KiteConnect
# Import configuration and utility functions
from config import Config
from utils import safe_float, safe_int, format_price

# Setup logger for this module
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
            # Check if API key is provided in configuration
            if not Config.ZERODHA_API_KEY:
                logger.error("Zerodha API key not provided")
                return

            # Initialize KiteConnect object with API key
            self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)

            # Set access token if provided in configuration
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
            logger.error(f"Error getting LTP for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, transaction_type: str, quantity: int,
                    price: Optional[float] = None, trigger_price: Optional[float] = None,
                    exchange: str = "NFO", product: str = "MIS",
                    order_type: str = "MARKET") -> Optional[str]:
        """Place an order"""
        if not self.is_connected:
            logger.error("Kite client not connected")
            return None

        # Handle dry run mode for testing
        if Config.DRY_RUN:
            logger.info(f"DRY RUN - Order: {transaction_type} {quantity} {symbol} @ {price or 'MARKET'}")
            return "DRY_RUN_ORDER_ID"

        try:
            # Prepare order parameters
            order_params = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': order_type,
                'product': product,
                'variety': 'regular'
            }

            # Add price if specified (for LIMIT orders)
            if price:
                order_params['price'] = format_price(price)

            # Add trigger price if specified (for SL/SL-M orders)
            if trigger_price:
                order_params['trigger_price'] = format_price(trigger_price)

            # Place the order using Kite API
            order_response = self.kite.place_order(**order_params)
            logger.info(f"Order placed successfully: {order_response}")
            return order_response

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def get_historical_data(self, instrument_token: int, from_date: str, to_date: str, interval: str) -> List[Dict[str, Any]]:
        """Fetch historical data for an instrument"""
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
            logger.error(f"Error fetching historical  {e}")
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
                            price: float, stop_loss: float, target: float,
                            exchange: str = "NFO", product: str = "MIS") -> Optional[str]:
        """Place a bracket order"""
        if not self.is_connected:
            logger.error("Kite client not connected")
            return None

        # Handle dry run mode for testing
        if Config.DRY_RUN:
            logger.info(f"DRY RUN - Bracket Order: {transaction_type} {quantity} {symbol} SL:{stop_loss} TGT:{target}")
            return "DRY_RUN_BO_ID"

        try:
            # Prepare bracket order parameters
            bo_params = {
                'tradingsymbol': symbol,
                'exchange': exchange,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'order_type': 'LIMIT',
                'product': product,
                'variety': 'bo',  # Bracket Order variety
                'price': format_price(price),
                'stoploss': format_price(stop_loss),
                'squareoff': format_price(target)
            }

            # Place the bracket order using Kite API
            bo_response = self.kite.place_order(**bo_params)
            logger.info(f"Bracket order placed successfully: {bo_response}")
            return bo_response

        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {e}")
            return None

    # CORRECTED METHOD: Ensure Optional is imported and used correctly
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """
        Get instrument token for given symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'NIFTY23JAN23FUT')
            
        Returns:
            Optional[int]: Instrument token if found, None otherwise
        """
        try:
            # This is a simplified version - you might need to fetch instruments
            # and search for the correct one based on symbol, expiry, etc.
            # For NIFTY, you'd typically look for the futures or options contract
            # This example assumes a mapping or lookup mechanism
            # In a real implementation, you would search through self.get_instruments()
            
            # Example placeholder mapping (replace with actual logic)
            instrument_map = {
                "NIFTY": 256265,  # Example token, replace with actual logic
                "BANKNIFTY": 260105, # Example token
                # Add other symbols as needed
            }
            
            token = instrument_map.get(symbol)
            if token:
                logger.debug(f"Instrument token found for {symbol}: {token}")
                return token
            else:
                logger.warning(f"Instrument token not found for symbol: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None

# Example usage (if run as script)
if __name__ == "__main__":
    # Create KiteClient instance
    client = KiteClient()
    
    # Check connection status
    if client.is_connected:
        print("Kite client connected successfully")
        
        # Example: Get LTP for NIFTY (using example token)
        token = client.get_instrument_token("NIFTY")
        if token:
            ltp = client.get_ltp("NIFTY23JAN23FUT") # Use actual symbol
            print(f"NIFTY LTP: {ltp}")
    else:
        print("Failed to connect to Kite")
