#!/usr/bin/env python3
"""
Kite Client Wrapper for Nifty Scalper Bot
Handles Kite Connect API integration and authentication
"""
import os
import sys
import logging
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, NetworkException
from config import Config
logger = logging.getLogger(__name__)
class KiteClient:
    """Enhanced Kite Connect client with proper error handling"""
    def __init__(self):
        self.api_key = Config.KITE_API_KEY
        self.api_secret = Config.KITE_API_SECRET
        self.request_token_file = "request_token.txt"
        self.kite = None
        self.is_connected = False
        self._initialize_client()
    def _initialize_client(self):
        """Initialize Kite Connect client and establish connection"""
        try:
            if not self.api_key or not self.api_secret:
                raise ValueError("Kite API key or secret not configured")
            # Initialize KiteConnect object
            self.kite = KiteConnect(api_key=self.api_key)
            # Try to load existing request token
            request_token = self._load_request_token()
            if request_token:
                # Generate session and set access token
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.kite.set_access_token(data["access_token"])
                logger.info("Kite client initialized with existing session")
                self.is_connected = True
                return
            # If no token, initiate login (this part would typically be handled manually)
            logger.warning("No request token found. Please generate one and save it to request_token.txt")
            login_url = self.kite.login_url()
            logger.info(f"Login URL: {login_url}")
        except TokenException as e:
            logger.error(f"Token exception: {e}")
        except NetworkException as e:
            logger.error(f"Network exception: {e}")
        except Exception as e:
            logger.error(f"Error initializing Kite client: {e}")
    def _load_request_token(self) -> str:
        """Load request token from file"""
        try:
            if os.path.exists(self.request_token_file):
                with open(self.request_token_file, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading request token: {e}")
        return ""
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for given symbol"""
        try:
            # This is a simplified version - you might need to fetch instruments
            # and search for the correct one based on symbol, expiry, etc.
            # For NIFTY, you'd typically look for the futures or options contract
            # This example assumes a mapping or lookup mechanism
            instrument_map = {
                "NIFTY": 256265,  # Example token, replace with actual logic
                # Add other symbols as needed
            }
            return instrument_map.get(symbol)
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None
    def get_ltp(self, instrument_token: int) -> Optional[float]:
        """Get last traded price for instrument"""
        try:
            ltp_data = self.kite.ltp([instrument_token])
            return ltp_data[str(instrument_token)]['last_price']
        except Exception as e:
            logger.error(f"Error getting LTP: {e}")
            return None
# Example usage (if run as script)
if __name__ == "__main__":
    client = KiteClient()
    if client.is_connected:
        print("Kite client connected successfully")
        # Example: Get LTP for NIFTY (using example token)
        token = client.get_instrument_token("NIFTY")
        if token:
            ltp = client.get_ltp(token)
            print(f"NIFTY LTP: {ltp}")
    else:
        print("Failed to connect to Kite")
