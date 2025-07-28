import logging
import json
import time
from kiteconnect import KiteTicker
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)
timezone = pytz.timezone('Asia/Kolkata')

class WebSocketClient:
    def __init__(self):
        self.kws = None
        self.is_connected = False
        self.subscribed_tokens = set()
        self.on_ticks_callback = None
        self.on_connect_callback = None
        self.on_close_callback = None
        self.on_error_callback = None
        self.retry_count = 0
        self.max_retries = 5
        self.retry_delay = 5
        
    def initialize_connection(self) -> bool:
        """Initialize WebSocket connection"""
        try:
            if not ZERODHA_API_KEY or not ZERODHA_ACCESS_TOKEN:
                logger.error("❌ Zerodha credentials not configured")
                return False
            
            # Create KiteTicker instance
            self.kws = KiteTicker(ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN)
            
            # Set up callbacks
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            logger.info("✅ WebSocket connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing WebSocket connection: {e}")
            return False
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        try:
            if self.on_ticks_callback:
                self.on_ticks_callback(ticks)
        except Exception as e:
            logger.error(f"❌ Error in ticks callback: {e}")
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection"""
        try:
            self.is_connected = True
            self.retry_count = 0
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"✅ WebSocket connected successfully at {ist_time}")
            
            # Resubscribe to tokens if any
            if self.subscribed_tokens:
                self.subscribe_tokens(list(self.subscribed_tokens))
            
            if self.on_connect_callback:
                self.on_connect_callback(response)
        except Exception as e:
            logger.error(f"❌ Error in connect callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket closure"""
        try:
            self.is_connected = False
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f" WebSocket closed at {ist_time}. Code: {code}, Reason: {reason}")
            
            if self.on_close_callback:
                self.on_close_callback(code, reason)
                
            # Attempt reconnection
            self._attempt_reconnection()
        except Exception as e:
            logger.error(f"❌ Error in close callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        try:
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.error(f" WebSocket error at {ist_time}. Code: {code}, Reason: {reason}")
            
            if self.on_error_callback:
                self.on_error_callback(code, reason)
                
            # Attempt reconnection with exponential backoff
            self._attempt_reconnection()
        except Exception as e:
            logger.error(f"❌ Error in error callback: {e}")
    
    def _attempt_reconnection(self):
        """Attempt reconnection with exponential backoff"""
        try:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                delay = min(self.retry_delay * (2 ** (self.retry_count - 1)), 60)
                
                ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
                logger.info(f"🔄 Attempting reconnection {self.retry_count}/{self.max_retries} in {delay} seconds at {ist_time}")
                
                time.sleep(delay)
                
                if self.initialize_connection():
                    self.start_streaming()
            else:
                logger.error("❌ Maximum reconnection attempts exceeded")
        except Exception as e:
            logger.error(f"❌ Error attempting reconnection: {e}")
    
    def set_ticks_callback(self, callback):
        """Set ticks callback function"""
        self.on_ticks_callback = callback
        logger.info("✅ Ticks callback set")
    
    def set_connect_callback(self, callback):
        """Set connect callback function"""
        self.on_connect_callback = callback
        logger.info("✅ Connect callback set")
    
    def set_close_callback(self, callback):
        """Set close callback function"""
        self.on_close_callback = callback
        logger.info("✅ Close callback set")
    
    def set_error_callback(self, callback):
        """Set error callback function"""
        self.on_error_callback = callback
        logger.info("✅ Error callback set")
    
    def subscribe_tokens(self, tokens: list) -> bool:
        """Subscribe to market data for given tokens"""
        try:
            if not self.kws or not self.is_connected:
                logger.warning("⚠️  WebSocket not connected. Cannot subscribe.")
                return False
            
            if not tokens:
                logger.warning("⚠️  No tokens provided for subscription")
                return False
            
            # Add to subscribed tokens
            self.subscribed_tokens.update(tokens)
            
            # Subscribe and set mode
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"✅ Subscribed to {len(tokens)} tokens at {ist_time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error subscribing to tokens: {e}")
            return False
    
    def unsubscribe_tokens(self, tokens: list) -> bool:
        """Unsubscribe from market data for given tokens"""
        try:
            if not self.kws or not self.is_connected:
                logger.warning("⚠️  WebSocket not connected. Cannot unsubscribe.")
                return False
            
            if not tokens:
                logger.warning("⚠️  No tokens provided for unsubscription")
                return False
            
            # Remove from subscribed tokens
            self.subscribed_tokens.difference_update(tokens)
            
            # Unsubscribe
            self.kws.unsubscribe(tokens)
            
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"✅ Unsubscribed from {len(tokens)} tokens at {ist_time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unsubscribing from tokens: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start WebSocket streaming"""
        try:
            if not self.kws:
                if not self.initialize_connection():
                    return False
            
            # Start streaming in a separate thread
            import threading
            streaming_thread = threading.Thread(target=self._stream_worker, daemon=True)
            streaming_thread.start()
            
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"✅ Market data streaming started at {ist_time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return False
    
    def _stream_worker(self):
        """Worker function for streaming"""
        try:
            while True:
                try:
                    if self.kws:
                        self.kws.connect(threaded=False)
                    else:
                        logger.error(" WebSocket client not initialized")
                        break
                except Exception as e:
                    logger.error(f" WebSocket connection error: {e}")
                    if self.retry_count < self.max_retries:
                        self._attempt_reconnection()
                    else:
                        logger.error(" Maximum retries exceeded. Stopping streaming.")
                        break
        except Exception as e:
            logger.error(f"❌ Error in stream worker: {e}")
    
    def stop_streaming(self):
        """Stop WebSocket streaming"""
        try:
            self.is_connected = False
            self.retry_count = 0
            
            if self.kws:
                self.kws.close()
            
            ist_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"✅ Market data streaming stopped at {ist_time}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping streaming: {e}")
    
    def get_connection_status(self) -> dict:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'subscribed_tokens': len(self.subscribed_tokens),
            'tokens': list(self.subscribed_tokens),
            'retry_count': self.retry_count
        }

# Example usage
if __name__ == "__main__":
    print("WebSocket Client ready!")
    print("Import and use: from src.data_streaming.websocket_client import WebSocketClient")
