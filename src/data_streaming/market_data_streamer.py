import logging
import time
import threading
from typing import Dict, List, Callable, Optional
from kiteconnect import KiteTicker
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN

logger = logging.getLogger(__name__)

class MarketDataStreamer:
    def __init__(self):
        self.api_key = ZERODHA_API_KEY
        self.access_token = ZERODHA_ACCESS_TOKEN
        self.kws = None
        self.is_connected = False
        self.is_streaming = False
        self.subscribed_tokens = set()
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.retry_delay = 5  # seconds
        self.max_retry_delay = 60  # seconds
        
        # Callbacks
        self.ticks_callback = None
        self.connect_callback = None
        self.close_callback = None
        self.error_callback = None
        
        # Threading
        self.streaming_thread = None
        self.should_stop = False
        
    def set_ticks_callback(self, callback: Callable):
        """Set ticks callback function"""
        self.ticks_callback = callback
        logger.info("✅ Ticks callback set")
    
    def set_connect_callback(self, callback: Callable):
        """Set connect callback function"""
        self.connect_callback = callback
        logger.info("✅ Connect callback set")
    
    def set_close_callback(self, callback: Callable):
        """Set close callback function"""
        self.close_callback = callback
        logger.info("✅ Close callback set")
    
    def set_error_callback(self, callback: Callable):
        """Set error callback function"""
        self.error_callback = callback
        logger.info("✅ Error callback set")
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        try:
            if self.ticks_callback:
                self.ticks_callback(ticks)
        except Exception as e:
            logger.error(f"❌ Error in ticks callback: {e}")
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection"""
        try:
            self.is_connected = True
            self.connection_attempts = 0  # Reset on successful connection
            logger.info("✅ WebSocket connected successfully")
            
            # Resubscribe to tokens if any
            if self.subscribed_tokens:
                self._resubscribe_tokens()
            
            if self.connect_callback:
                self.connect_callback(response)
                
        except Exception as e:
            logger.error(f"❌ Error in connect callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket closure"""
        try:
            self.is_connected = False
            self.is_streaming = False
            logger.info(f" WebSocket closed. Code: {code}, Reason: {reason}")
            
            if self.close_callback:
                self.close_callback(code, reason)
                
            # Attempt reconnection if not explicitly stopped
            if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                self._attempt_reconnection()
                
        except Exception as e:
            logger.error(f"❌ Error in close callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        try:
            logger.error(f" WebSocket error. Code: {code}, Reason: {reason}")
            
            self.is_connected = False
            self.is_streaming = False
            
            if self.error_callback:
                self.error_callback(code, reason)
                
            # Attempt reconnection with exponential backoff
            if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                self._attempt_reconnection()
                
        except Exception as e:
            logger.error(f"❌ Error in error callback: {e}")
    
    def _attempt_reconnection(self):
        """Attempt reconnection with exponential backoff"""
        try:
            self.connection_attempts += 1
            retry_delay = min(self.retry_delay * (2 ** (self.connection_attempts - 1)), self.max_retry_delay)
            
            logger.info(f"🔄 Attempting reconnection (attempt {self.connection_attempts}/{self.max_connection_attempts}) in {retry_delay} seconds...")
            
            if not self.should_stop:
                time.sleep(retry_delay)
                if not self.should_stop:
                    self.initialize_connection()
                    
        except Exception as e:
            logger.error(f"❌ Error attempting reconnection: {e}")
    
    def _resubscribe_tokens(self):
        """Resubscribe to previously subscribed tokens"""
        try:
            if self.subscribed_tokens and self.kws:
                tokens_list = list(self.subscribed_tokens)
                self.kws.subscribe(tokens_list)
                self.kws.set_mode(self.kws.MODE_FULL, tokens_list)
                logger.info(f"✅ Resubscribed to {len(tokens_list)} tokens")
        except Exception as e:
            logger.error(f"❌ Error resubscribing tokens: {e}")
    
    def initialize_connection(self) -> bool:
        """Initialize WebSocket connection"""
        try:
            if not self.api_key or not self.access_token:
                logger.error("❌ Zerodha credentials not configured")
                return False
            
            # Create KiteTicker instance
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            # Set callbacks
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            logger.info("✅ KiteTicker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing connection: {e}")
            return False
    
    def subscribe_tokens(self, tokens: List[int]) -> bool:
        """Subscribe to market data for given tokens"""
        try:
            if not self.kws:
                logger.warning("⚠️  KiteTicker not initialized")
                return False
            
            if not self.is_connected:
                logger.warning("⚠️  WebSocket not connected. Cannot subscribe.")
                return False
            
            # Add to subscribed tokens
            self.subscribed_tokens.update(tokens)
            
            # Subscribe and set mode
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            
            logger.info(f"✅ Subscribed to {len(tokens)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error subscribing to tokens: {e}")
            return False
    
    def unsubscribe_tokens(self, tokens: List[int]) -> bool:
        """Unsubscribe from market data for given tokens"""
        try:
            if not self.kws or not self.is_connected:
                logger.warning("⚠️  WebSocket not connected. Cannot unsubscribe.")
                return False
            
            # Remove from subscribed tokens
            self.subscribed_tokens.difference_update(tokens)
            
            # Unsubscribe
            self.kws.unsubscribe(tokens)
            
            logger.info(f"✅ Unsubscribed from {len(tokens)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unsubscribing from tokens: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start market data streaming"""
        try:
            if not self.kws:
                if not self.initialize_connection():
                    return False
            
            # Reset stop flag
            self.should_stop = False
            
            # Start streaming in a separate thread
            self.streaming_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.streaming_thread.start()
            
            self.is_streaming = True
            logger.info("✅ Market data streaming started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return False
    
    def _stream_worker(self):
        """Worker function for streaming"""
        try:
            while not self.should_stop:
                try:
                    if self.kws and not self.should_stop:
                        self.kws.connect(threaded=False)
                    else:
                        break
                except Exception as e:
                    logger.error(f" WebSocket connection error: {e}")
                    if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                        self._attempt_reconnection()
                    else:
                        break
        except Exception as e:
            logger.error(f"❌ Error in stream worker: {e}")
    
    def stop_streaming(self):
        """Stop market data streaming"""
        try:
            self.should_stop = True
            self.is_connected = False
            self.is_streaming = False
            self.connection_attempts = 0
            
            if self.kws:
                self.kws.close()
            
            # Wait for streaming thread to finish
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=5)
            
            logger.info("✅ Market data streaming stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping streaming: {e}")
    
    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'streaming': self.is_streaming,
            'subscribed_tokens': len(self.subscribed_tokens),
            'tokens': list(self.subscribed_tokens),
            'connection_attempts': self.connection_attempts,
            'max_connection_attempts': self.max_connection_attempts
        }

# Example usage
if __name__ == "__main__":
    print("Market Data Streamer ready!")
    print("Import and use: from src.data_streaming.market_data_streamer import MarketDataStreamer")
