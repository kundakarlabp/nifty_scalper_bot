# src/data_streaming/market_data_streamer.py
"""
Handles real-time market data streaming from Zerodha Kite WebSocket.
Manages connections, subscriptions, callbacks, and automatic reconnection.
"""
import logging
import time
import threading
from typing import Dict, List, Callable, Any, Optional
from kiteconnect import KiteTicker
from kiteconnect.exceptions import KiteException

# Import configuration using the Config class for consistency
from config import Config

logger = logging.getLogger(__name__)

class MarketDataStreamer:
    """
    Manages the Kite WebSocket connection for streaming live market data.
    """
    def __init__(self):
        """Initializes the MarketDataStreamer with configuration and state."""
        self.api_key: str = Config.ZERODHA_API_KEY
        self.access_token: str = Config.KITE_ACCESS_TOKEN
        self.kws: Optional[KiteTicker] = None
        self.is_connected: bool = False
        self.is_streaming: bool = False
        self.subscribed_tokens: set = set()
        self.connection_attempts: int = 0
        self.max_connection_attempts: int = 5
        self.retry_delay: int = 5  # seconds
        self.max_retry_delay: int = 60  # seconds

        # Callback hooks for external logic
        self.ticks_callback: Optional[Callable[[List[Dict]], None]] = None
        self.connect_callback: Optional[Callable[[Dict], None]] = None
        self.close_callback: Optional[Callable[[int, str], None]] = None
        self.error_callback: Optional[Callable[[int, str], None]] = None

        self.streaming_thread: Optional[threading.Thread] = None
        self.should_stop: bool = False

    def set_ticks_callback(self, callback: Callable[[List[Dict]], None]) -> None:
        """Set the callback function for handling incoming ticks."""
        self.ticks_callback = callback
        logger.debug("✅ Ticks callback set")

    def set_connect_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set the callback function for WebSocket connection events."""
        self.connect_callback = callback
        logger.debug("✅ Connect callback set")

    def set_close_callback(self, callback: Callable[[int, str], None]) -> None:
        """Set the callback function for WebSocket close events."""
        self.close_callback = callback
        logger.debug("✅ Close callback set")

    def set_error_callback(self, callback: Callable[[int, str], None]) -> None:
        """Set the callback function for WebSocket error events."""
        self.error_callback = callback
        logger.debug("✅ Error callback set")

    def _on_ticks(self, ws: KiteTicker, ticks: List[Dict]) -> None:
        """Internal handler for incoming ticks from Kite WebSocket."""
        try:
            if self.ticks_callback:
                self.ticks_callback(ticks)
            else:
                logger.warning(".Ticks callback is not set, dropping ticks.")
        except Exception as e:
            logger.error(f"❌ Error in ticks callback handler: {e}", exc_info=True)

    def _on_connect(self, ws: KiteTicker, response: Dict) -> None:
        """Internal handler for WebSocket connection established."""
        try:
            self.is_connected = True
            self.connection_attempts = 0
            logger.info("✅ Kite WebSocket connected successfully.")
            
            # Resubscribe to tokens if any were previously subscribed
            if self.subscribed_tokens:
                self._resubscribe_tokens()
                
            # Notify external logic via callback
            if self.connect_callback:
                self.connect_callback(response)
        except Exception as e:
            logger.error(f"❌ Error in connect callback handler: {e}", exc_info=True)

    def _on_close(self, ws: KiteTicker, code: int, reason: str) -> None:
        """Internal handler for WebSocket connection closed."""
        try:
            logger.info(f"🔌 Kite WebSocket closed. Code: {code}, Reason: {reason}")
            self.is_connected = False
            self.is_streaming = False
            
            # Notify external logic via callback
            if self.close_callback:
                self.close_callback(code, reason)
                
            # Attempt reconnection if not explicitly stopped and within retry limits
            if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                self._attempt_reconnection()
            else:
                if self.should_stop:
                    logger.info("⏹️ WebSocket closure was intentional (stop requested).")
                else:
                    logger.error("🚫 Maximum reconnection attempts reached. Stopping streamer.")
                    
        except Exception as e:
            logger.error(f"❌ Error in close callback handler: {e}", exc_info=True)

    def _on_error(self, ws: KiteTicker, code: int, reason: str) -> None:
        """Internal handler for WebSocket errors."""
        try:
            logger.error(f"⚠️ Kite WebSocket error. Code: {code}, Reason: {reason}")
            self.is_connected = False
            self.is_streaming = False
            
            # Notify external logic via callback
            if self.error_callback:
                self.error_callback(code, reason)
                
            # Attempt reconnection if not explicitly stopped and within retry limits
            if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                self._attempt_reconnection()
            else:
                if self.should_stop:
                    logger.info("⏹️ WebSocket error handling stopped (stop requested).")
                else:
                    logger.error("🚫 Maximum reconnection attempts reached after error. Stopping streamer.")
                    
        except Exception as e:
            logger.error(f"❌ Error in error callback handler: {e}", exc_info=True)

    def _attempt_reconnection(self) -> None:
        """Attempts to reconnect to the WebSocket with exponential backoff."""
        self.connection_attempts += 1
        # Exponential backoff, capped at max_retry_delay
        delay = min(self.retry_delay * (2 ** (self.connection_attempts - 1)), self.max_retry_delay)
        logger.info(f"🔁 Attempting to reconnect in {delay} seconds... (Attempt {self.connection_attempts}/{self.max_connection_attempts})")
        
        # Use a separate thread for sleep to not block the WebSocket thread
        def delayed_reconnect():
            time.sleep(delay)
            if not self.should_stop:
                logger.info("🔁 Initiating reconnection...")
                self.initialize_connection() # This will call connect() internally
        
        threading.Thread(target=delayed_reconnect, daemon=True).start()

    def _resubscribe_tokens(self) -> None:
        """Resubscribes to all previously subscribed tokens after a reconnection."""
        if self.kws and self.subscribed_tokens:
            tokens_list = list(self.subscribed_tokens)
            try:
                self.kws.subscribe(tokens_list)
                self.kws.set_mode(self.kws.MODE_FULL, tokens_list)
                logger.info(f"✅ Resubscribed to {len(tokens_list)} tokens after reconnection.")
            except Exception as e:
                logger.error(f"❌ Failed to resubscribe tokens: {e}", exc_info=True)

    def initialize_connection(self) -> bool:
        """
        Initializes the KiteTicker WebSocket client.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if not self.api_key or not self.access_token:
            logger.error("❌ Cannot initialize Kite connection: API key or access token is missing in config.")
            return False
            
        try:
            # Create a new KiteTicker instance
            self.kws = KiteTicker(self.api_key, self.access_token)
            
            # Assign internal handlers to KiteTicker events
            self.kws.on_ticks = self._on_ticks
            self.kws.on_connect = self._on_connect
            self.kws.on_close = self._on_close
            self.kws.on_error = self._on_error
            
            logger.info("✅ KiteTicker client initialized.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize KiteTicker client: {e}", exc_info=True)
            return False

    def subscribe_tokens(self, tokens: List[int]) -> bool:
        """
        Subscribes to a list of instrument tokens for live data.

        Args:
            tokens (List[int]): A list of instrument tokens.

        Returns:
            bool: True if subscription request was sent, False otherwise.
        """
        if not self.kws:
            logger.error("❌ Cannot subscribe: KiteTicker client is not initialized.")
            return False
        if not self.is_connected:
            logger.warning("⚠️ Cannot subscribe: WebSocket is not currently connected. Tokens will be subscribed upon next connection.")
            # Store tokens to subscribe later in _on_connect
            self.subscribed_tokens.update(tokens)
            return False # Indicate subscription wasn't immediate

        try:
            self.subscribed_tokens.update(tokens)
            self.kws.subscribe(tokens)
            self.kws.set_mode(self.kws.MODE_FULL, tokens)
            logger.info(f"✅ Subscribed to {len(tokens)} tokens: {tokens}")
            return True
        except Exception as e:
            logger.error(f"❌ Subscription error for tokens {tokens}: {e}", exc_info=True)
            return False

    def unsubscribe_tokens(self, tokens: List[int]) -> bool:
        """
        Unsubscribes from a list of instrument tokens.

        Args:
            tokens (List[int]): A list of instrument tokens.

        Returns:
            bool: True if unsubscription request was sent, False otherwise.
        """
        if not self.kws or not self.is_connected:
            logger.warning("⚠️ Cannot unsubscribe: WebSocket is not connected.")
            # Remove from local set anyway
            self.subscribed_tokens.difference_update(tokens)
            return False

        try:
            self.subscribed_tokens.difference_update(tokens)
            self.kws.unsubscribe(tokens)
            logger.info(f"✅ Unsubscribed from {len(tokens)} tokens: {tokens}")
            return True
        except Exception as e:
            logger.error(f"❌ Unsubscription error for tokens {tokens}: {e}", exc_info=True)
            return False

    def start_streaming(self) -> bool:
        """
        Starts the WebSocket streaming in a background thread.

        Returns:
            bool: True if the streaming thread was started, False otherwise.
        """
        if not self.kws and not self.initialize_connection():
            logger.error("❌ Cannot start streaming: KiteTicker initialization failed.")
            return False

        self.should_stop = False
        try:
            # Start the WebSocket connection in a separate thread
            # Using `threaded=False` inside a thread is the recommended way
            self.streaming_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.streaming_thread.start()
            self.is_streaming = True
            logger.info("🚀 Market data streaming started.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start streaming thread: {e}", exc_info=True)
            self.is_streaming = False
            return False

    def _stream_worker(self) -> None:
        """The worker function that runs the Kite WebSocket connection."""
        try:
            if self.kws and not self.should_stop:
                logger.debug("🔗 Connecting to Kite WebSocket...")
                # This is the blocking call that runs the WebSocket loop
                self.kws.connect(threaded=False)
                # If connect() returns, it means the connection was closed
                logger.debug("🔗 Kite WebSocket connect() loop exited.")
        except KiteException as e:
            logger.error(f"❌ KiteException in WebSocket connection: {e.message} (Code: {e.code})")
            # Let the _on_error or _on_close handler deal with reconnection logic
        except Exception as e:
            logger.error(f"❌ Unexpected error in WebSocket connection worker: {e}", exc_info=True)
            # Potentially trigger reconnection here if not handled by Kite's callbacks
            if not self.should_stop and self.connection_attempts < self.max_connection_attempts:
                 self._attempt_reconnection()

    def stop_streaming(self) -> None:
        """Stops the WebSocket streaming and cleans up resources."""
        logger.info("🛑 Stopping market data streaming...")
        self.should_stop = True
        self.connection_attempts = 0 # Reset for next start
        
        # Signal the Kite WebSocket to close
        if self.kws:
            try:
                self.kws.close()
                logger.debug("🔌 Kite WebSocket close signal sent.")
            except Exception as e:
                logger.warning(f"⚠️ Error while closing Kite WebSocket: {e}", exc_info=True)
        
        # Wait for the streaming thread to finish (with a timeout)
        if self.streaming_thread and self.streaming_thread.is_alive():
            logger.debug("⏳ Waiting for streaming thread to finish...")
            self.streaming_thread.join(timeout=5) # Wait up to 5 seconds
            if self.streaming_thread.is_alive():
                logger.warning("⚠️ Streaming thread did not finish within the timeout.")
            else:
                logger.debug("✅ Streaming thread finished.")
        
        # Reset state
        self.is_connected = False
        self.is_streaming = False
        logger.info("🛑 Market data streaming stopped.")

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the WebSocket connection.

        Returns:
            Dict[str, Any]: A dictionary containing connection status information.
        """
        return {
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "tokens": list(self.subscribed_tokens),
            "connection_attempts": self.connection_attempts,
            "max_connection_attempts": self.max_connection_attempts
        }

# Example usage (if run directly)
if __name__ == "__main__":
    print("✅ MarketDataStreamer module is ready for use.")
