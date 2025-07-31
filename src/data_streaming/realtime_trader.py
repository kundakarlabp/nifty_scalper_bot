# src/data_streaming/realtime_trader.py
"""
Core real-time trading engine integrating data streaming, strategy,
risk management, execution (via OrderExecutor), and Telegram notifications.
"""
import sys
import os
# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import time
import threading # Import threading for polling
import pytz
from typing import Dict, List, Any, Optional
from kiteconnect import KiteConnect

# Import configuration using the Config class for consistency
from config import Config

# Import necessary modules
from src.data_streaming.market_data_streamer import MarketDataStreamer
from src.data_streaming.data_processor import StreamingDataProcessor
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing
# Import the new OrderExecutor
from src.execution.order_executor import OrderExecutor
# Import Telegram Controller
from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)

class RealTimeTrader:
    """
    Core real-time trading engine integrating data streaming, strategy,
    risk management, and execution (simulated or live via OrderExecutor).
    """
    def __init__(self, order_executor: Optional[OrderExecutor] = None):
        """
        Initialize the RealTimeTrader.

        Args:
            order_executor (Optional[OrderExecutor]): An instance of OrderExecutor
                for live order management. If None, execution will be simulated.
        """
        self.kite: Optional[KiteConnect] = None
        self.order_executor: Optional[OrderExecutor] = order_executor
        self.is_trading: bool = False
        self.execution_enabled: bool = False
        self.is_connected: bool = False
        self.start_time: Optional[float] = None
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.trading_instruments: List[int] = [] # List of instrument tokens
        # Store mapping from token to symbol/exchange for order placement and status
        self.token_symbol_map: Dict[int, Dict[str, str]] = {}
        self.active_signals: Dict[int, Dict[str, Any]] = {}
        # This can be used to track positions if needed at this level,
        # though OrderExecutor also tracks them.
        self.active_positions: Dict[str, Dict[str, Any]] = {} # Placeholder

        # --- Initialize core modules ---
        self.streamer = MarketDataStreamer()
        self.processor = StreamingDataProcessor()

        # --- Initialize Strategy ---
        self.strategy = DynamicScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )

        # --- Initialize Risk Manager ---
        self.risk_manager = PositionSizing(
            account_size=Config.ACCOUNT_SIZE,
            risk_per_trade=Config.RISK_PER_TRADE,
            max_drawdown=Config.MAX_DRAWDOWN
        )

        # --- Initialize Telegram Controller WITH callbacks ---
        self.telegram_controller = TelegramController(
            status_callback=self.get_trading_status,
            control_callback=self._handle_telegram_control
        )
        # Store thread reference for Telegram polling
        self.telegram_polling_thread: Optional[threading.Thread] = None
        self._is_polling = False # Flag to control polling loop

        # --- Register callbacks for the streamer ---
        self.streamer.set_ticks_callback(self._handle_ticks)
        self.streamer.set_connect_callback(self._handle_connect)
        self.streamer.set_close_callback(self._handle_close)
        self.streamer.set_error_callback(self._handle_error)

        # --- Setup Kite Connect API client ---
        self.setup_kite()

    def setup_kite(self):
        """Initialize the Kite Connect client with credentials."""
        try:
            if Config.ZERODHA_API_KEY and Config.KITE_ACCESS_TOKEN:
                self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("✅ Zerodha Kite Connect initialized.")
                # Send startup alert via Telegram
                self.telegram_controller.send_startup_alert()
            else:
                logger.warning("⚠️ Missing Zerodha API key or access token in Config.")
                self.telegram_controller.send_message("⚠️ Bot started but Kite credentials missing.")
        except Exception as e:
            logger.error(f"❌ Kite Connect setup error: {e}")
            self.telegram_controller.send_message(f"❌ Kite Connect setup error: {e}")

    # --- WebSocket Callbacks ---
    def _handle_connect(self, _):
        """Callback when WebSocket connects."""
        logger.info("📡 WebSocket connected")
        self.is_connected = True # Set internal flag
        # Send connection status to Telegram
        self.telegram_controller.send_message("📡 *WebSocket connected*")
        if self.trading_instruments:
            # subscribe_tokens returns True/False, not a list
            subscription_success = self.streamer.subscribe_tokens(self.trading_instruments)
            if subscription_success:
                logger.info(f"✅ Subscribed to {len(self.trading_instruments)} tokens after connection/reconnection.")
            else:
                logger.warning("⚠️ Failed to subscribe to tokens after connection/reconnection.")
        # Note: The original logic tried to get 'successfully_subscribed' from subscribe_tokens
        # and then use len() on it. The provided market_data_streamer.py returns bool.
        # The _resubscribe_tokens logic is handled internally within market_data_streamer.py's _on_connect.


    def _handle_close(self, code, reason):
        """Callback when WebSocket closes."""
        logger.warning(f"🔌 WebSocket closed | Code: {code}, Reason: {reason}")
        self.is_connected = False
        # Send disconnection status to Telegram
        self.telegram_controller.send_message(f"🔌 *WebSocket closed* | Code: {code}")

    def _handle_error(self, ws, error): # Adjusted signature to match common KiteTicker callback
        """Callback when WebSocket encounters an error."""
        logger.error(f"⚠️ WebSocket error: {error}")
        # Send error status to Telegram
        self.telegram_controller.send_message(f"⚠️ *WebSocket error*: {error}")

    def _handle_ticks(self, ticks):
        """
        Main tick processing callback.
        Processes ticks, updates OHLC, and checks for signals.
        """
        try:
            for tick in ticks:
                processed_tick = self.processor.process_tick(tick)
                if not processed_tick:
                    continue
                token = processed_tick['instrument_token']
                now = time.time()
                last_tick_time = getattr(self, f'_last_tick_time_{token}', 0)
                if now - last_tick_time >= 60:
                    setattr(self, f'_last_tick_time_{token}', now)
                    self.processor.update_ohlc(token, '1min')
                    self._check_trading_signals(token)
        except Exception as e:
            logger.error(f"❌ Tick handler error: {e}", exc_info=True)
            # Send error status to Telegram
            self.telegram_controller.send_message(f"❌ *Tick handler error*: {e}")

    # --- Signal & Execution Logic ---
    def _check_trading_signals(self, token: int):
        """
        Fetch latest data and generate/check for trading signals.
        """
        try:
            # Fetch last 100 candles for analysis
            ohlc_data = self.processor.get_latest_data(token, 100)
            if ohlc_data is None or ohlc_data.empty or len(ohlc_data) < 50: # Ensure enough data
                return
            current_price = self.processor.get_current_price(token)
            if current_price is None:
                return

            # Generate signal using the strategy module
            signal = self.strategy.generate_signal(ohlc_data, current_price)

            # If a valid signal is generated and meets confidence threshold from Config
            if signal and signal.get('confidence', 0) >= Config.CONFIDENCE_THRESHOLD:
                # Avoid duplicate signals for the same token
                if token not in self.active_signals:
                    self._handle_trading_signal(token, signal)
        except Exception as e:
            logger.error(f"❌ Signal check failed for token {token}: {e}", exc_info=True)
            # Send error status to Telegram
            self.telegram_controller.send_message(f"❌ *Signal check failed* for token {token}: {e}")

    def _handle_trading_signal(self, token: int, signal: Dict[str, Any]):
        """
        Process a generated signal: calculate position size, send alerts,
        and potentially execute (if enabled).
        """
        execution_status = "DISABLED"
        position_details = {}
        try:
            logger.info(f"🎯 Signal for token {token}: {signal['signal']}")
            # Calculate position size based on risk management rules
            position_details = self.risk_manager.calculate_position_size(
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                signal_confidence=signal['confidence'],
                market_volatility=signal.get('market_volatility', 0) # Provide default if missing
            )
            # Skip if position size calculation failed or is zero
            if not position_details or position_details.get('quantity', 0) <= 0:
                logger.warning(f"⚠️ Position size invalid for token {token}")
                self.telegram_controller.send_message(f"⚠️ *Position size invalid* for token {token}")
                execution_status = "INVALID_SIZE"
                # Store signal for tracking even if invalid
                self._store_signal_for_tracking(token, signal, position_details, execution_status)
                return # Exit early

            # --- Use the enhanced signal alert ---
            self.telegram_controller.send_signal_alert(token, signal, position_details)
            logger.info(f"Enhanced Signal Alert Sent for token {token}")

            # --- Handle Execution ---
            if self.execution_enabled and self.order_executor and self.kite:
                logger.info("💼 Initiating live execution via OrderExecutor...")
                
                # Get symbol and exchange from the stored mapping
                instrument_info = self.token_symbol_map.get(token)
                if not instrument_info:
                    error_msg = f"❌ Cannot execute: No symbol/exchange mapping for token {token}"
                    logger.error(error_msg)
                    self.telegram_controller.send_message(error_msg)
                    execution_status = "FAILED_MAPPING"
                    self._store_signal_for_tracking(token, signal, position_details, execution_status)
                    return # Exit early

                symbol = instrument_info["symbol"]
                exchange = instrument_info["exchange"]
                transaction_type = signal['signal'] # Assuming 'BUY' or 'SELL'

                # 1. Place Entry Order
                entry_order_id = self.order_executor.place_entry_order(
                    symbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=position_details['quantity']
                    # product and order_type can use defaults from Config/OrderExecutor
                )
                if entry_order_id:
                    logger.info(f"✅ Entry order placed, Order ID: {entry_order_id}")
                    # 2. Wait for order fill confirmation (simplified)
                    # In a real scenario, you'd poll kite.order_history or listen via OMS WebSocket
                    # This is a placeholder wait. Consider a more robust check.
                    time.sleep(2) # TODO: Replace with proper order status check

                    # 3. Get filled price (simplified, use actual order history)
                    # Placeholder: using signal's entry price. Improve this.
                    filled_entry_price = signal['entry_price']
                    logger.info(f"ℹ️ Using signal entry price {filled_entry_price} as filled price (placeholder)")

                    # 4. Setup GTT Orders
                    gtt_success = self.order_executor.setup_gtt_orders(
                        entry_order_id=entry_order_id,
                        entry_price=filled_entry_price,
                        stop_loss_price=signal['stop_loss'],
                        target_price=signal['target'],
                        symbol=symbol,
                        exchange=exchange,
                        quantity=position_details['quantity'],
                        transaction_type=transaction_type # Must match entry order
                    )
                    if gtt_success:
                        logger.info("✅ Entry order and GTTs placed successfully via OrderExecutor")
                        execution_status = "SUCCESS"
                        # Update risk manager that a position is open
                        self.risk_manager.update_position_status(is_open=True)
                        # Send success alert
                        self.telegram_controller.send_message(
                            f"✅ *Trade Executed*\n"
                            f"Symbol: {symbol}\n"
                            f"Type: {transaction_type}\n"
                            f"Qty: {position_details['quantity']}\n"
                            f"Entry: {filled_entry_price:.2f}"
                        )
                    else:
                        error_msg = "❌ Failed to place GTT orders via OrderExecutor"
                        logger.error(error_msg)
                        self.telegram_controller.send_message(error_msg)
                        execution_status = "FAILED_GTT"
                        # TODO: Consider cancelling the entry order if GTTs failed
                else:
                    error_msg = "❌ Failed to place entry order via OrderExecutor"
                    logger.error(error_msg)
                    self.telegram_controller.send_message(error_msg)
                    execution_status = "FAILED_ENTRY"
            else:
                # Execution is disabled or components missing
                if not self.execution_enabled:
                    logger.info("⚠️ Execution is disabled (simulation mode)")
                    execution_status = "SIMULATED"
                elif not self.order_executor:
                    logger.warning("⚠️ OrderExecutor not available for execution")
                    execution_status = "NO_EXECUTOR"
                elif not self.kite:
                    error_msg = "❌ Kite client not initialized for execution"
                    logger.error(error_msg)
                    self.telegram_controller.send_message(error_msg)
                    execution_status = "NO_KITE"
            
            # Store the active signal regardless of execution outcome for tracking
            self._store_signal_for_tracking(token, signal, position_details, execution_status)

        except Exception as e:
            error_msg = f"❌ Error handling signal for token {token}: {e}"
            logger.error(error_msg, exc_info=True)
            self.telegram_controller.send_message(error_msg)
            execution_status = "ERROR_EXCEPTION"
            # Store signal with error status
            self._store_signal_for_tracking(token, signal, position_details, execution_status, error=str(e))

    def _store_signal_for_tracking(self, token: int, signal: Dict[str, Any], position_details: Dict[str, Any], execution_status: str, error: str = ""):
        """Helper to store signal information in active_signals."""
        self.active_signals[token] = {
            "signal": signal,
            "position_info": position_details,
            "timestamp": time.time(),
            "status": "processed" if not error else "error",
            "execution_status": execution_status,
            "error": error
        }

    # --- Public Methods for Control ---
    def add_trading_instrument(self, token: int, symbol: str, exchange: str) -> bool:
        """
        Add an instrument token to the watchlist and subscribe.
        Also stores the symbol/exchange mapping needed for order execution.

        Args:
            token (int): The instrument token.
            symbol (str): The trading symbol (e.g., NIFTY23APR18000CE).
            exchange (str): The exchange (e.g., NFO).

        Returns:
            bool: True if added successfully, False otherwise.
        """
        try:
            if token not in self.trading_instruments:
                self.trading_instruments.append(token)
                # Store the mapping for order execution and status display
                self.token_symbol_map[token] = {"symbol": symbol, "exchange": exchange}
                logger.info(f"➕ Token added: {token} -> {symbol} ({exchange})")
                # Send notification to Telegram
                self.telegram_controller.send_message(f"➕ *Instrument Added*\nToken: {token}\nSymbol: {symbol}")

                # Subscribe immediately if streamer is already connected
                if self.streamer.is_connected:
                    successfully_subscribed = self.streamer.subscribe_tokens([token])
                    # Note: subscribe_tokens returns bool, not list of tokens.
                    if not successfully_subscribed:
                         logger.warning(f"⚠️ Failed to subscribe to token {token} immediately.")
            else:
                 logger.info(f"ℹ️ Token {token} already in trading list.")
            return True
        except Exception as e:
            error_msg = f"❌ Add instrument error for token {token}: {e}"
            logger.error(error_msg, exc_info=True)
            self.telegram_controller.send_message(error_msg)
            return False

    def remove_trading_instrument(self, token: int) -> bool:
        """
        Remove an instrument token from the watchlist and unsubscribe.

        Args:
            token (int): The instrument token.

        Returns:
            bool: True if removed successfully, False otherwise.
        """
        try:
            if token in self.trading_instruments:
                self.trading_instruments.remove(token)
                logger.info(f"➖ Token removed: {token}")
                # Send notification to Telegram
                token_symbol = self.token_symbol_map.get(token, {}).get('symbol', 'Unknown')
                self.telegram_controller.send_message(f"➖ *Instrument Removed*\nToken: {token}\nSymbol: {token_symbol}")

                # Remove mapping
                removed_info = self.token_symbol_map.pop(token, None)
                if not removed_info:
                    logger.warning(f"⚠️ No symbol/exchange mapping found for removed token {token}")

                # Unsubscribe if streamer is connected
                if self.streamer.is_connected:
                    self.streamer.unsubscribe_tokens([token])

                # Clean up related data
                self.processor.clear_buffer(token)
                self.active_signals.pop(token, None)
                # Optionally remove from active_positions if implemented and used
                self.active_positions.pop(str(token), None) # Assuming key is str(token)
            else:
                 logger.info(f"ℹ️ Token {token} not found in trading list.")
            return True
        except Exception as e:
            error_msg = f"❌ Remove instrument error for token {token}: {e}"
            logger.error(error_msg, exc_info=True)
            self.telegram_controller.send_message(error_msg)
            return False

    def enable_trading(self, enable: bool = True):
        """Enable or disable actual order execution."""
        self.execution_enabled = enable
        status_msg = "enabled" if enable else "disabled"
        status_emoji = "✅" if enable else "⚠️"
        logger.info(f"{status_emoji} Trading execution {status_msg}")
        # Notify via Telegram
        self.telegram_controller.send_message(f"{status_emoji} *Trading execution {status_msg}*")

    # --- NEW: Telegram Control Callback Method ---
    def _handle_telegram_control(self, enable: bool) -> bool:
        """
        Callback method for Telegram controller to enable/disable trading.
        This is called by the TelegramController when /enable or /disable is used.
        """
        try:
            self.enable_trading(enable)
            logger.info(f"{'✅' if enable else '⚠️'} Trading execution toggled via Telegram to {'ENABLED' if enable else 'DISABLED'}")
            return True
        except Exception as e:
            logger.error(f"❌ Error in Telegram control callback: {e}", exc_info=True)
            return False

    def start_trading(self) -> bool:
        """Start the real-time trading session."""
        try:
            if not self.trading_instruments:
                warning_msg = "⚠️ No tokens configured for trading"
                logger.warning(warning_msg)
                self.telegram_controller.send_message(warning_msg)
                return False

            # Initialize and start the data stream
            if not self.streamer.initialize_connection():
                error_msg = "❌ Failed to initialize WebSocket connection"
                logger.error(error_msg)
                self.telegram_controller.send_message(error_msg)
                return False

            # Subscription happens in _handle_connect callback or can be done here if connection is immediate
            # self.streamer.subscribe_tokens(self.trading_instruments)

            if not self.streamer.start_streaming():
                error_msg = "❌ Failed to start WebSocket streaming"
                logger.error(error_msg)
                self.telegram_controller.send_message(error_msg)
                return False

            # Set internal state
            self.is_trading = True
            self.start_time = time.time()
            logger.info("✅ Real-time trading session started")

            # --- Start Telegram Polling in a separate thread ---
            if not self._is_polling:
                self._is_polling = True
                self.telegram_polling_thread = threading.Thread(target=self._run_telegram_polling, daemon=True)
                self.telegram_polling_thread.start()
                logger.info("📡 Telegram polling thread started.")

            # Send session start alert via Telegram controller's dedicated method
            self.telegram_controller.send_realtime_session_alert("START")
            return True

        except Exception as e:
            error_msg = f"❌ Error starting trading session: {e}"
            logger.error(error_msg, exc_info=True)
            self.telegram_controller.send_message(error_msg)
            return False

    def stop_trading(self):
        """Stop the real-time trading session."""
        try:
            logger.info("🛑 Stopping real-time trading session...")
            self.streamer.stop_streaming()

            # --- Stop Telegram Polling ---
            self._is_polling = False
            if self.telegram_polling_thread and self.telegram_polling_thread.is_alive():
                logger.info("🛑 Stopping Telegram polling...")
                self.telegram_controller.stop_polling() # Signal the polling loop to stop
                # Optionally join with a timeout if it doesn't stop promptly
                # self.telegram_polling_thread.join(timeout=2)
            self.telegram_polling_thread = None

            # Clear internal state
            self.active_signals.clear()
            self.active_positions.clear() # If used
            was_trading = self.is_trading # Store state before changing it
            self.is_trading = False
            self.is_connected = False
            logger.info("🛑 Real-time trading session stopped")

            # Send session stop alert via Telegram controller's dedicated method
            # Only send stop message if it was actually trading
            if was_trading:
                 self.telegram_controller.send_realtime_session_alert("STOP")

        except Exception as e:
            error_msg = f"❌ Error stopping trading session: {e}"
            logger.error(error_msg, exc_info=True)
            self.telegram_controller.send_message(error_msg)

    # --- Helper method for Telegram polling thread ---
    def _run_telegram_polling(self):
        """Wrapper to run the Telegram controller's polling loop."""
        try:
            logger.debug("Starting Telegram controller polling loop...")
            self.telegram_controller.start_polling()
        except Exception as e:
            logger.error(f"❌ Unexpected error in Telegram polling thread: {e}", exc_info=True)
        finally:
            logger.info("🛑 Telegram polling thread finished.")

    def get_trading_status(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the trading engine."""
        try:
            uptime_seconds = time.time() - self.start_time if self.start_time and self.is_trading else 0
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            uptime_formatted = f"{hours}h {minutes}m {seconds}s"

            # Get streaming status from streamer
            streaming_status = self.streamer.get_connection_status() if hasattr(self.streamer, 'get_connection_status') else {'connected': self.is_connected}

            # Get processor buffer status
            processor_status = self.processor.get_buffer_status() if hasattr(self.processor, 'get_buffer_status') else {}

            # Get risk manager status
            risk_status = self.risk_manager.get_risk_status() if hasattr(self.risk_manager, 'get_risk_status') else {}

            return {
                "is_trading": self.is_trading,
                "execution_enabled": self.execution_enabled,
                "streaming_status": streaming_status,
                "active_signals": len(self.active_signals),
                "active_positions": len(self.active_positions), # Consider if this is the right count
                "trading_instruments_count": len(self.trading_instruments),
                "processor_status": processor_status,
                "risk_status": risk_status,
                "uptime_seconds": uptime_seconds,
                "uptime_formatted": uptime_formatted
            }
        except Exception as e:
            logger.error(f"❌ Error fetching trading status: {e}", exc_info=True)
            # Return a minimal status indicating the error
            return {
                "is_trading": self.is_trading,
                "execution_enabled": self.execution_enabled,
                "error": f"Status fetch error: {str(e)[:100]}" # Limit error string length
            }

# Example usage (if run directly)
if __name__ == "__main__":
    print("RealTimeTrader class defined. Use via src/main.py.")
