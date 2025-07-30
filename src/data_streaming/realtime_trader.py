# src/data_streaming/realtime_trader.py
"""
Core real-time trading engine integrating data streaming, strategy,
risk management, execution (via OrderExecutor), and Telegram notifications.
"""
import sys
import os
# Ensure correct path resolution for imports
# Consider if this is still necessary based on your project structure and deployment method (e.g., Docker)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import time
import pytz
from typing import Dict, List, Any, Optional
from kiteconnect import KiteConnect

# Import configuration using the Config class for consistency
# This assumes your config.py defines a Config class
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
        self.order_executor: Optional[OrderExecutor] = order_executor # Store the executor
        self.is_trading: bool = False
        self.execution_enabled: bool = False
        self.is_connected: bool = False
        self.start_time: Optional[float] = None
        self.timezone = pytz.timezone("Asia/Kolkata")

        self.trading_instruments: List[int] = []
        # Store mapping from token to symbol/exchange for order placement
        self.token_symbol_map: Dict[int, Dict[str, str]] = {}
        self.active_signals: Dict[int, Dict[str, Any]] = {}
        # This can be used to track positions if needed at this level,
        # though OrderExecutor also tracks them.
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        # Initialize core modules
        self.streamer = MarketDataStreamer()
        self.processor = StreamingDataProcessor()
        
        # Initialize Strategy
        # Pass parameters that scalping_strategy.py's __init__ now accepts.
        self.strategy = DynamicScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
            # Add other strategy parameters from Config if needed and accepted by __init__
            # ema_fast_period=Config.EMA_FAST_PERIOD, ...
        )
        
        self.risk_manager = PositionSizing(
            account_size=Config.ACCOUNT_SIZE,
            risk_per_trade=Config.RISK_PER_TRADE,
            max_drawdown=Config.MAX_DRAWDOWN
        )
        self.telegram_controller = TelegramController()

        # Register callbacks for the streamer
        self.streamer.set_ticks_callback(self._handle_ticks)
        self.streamer.set_connect_callback(self._handle_connect)
        self.streamer.set_close_callback(self._handle_close)
        self.streamer.set_error_callback(self._handle_error)

        # Setup Kite Connect API client
        self.setup_kite()

    def setup_kite(self):
        """Initialize the Kite Connect client with credentials."""
        try:
            if Config.ZERODHA_API_KEY and Config.KITE_ACCESS_TOKEN:
                self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("✅ Zerodha Kite Connect initialized.")
            else:
                logger.warning("⚠️ Missing Zerodha API key or access token in Config.")
        except Exception as e:
            logger.error(f"❌ Kite Connect setup error: {e}")

    # --- WebSocket Callbacks ---

    def _handle_connect(self, _):
        """Callback when WebSocket connects."""
        logger.info("📡 WebSocket connected")
        if self.trading_instruments:
            self.streamer.subscribe_tokens(self.trading_instruments)

    def _handle_close(self, code, reason):
        """Callback when WebSocket closes."""
        logger.warning(f"🔌 WebSocket closed | Code: {code}, Reason: {reason}")

    def _handle_error(self, code, reason):
        """Callback when WebSocket encounters an error."""
        logger.error(f"⚠️ WebSocket error | Code: {code}, Reason: {reason}")

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
                # Use a dictionary to track last tick time per token more cleanly
                last_tick_time = getattr(self, f'_last_tick_time_{token}', 0)

                # Update OHLC and check signals every minute per token
                if now - last_tick_time >= 60:
                    setattr(self, f'_last_tick_time_{token}', now)
                    self.processor.update_ohlc(token, '1min')
                    self._check_trading_signals(token)

        except Exception as e:
            logger.error(f"❌ Tick handler error: {e}")

    # --- Signal & Execution Logic ---

    def _check_trading_signals(self, token: int):
        """
        Fetch latest data and generate/check for trading signals.
        """
        try:
            # Fetch last 100 candles for analysis
            ohlc_data = self.processor.get_latest_data(token, 100)
            if not ohlc_data or len(ohlc_data) < 50: # Ensure enough data
                return

            current_price = self.processor.get_current_price(token)
            if current_price is None:
                return

            # Generate signal using the strategy module
            # Pass the current_price explicitly as required by the updated strategy
            signal = self.strategy.generate_signal(ohlc_data, current_price)

            # If a valid signal is generated and meets confidence threshold from Config
            if signal and signal.get('confidence', 0) >= Config.CONFIDENCE_THRESHOLD:
                # Avoid duplicate signals for the same token
                if token not in self.active_signals:
                    self._handle_trading_signal(token, signal)

        except Exception as e:
            logger.error(f"❌ Signal check failed for token {token}: {e}")

    def _handle_trading_signal(self, token: int, signal: Dict[str, Any]):
        """
        Process a generated signal: calculate position size, send alerts,
        and potentially execute (if enabled).
        """
        try:
            logger.info(f"🎯 Signal for token {token}: {signal['signal']}")

            # Calculate position size based on risk management rules
            position_details = self.risk_manager.calculate_position_size(
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                signal_confidence=signal['confidence'],
                market_volatility=signal['market_volatility']
            )

            # Skip if position size calculation failed or is zero
            if not position_details or position_details.get('quantity', 0) <= 0:
                logger.warning(f"⚠️ Position size invalid for token {token}: {position_details}")
                return

            # Format and send Telegram alert
            alert_message = f"""
🎯 *REAL-TIME SIGNAL*
📈 Token: {token}
📊 Direction: {signal['signal']}
💰 Entry: {signal['entry_price']:.2f}
🛑 SL: {signal['stop_loss']:.2f}
🎯 Target: {signal['target']:.2f}
🔥 Confidence: {signal['confidence']*100:.1f}%
🌊 Volatility: {signal['market_volatility']:.2f}
📦 Qty: {position_details['quantity']} ({position_details['lots']} lots)
🧠 Reason: {', '.join(signal.get('reasons', [])[:3])}
            """
            self.telegram_controller.send_message(alert_message)
            logger.info(f"Signal Alert Sent:\n{alert_message}")

            # Handle Execution
            execution_status = "DISABLED"
            if self.execution_enabled and self.order_executor and self.kite:
                logger.info("💼 Initiating live execution via OrderExecutor...")
                
                # Get symbol and exchange from the pre-populated map
                instrument_info = self.token_symbol_map.get(token)
                if not instrument_info:
                    logger.error(f"❌ Cannot execute: No symbol/exchange mapping for token {token}")
                    execution_status = "FAILED_MAPPING"
                else:
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
                        # 2. Wait for order fill confirmation (simplified)
                        # In a real scenario, you'd poll kite.order_history or listen via OMS WebSocket
                        # This is a placeholder wait. Consider a more robust check.
                        time.sleep(2)
                        # Get filled price (simplified, use actual order history)
                        # For now, using the signal's entry price as a placeholder
                        filled_entry_price = signal['entry_price']

                        # 3. Setup GTT Orders
                        gtt_success = self.order_executor.setup_gtt_orders(
                            entry_order_id=entry_order_id,
                            entry_price=filled_entry_price,
                            stop_loss_price=signal['stop_loss'],
                            target_price=signal['target'],
                            symbol=symbol, # Use actual symbol from selection
                            exchange=exchange, # Use actual exchange
                            quantity=position_details['quantity'],
                            transaction_type=transaction_type # Must match entry order
                        )
                        if gtt_success:
                            logger.info("✅ Entry order and GTTs placed successfully via OrderExecutor")
                            execution_status = "SUCCESS"
                            # Update risk manager that a position is open
                            self.risk_manager.update_position_status(is_open=True)
                        else:
                            logger.error("❌ Failed to place GTT orders via OrderExecutor")
                            execution_status = "FAILED_GTT"
                            # TODO: Consider cancelling the entry order if GTTs failed
                    else:
                        logger.error("❌ Failed to place entry order via OrderExecutor")
                        execution_status = "FAILED_ENTRY"
            else:
                if not self.execution_enabled:
                    logger.info("⚠️ Execution is disabled (simulation mode)")
                    execution_status = "SIMULATED"
                elif not self.order_executor:
                    logger.warning("⚠️ OrderExecutor not available for execution")
                    execution_status = "NO_EXECUTOR"
                elif not self.kite:
                    logger.error("❌ Kite client not initialized for execution")
                    execution_status = "NO_KITE"

            # Store the active signal regardless of execution outcome for tracking
            self.active_signals[token] = {
                "signal": signal,
                "position_info": position_details,
                "timestamp": time.time(),
                "status": "processed",
                "execution_status": execution_status # Track execution outcome
            }

        except Exception as e:
            logger.error(f"❌ Error handling signal for token {token}: {e}", exc_info=True)
            # Optionally store error status in active_signals
            self.active_signals[token] = {
                "signal": signal,
                "position_info": position_details if 'position_details' in locals() else {},
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
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
                # Store the mapping for order execution
                self.token_symbol_map[token] = {"symbol": symbol, "exchange": exchange}
                logger.info(f"➕ Token added: {token} -> {symbol} ({exchange})")
                # Subscribe immediately if streamer is already connected
                if self.streamer.is_connected:
                    self.streamer.subscribe_tokens([token])
            return True
        except Exception as e:
            logger.error(f"❌ Add instrument error: {e}")
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
                # Remove mapping
                self.token_symbol_map.pop(token, None)
                # Unsubscribe if streamer is connected
                if self.streamer.is_connected:
                    self.streamer.unsubscribe_tokens([token])
                # Clean up related data
                self.processor.clear_buffer(token)
                self.active_signals.pop(token, None)
                # Optionally remove from active_positions if implemented
            return True
        except Exception as e:
            logger.error(f"❌ Remove instrument error: {e}")
            return False

    def enable_trading(self, enable: bool = True):
        """Enable or disable actual order execution."""
        self.execution_enabled = enable
        status = "enabled" if enable else "disabled"
        logger.info(f"{'✅' if enable else '⚠️'} Trading execution {status}")
        # Optionally notify via Telegram
        # self.telegram_controller.send_message(f"{'✅' if enable else '⚠️'} Trading execution {status}")

    def start_trading(self) -> bool:
        """Start the real-time trading session."""
        try:
            if not self.trading_instruments:
                logger.warning("⚠️ No tokens configured for trading")
                return False

            # Initialize and start the data stream
            if not self.streamer.initialize_connection():
                logger.error("❌ Failed to initialize WebSocket connection")
                return False

            self.streamer.subscribe_tokens(self.trading_instruments)

            if not self.streamer.start_streaming():
                logger.error("❌ Failed to start WebSocket streaming")
                return False

            # Set internal state
            self.is_trading = True
            self.start_time = time.time()
            logger.info("✅ Real-time trading session started")
            self.telegram_controller.send_message("✅ Real-time trading session started")
            return True

        except Exception as e:
            logger.error(f"❌ Error starting trading session: {e}")
            return False

    def stop_trading(self):
        """Stop the real-time trading session."""
        try:
            self.streamer.stop_streaming()
            # Clear internal state
            self.active_signals.clear()
            self.active_positions.clear() # If used
            self.is_trading = False
            logger.info("🛑 Real-time trading session stopped")
            self.telegram_controller.send_message("🛑 Real-time trading session stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping trading session: {e}")

    def get_trading_status(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the trading engine."""
        try:
            uptime_seconds = time.time() - self.start_time if self.start_time else 0
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            uptime_formatted = f"{hours}h {minutes}m {seconds}s"

            return {
                "is_trading": self.is_trading,
                "execution_enabled": self.execution_enabled,
                "streaming_status": self.streamer.get_connection_status(),
                "active_signals": len(self.active_signals),
                "active_positions": len(self.active_positions),
                "trading_instruments_count": len(self.trading_instruments),
                "processor_status": self.processor.get_buffer_status(),
                "risk_status": self.risk_manager.get_risk_status(),
                "uptime_seconds": uptime_seconds,
                "uptime_formatted": uptime_formatted
            }
        except Exception as e:
            logger.error(f"❌ Error fetching trading status: {e}")
            return {}

# Example usage (if run directly)
if __name__ == "__main__":
    # Ensure logs directory exists if using the default logging setup here
    # os.makedirs("logs", exist_ok=True)
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     handlers=[
    #         logging.FileHandler("logs/realtime_trader.log"),
    #         logging.StreamHandler()
    #     ]
    # )

    # Note: Running this directly won't perform trades without a full setup
    # including Kite credentials, OrderExecutor, and selected instruments.
    # The main entry point is src/main.py
    print("RealTimeTrader class defined. Use via src/main.py.")
