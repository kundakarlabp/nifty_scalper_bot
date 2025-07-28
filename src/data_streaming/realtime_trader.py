import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import pytz
from kiteconnect import KiteConnect
import requests
from src.data_streaming.market_data_streamer import MarketDataStreamer
from src.data_streaming.data_processor import StreamingDataProcessor
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.notifications.telegram_controller import TelegramController
from config import (ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN,
                   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BASE_STOP_LOSS_POINTS,
                   BASE_TARGET_POINTS, CONFIDENCE_THRESHOLD, ACCOUNT_SIZE,
                   RISK_PER_TRADE, MAX_DRAWDOWN)

logger = logging.getLogger(__name__)

class RealTimeTrader:
    def __init__(self):
        self.kite = None
        self.is_connected = False
        self.is_trading = False
        self.execution_enabled = False
        self.timezone = pytz.timezone('Asia/Kolkata')
        self.trading_instruments = []
        self.active_signals = {}
        self.active_positions = {}
        self.start_time = None
        
        # Initialize components
        self.streamer = MarketDataStreamer()
        self.processor = StreamingDataProcessor()
        self.strategy = DynamicScalpingStrategy(
            base_stop_loss_points=BASE_STOP_LOSS_POINTS,
            base_target_points=BASE_TARGET_POINTS,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        self.risk_manager = PositionSizing(
            account_size=ACCOUNT_SIZE,
            risk_per_trade=RISK_PER_TRADE,
            max_drawdown=MAX_DRAWDOWN
        )
        self.telegram_controller = TelegramController()
        
        # Set up streamer callbacks
        self.streamer.set_ticks_callback(self._handle_ticks)
        self.streamer.set_connect_callback(self._handle_connect)
        self.streamer.set_close_callback(self._handle_close)
        self.streamer.set_error_callback(self._handle_error)
        
        self.setup_kite()
    
    def setup_kite(self):
        """Initialize Kite Connect"""
        try:
            if ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN:
                self.kite = KiteConnect(api_key=ZERODHA_API_KEY)
                self.kite.set_access_token(ZERODHA_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("✅ Kite Connect initialized for order execution")
            else:
                logger.warning("⚠️  Zerodha credentials not available for order execution")
        except Exception as e:
            logger.error(f"❌ Error initializing Kite Connect: {e}")
            self.is_connected = False
    
    def _handle_ticks(self, ticks):
        """Handle incoming market data ticks"""
        try:
            for tick in ticks:
                # Process the tick
                processed_tick = self.processor.process_tick(tick)
                
                if processed_tick:
                    token = processed_tick['instrument_token']
                    
                    # Update OHLC data periodically
                    current_time = time.time()
                    last_update = getattr(self, f'_last_ohlc_update_{token}', 0)
                    
                    if current_time - last_update > 60:  # Update every minute
                        self.processor.update_ohlc(token, '1min')
                        setattr(self, f'_last_ohlc_update_{token}', current_time)
                        
                        # Check for trading signals
                        self._check_trading_signals(token)
                
        except Exception as e:
            logger.error(f"Error handling ticks: {e}")
    
    def _handle_connect(self, response):
        """Handle WebSocket connection"""
        logger.info("✅ Connected to market data stream")
        # Resubscribe to instruments if needed
        if self.trading_instruments:
            self.streamer.subscribe_tokens(self.trading_instruments)
    
    def _handle_close(self, code, reason):
        """Handle WebSocket closure"""
        logger.info(f" WebSocket closed. Code: {code}, Reason: {reason}")
    
    def _handle_error(self, code, reason):
        """Handle WebSocket errors"""
        logger.error(f" WebSocket error. Code: {code}, Reason: {reason}")
    
    def _check_trading_signals(self, token: int):
        """Check for trading signals for a token"""
        try:
            # Get latest OHLC data
            ohlc_data = self.processor.get_latest_data(token, 100)
            if ohlc_data is None or len(ohlc_data) < 50:
                return
            
            # Get current price
            current_price = self.processor.get_current_price(token)
            if current_price is None:
                return
            
            # Generate signal
            signal = self.strategy.generate_signal(ohlc_data, current_price)
            
            if signal and signal['confidence'] >= CONFIDENCE_THRESHOLD:
                # Check if we already have an active signal for this token
                if token not in self.active_signals:
                    self._handle_trading_signal(token, signal)
                else:
                    logger.info(f"Signal already active for token {token}")
            
        except Exception as e:
            logger.error(f"Error checking trading signals for token {token}: {e}")
    
    def _handle_trading_signal(self, token: int, signal: Dict):
        """Handle generated trading signal"""
        try:
            logger.info(f"🎯 Trading signal generated for token {token}: {signal['signal']}")
            
            # Store active signal
            self.active_signals[token] = {
                'signal': signal,
                'timestamp': time.time(),
                'status': 'pending'
            }
            
            # Calculate position size
            position_info = self.risk_manager.calculate_position_size(
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                signal_confidence=signal['confidence'],
                market_volatility=signal['market_volatility']
            )
            
            if position_info['quantity'] > 0:
                # Log the signal
                signal_message = f"""
                🎯 REAL-TIME TRADING SIGNAL
                
                📊 Token: {token}
                📈 Direction: {signal['signal']}
                💰 Entry Price: {signal['entry_price']:.2f}
                �� Stop Loss: {signal['stop_loss']:.2f}
                ✅ Target: {signal['target']:.2f}
                🔥 Confidence: {signal['confidence']*100:.1f}%
                🌊 Volatility: {signal['market_volatility']:.2f}
                📦 Recommended: {position_info['lots']} lots ({position_info['quantity']} qty)
                
                📝 Reasons: {', '.join(signal['reasons'][:3])}
                """
                
                logger.info(signal_message)
                
                # Send Telegram alert
                self.telegram_controller.send_message(signal_message)
                
                # Execute trade if enabled
                if self.execution_enabled:
                    # In a real implementation, you would execute the trade here
                    # self._execute_trade(token, signal, position_info)
                    pass
                else:
                    logger.info("⚠️  Trade execution disabled (simulation mode)")
                
                # Update signal status
                self.active_signals[token]['status'] = 'executed'
                self.active_signals[token]['position_info'] = position_info
                
            else:
                logger.warning(f"⚠️ Position size is zero for signal on token {token}")
                self.active_signals[token]['status'] = 'rejected'
                
        except Exception as e:
            logger.error(f"Error handling trading signal for token {token}: {e}")
    
    def add_trading_instrument(self, token: int) -> bool:
        """Add instrument for real-time trading"""
        try:
            if token not in self.trading_instruments:
                self.trading_instruments.append(token)
                logger.info(f"✅ Added trading instrument: {token}")
                
                # Subscribe if streamer is connected
                if self.streamer.is_connected:
                    self.streamer.subscribe_tokens([token])
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error adding trading instrument {token}: {e}")
            return False
    
    def remove_trading_instrument(self, token: int) -> bool:
        """Remove instrument from real-time trading"""
        try:
            if token in self.trading_instruments:
                self.trading_instruments.remove(token)
                logger.info(f"✅ Removed trading instrument: {token}")
                
                # Unsubscribe if streamer is connected
                if self.streamer.is_connected:
                    self.streamer.unsubscribe_tokens([token])
                
                # Remove active signals for this token
                if token in self.active_signals:
                    del self.active_signals[token]
                
                # Clear data processor buffer
                self.processor.clear_buffer(token)
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing trading instrument {token}: {e}")
            return False
    
    def enable_trading(self, enable: bool = True):
        """Enable or disable real trading execution"""
        self.execution_enabled = enable
        logger.info(f"{'✅' if enable else '⚠️'} Trade execution {'enabled' if enable else 'disabled'}")
    
    def start_trading(self) -> bool:
        """Start real-time trading"""
        try:
            if not self.trading_instruments:
                logger.warning("⚠️ No trading instruments configured")
                return False
            
            # Initialize streamer
            if not self.streamer.initialize_connection():
                logger.error("❌ Failed to initialize market data streamer")
                return False
            
            # Subscribe to trading instruments
            self.streamer.subscribe_tokens(self.trading_instruments)
            
            # Start streaming
            if not self.streamer.start_streaming():
                logger.error("❌ Failed to start market data streaming")
                return False
            
            self.is_trading = True
            self.start_time = time.time()
            logger.info("✅ Real-time trading started successfully")
            
            # Send Telegram alert
            self.telegram_controller.send_message("✅ Real-time trading started successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting real-time trading: {e}")
            return False
    
    def stop_trading(self):
        """Stop real-time trading"""
        try:
            self.is_trading = False
            self.streamer.stop_streaming()
            
            # Clear active signals and positions
            self.active_signals.clear()
            self.active_positions.clear()
            
            logger.info("✅ Real-time trading stopped")
            
            # Send Telegram alert
            self.telegram_controller.send_message("🛑 Real-time trading stopped")
            
        except Exception as e:
            logger.error(f"Error stopping real-time trading: {e}")
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        try:
            uptime = 0
            if self.start_time:
                uptime = time.time() - self.start_time
            
            return {
                'is_trading': self.is_trading,
                'execution_enabled': self.execution_enabled,
                'streaming_status': self.streamer.get_connection_status(),
                'active_signals': len(self.active_signals),
                'active_positions': len(self.active_positions),
                'trading_instruments': len(self.trading_instruments),
                'processor_status': self.processor.get_buffer_status(),
                'risk_status': self.risk_manager.get_risk_status(),
                'uptime_seconds': uptime,
                'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
            }
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    trader = RealTimeTrader()
    
    # Add Nifty 50 for trading (token 256265)
    trader.add_trading_instrument(256265)
    
    print("Real-time trader initialized. Ready for trading.")
    print(f"Trading status: {trader.get_trading_status()}")
