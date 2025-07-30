# src/data_streaming/realtime_trader.py
"""
Core real-time trading engine integrating data streaming, strategy,
risk management, execution (via OrderExecutor), and Telegram notifications.
"""
import sys
import os
import logging
import time
import pytz
from typing import Dict, List, Any, Optional
from kiteconnect import KiteConnect

# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from src.data_streaming.market_data_streamer import MarketDataStreamer
from src.data_streaming.data_processor import StreamingDataProcessor
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)

class RealTimeTrader:
    def __init__(self, order_executor: Optional[OrderExecutor] = None):
        self.kite: Optional[KiteConnect] = None
        self.order_executor: Optional[OrderExecutor] = order_executor
        self.is_trading: bool = False
        self.execution_enabled: bool = False
        self.is_connected: bool = False
        self.start_time: Optional[float] = None
        self.timezone = pytz.timezone("Asia/Kolkata")

        self.trading_instruments: List[int] = []
        self.token_symbol_map: Dict[int, Dict[str, str]] = {}  # token -> {symbol, exchange}
        self.active_signals: Dict[int, Dict[str, Any]] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        self.streamer = MarketDataStreamer()
        self.processor = StreamingDataProcessor()
        self.strategy = DynamicScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        self.risk_manager = PositionSizing(
            account_size=Config.ACCOUNT_SIZE,
            risk_per_trade=Config.RISK_PER_TRADE,
            max_drawdown=Config.MAX_DRAWDOWN
        )
        self.telegram_controller = TelegramController()

        self.streamer.set_ticks_callback(self._handle_ticks)
        self.streamer.set_connect_callback(self._handle_connect)
        self.streamer.set_close_callback(self._handle_close)
        self.streamer.set_error_callback(self._handle_error)

        self.setup_kite()

    def setup_kite(self):
        try:
            if Config.ZERODHA_API_KEY and Config.KITE_ACCESS_TOKEN:
                self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.is_connected = True
                logger.info("✅ Zerodha Kite Connect initialized.")
            else:
                logger.warning("⚠️ Missing Zerodha API key or access token.")
        except Exception as e:
            logger.error(f"❌ Kite Connect setup error: {e}")

    def _handle_connect(self, _):
        logger.info("📡 WebSocket connected")
        if self.trading_instruments:
            self.streamer.subscribe_tokens(self.trading_instruments)

    def _handle_close(self, code, reason):
        logger.warning(f"🔌 WebSocket closed | Code: {code}, Reason: {reason}")

    def _handle_error(self, code, reason):
        logger.error(f"⚠️ WebSocket error | Code: {code}, Reason: {reason}")

    def _handle_ticks(self, ticks):
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
            logger.error(f"❌ Tick handler error: {e}")

    def _check_trading_signals(self, token: int):
        try:
            ohlc_data = self.processor.get_latest_data(token, 100)
            if not ohlc_data or len(ohlc_data) < 50:
                return

            current_price = self.processor.get_current_price(token)
            if current_price is None:
                return

            signal = self.strategy.generate_signal(ohlc_data, current_price)
            if signal and signal.get('confidence', 0) >= Config.CONFIDENCE_THRESHOLD:
                if token not in self.active_signals:
                    self._handle_trading_signal(token, signal)
        except Exception as e:
            logger.error(f"❌ Signal check failed for token {token}: {e}")

    def _handle_trading_signal(self, token: int, signal: Dict[str, Any]):
        try:
            logger.info(f"🎯 Signal for token {token}: {signal['signal']}")
            position_details = self.risk_manager.calculate_position_size(
                entry_price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                signal_confidence=signal['confidence'],
                market_volatility=signal['market_volatility']
            )

            if not position_details or position_details.get('quantity', 0) <= 0:
                logger.warning(f"⚠️ Position size invalid for token {token}: {position_details}")
                return

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

            execution_status = "DISABLED"
            if self.execution_enabled and self.order_executor and self.kite:
                instrument_info = self.token_symbol_map.get(token)
                if not instrument_info:
                    logger.error(f"❌ Cannot execute: No symbol/exchange mapping for token {token}")
                    execution_status = "FAILED_MAPPING"
                else:
                    symbol = instrument_info["symbol"]
                    exchange = instrument_info["exchange"]
                    transaction_type = signal['signal']

                    entry_order_id = self.order_executor.place_entry_order(
                        symbol=symbol,
                        exchange=exchange,
                        transaction_type=transaction_type,
                        quantity=position_details['quantity']
                    )

                    if entry_order_id:
                        time.sleep(2)
                        filled_entry_price = signal['entry_price']
                        gtt_success = self.order_executor.setup_gtt_orders(
                            entry_order_id=entry_order_id,
                            entry_price=filled_entry_price,
                            stop_loss_price=signal['stop_loss'],
                            target_price=signal['target'],
                            symbol=symbol,
                            exchange=exchange,
                            quantity=position_details['quantity'],
                            transaction_type=transaction_type
                        )
                        if gtt_success:
                            logger.info("✅ Entry order and GTTs placed successfully")
                            execution_status = "SUCCESS"
                            self.risk_manager.update_position_status(is_open=True)
                        else:
                            logger.error("❌ Failed to place GTT orders")
                            execution_status = "FAILED_GTT"
                    else:
                        logger.error("❌ Failed to place entry order")
                        execution_status = "FAILED_ENTRY"
            else:
                if not self.execution_enabled:
                    logger.info("⚠️ Execution is disabled (simulation mode)")
                    execution_status = "SIMULATED"
                elif not self.order_executor:
                    logger.warning("⚠️ OrderExecutor not available")
                    execution_status = "NO_EXECUTOR"
                elif not self.kite:
                    logger.error("❌ Kite client not initialized")
                    execution_status = "NO_KITE"

            self.active_signals[token] = {
                "signal": signal,
                "position_info": position_details,
                "timestamp": time.time(),
                "status": "processed",
                "execution_status": execution_status
            }

        except Exception as e:
            logger.error(f"❌ Error handling signal for token {token}: {e}", exc_info=True)

    def add_trading_instrument(self, token: int, symbol: str, exchange: str) -> bool:
        try:
            if token not in self.trading_instruments:
                self.trading_instruments.append(token)
                self.token_symbol_map[token] = {"symbol": symbol, "exchange": exchange}
                logger.info(f"➕ Token added: {token} -> {symbol} ({exchange})")
                if self.streamer.is_connected:
                    self.streamer.subscribe_tokens([token])
            return True
        except Exception as e:
            logger.error(f"❌ Add instrument error: {e}")
            return False

    def enable_trading(self, enable: bool = True):
        self.execution_enabled = enable
        status = "enabled" if enable else "disabled"
        logger.info(f"{'✅' if enable else '⚠️'} Trading execution {status}")

    def start_trading(self) -> bool:
        try:
            if not self.trading_instruments:
                logger.warning("⚠️ No tokens configured for trading")
                return False

            if not self.streamer.initialize_connection():
                logger.error("❌ Failed to initialize WebSocket connection")
                return False

            self.streamer.subscribe_tokens(self.trading_instruments)

            if not self.streamer.start_streaming():
                logger.error("❌ Failed to start WebSocket streaming")
                return False

            self.is_trading = True
            self.start_time = time.time()
            logger.info("✅ Real-time trading session started")
            self.telegram_controller.send_message("✅ Real-time trading session started")
            return True

        except Exception as e:
            logger.error(f"❌ Error starting trading session: {e}")
            return False

    def stop_trading(self):
        try:
            self.streamer.stop_streaming()
            self.active_signals.clear()
            self.active_positions.clear()
            self.is_trading = False
            logger.info("🛑 Real-time trading session stopped")
            self.telegram_controller.send_message("🛑 Real-time trading session stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping trading session: {e}")

    def get_trading_status(self) -> Dict[str, Any]:
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

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/realtime_trader.log"),
            logging.StreamHandler()
        ]
    )
    trader = RealTimeTrader()
    print("RealTimeTrader initialized. Use start_trading() to begin.")
