# src/data_streaming/realtime_trader.py

import logging
import threading
import atexit
import signal
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
# Import the schedule library
import schedule

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

# Corrected logger instantiation
logger = logging.getLogger(__name__)

class RealTimeTrader:
    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = Config.ENABLE_LIVE_TRADING

        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )
        self.risk_manager = PositionSizing()
        self.order_executor = self._init_order_executor()
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        self._polling_thread: Optional[threading.Thread] = None

        # Start Telegram polling in a daemon thread
        self._start_polling()

        # Schedule the data fetching and processing task
        # Adjust the frequency (e.g., '1' minute) according to your strategy's needs.
        # process_bar checks for Config.TIME_FILTER_START/END, so frequent checks are usually okay.
        schedule.every(1).minutes.do(self.fetch_and_process_data)
        logger.info("Scheduled fetch_and_process_data to run every 1 minute.")

        atexit.register(self.shutdown)
        logger.info("RealTimeTrader initialized and ready to receive commands.")

    def _init_order_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled. Using simulated order executor.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
            kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            logger.info("✅ Live order executor initialized with Kite Connect.")
            return OrderExecutor(kite=kite)
        except Exception as exc: # Catching generic exception is generally okay here for fallback logic
            logger.error("Failed to initialize live trading. Falling back to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()

    def start(self) -> bool:
        if self.is_trading:
            logger.info("Trader already running.")
            self.telegram_controller.send_message("🛑 Trader already running.")
            return True
        self.is_trading = True
        try:
            self.telegram_controller.send_realtime_session_alert("START")
            logger.info("✅ Trading started.")
        except Exception as exc:
            logger.warning("Failed to send START alert: %s", exc)
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            logger.info("Trader is not running.")
            self.telegram_controller.send_message("🛑 Trader is already stopped.")
            return True
        self.is_trading = False
        try:
            self.telegram_controller.send_realtime_session_alert("STOP")
            logger.info("🛑 Trading stopped. Telegram polling remains active.")
        except Exception as exc:
            logger.warning("Failed to send STOP alert: %s", exc)
        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = command.strip().lower()
        arg = arg.strip().lower() if arg else ""
        logger.info(f"Received command: /{command} {arg}")
        if command == "start":
            return self.start()
        elif command == "stop":
            return self.stop()
        elif command == "mode":
            if arg not in ["live", "shadow"]:
                logger.warning("Invalid mode argument: %s", arg)
                self.telegram_controller.send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                return False
            return self._set_live_mode(arg)
        else:
            logger.warning("Unknown control command: %s", command)
            self.telegram_controller.send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
            return False

    def _set_live_mode(self, mode: str) -> bool:
        desired_live = (mode == "live")
        if desired_live == self.live_mode:
            current_mode = "LIVE" if self.live_mode else "SHADOW"
            logger.info(f"Already in {current_mode} mode.")
            self.telegram_controller.send_message(f"🟢 Already in *{current_mode}* mode.", parse_mode="Markdown")
            return True
        if self.is_trading:
            logger.warning("Cannot change mode while trading is active. Stop trading first.")
            self.telegram_controller.send_message("🛑 Cannot change mode while trading. Use `/stop` first.", parse_mode="Markdown")
            return False
        if desired_live:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("🟢 Switched to LIVE mode.")
                self.telegram_controller.send_message("🚀 Switched to *LIVE* trading mode.", parse_mode="Markdown")
                return True
            except Exception as exc: # Catching generic exception is generally okay here for user feedback
                logger.error("Failed to switch to LIVE mode: %s", exc, exc_info=True)
                # Ensure the message is properly formatted on one line if needed
                self.telegram_controller.send_message(
                    f"❌ Failed to switch to LIVE mode: `{str(exc)[:100]}...` Reverted to SHADOW mode.", parse_mode="Markdown"
                )
                self.live_mode = False
                self.order_executor = OrderExecutor()
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("🛡️ Switched to SHADOW (simulation) mode.")
            self.telegram_controller.send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
            return True

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            logger.debug("Polling thread already running.")
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning("Failed to send startup alert: %s", e)
        self._polling_thread = threading.Thread(
            target=self.telegram_controller.start_polling,
            daemon=True # Correctly set as daemon thread
        )
        self._polling_thread.start()
        logger.info("✅ Telegram polling started (daemon).")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            # Avoid joining from within the same thread
            if threading.current_thread() != self._polling_thread:
                 # Use a timeout to prevent hanging
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def shutdown(self) -> None:
        # Check if shutdown is necessary
        if not self.is_trading and (not self._polling_thread or not self._polling_thread.is_alive()):
            return
        logger.info("👋 Shutting down RealTimeTrader...")
        self.stop() # Stop trading logic
        self._stop_polling() # Stop Telegram polling
        logger.info("✅ RealTimeTrader shutdown complete.")

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        logger.debug(f"process_bar called. Trading active: {self.is_trading}, OHLC data points: {len(ohlc) if ohlc is not None else 'None'}")
        if not self.is_trading:
            logger.debug("process_bar: Trading not active, returning.")
            return
        if ohlc is None or len(ohlc) < 30:
            logger.debug("Insufficient data to process bar (less than 30 points).")
            return
        try:
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                logger.error("OHLC data must have DatetimeIndex.")
                return
            ts = ohlc.index[-1]
            current_time_str = ts.strftime("%H:%M")
            if Config.TIME_FILTER_START and Config.TIME_FILTER_END:
                if current_time_str < Config.TIME_FILTER_START or current_time_str > Config.TIME_FILTER_END:
                    logger.debug(f"Time filter active ({Config.TIME_FILTER_START} - {Config.TIME_FILTER_END}). Current time {current_time_str} is outside range, skipping bar.")
                    return
            current_price = float(ohlc.iloc[-1]["close"])
            logger.debug(f"Current bar timestamp: {ts}, price: {current_price}")
            signal = self.strategy.generate_signal(ohlc, current_price)
            logger.debug(f"Strategy returned signal: {signal}")
            if not signal:
                logger.debug("No signal generated by strategy.")
                return
            signal_confidence = float(signal.get("confidence", 0.0))
            logger.debug(f"Signal confidence: {signal_confidence}, Threshold: {Config.CONFIDENCE_THRESHOLD}")
            if signal_confidence < Config.CONFIDENCE_THRESHOLD:
                logger.debug("Signal confidence below threshold, discarding.")
                return
            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0),
            )
            logger.debug(f"Position sizing returned: {position}")
            if not position or position.get("quantity", 0) <= 0:
                logger.debug("Position sizing failed or quantity is zero/negative.")
                return
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)
            transaction_type = signal.get("signal") or signal.get("direction")
            if not transaction_type:
                logger.warning("Missing signal direction.")
                return
            symbol = getattr(Config, "TRADE_SYMBOL", "NIFTY50")
            exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")
            logger.debug(f"Attempting to place entry order. Symbol: {symbol}, Exchange: {exchange}, Type: {transaction_type}, Qty: {position['quantity']}")
            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                logger.warning("Failed to place entry order.")
                return
            logger.debug("Attempting to setup GTT orders...")
            self.order_executor.setup_gtt_orders(
                entry_order_id=order_id,
                entry_price=signal.get("entry_price", current_price),
                stop_loss_price=signal.get("stop_loss", current_price),
                target_price=signal.get("target", current_price),
                symbol=symbol,
                exchange=exchange,
                quantity=position["quantity"],
                transaction_type=transaction_type,
            )
            self.trades.append({
                "order_id": order_id,
                "direction": transaction_type,
                "quantity": position["quantity"],
                "entry_price": signal.get("entry_price", current_price),
                "stop_loss": signal.get("stop_loss", current_price),
                "target": signal.get("target", current_price),
                "confidence": signal.get("confidence", 0.0),
            })
            logger.info(f"✅ Trade recorded: {transaction_type} {position['quantity']} @ {signal.get('entry_price', current_price)}")
        except Exception as exc:
            logger.error("Error processing bar: %s", exc, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        status.update(self.risk_manager.get_risk_status())
        return status

    def get_summary(self) -> str:
        lines = [
            f"📊 <b>Daily Summary</b>",
            f"🔁 <b>Total trades:</b> {len(self.trades)}",
            f"💰 <b>PNL:</b> {self.daily_pnl:.2f}",
            "────────────────────"
        ]
        for trade in self.trades:
            lines.append(
                f"{trade['direction']} {trade['quantity']} @ {trade['entry_price']:.2f} "
                f"(SL {trade['stop_loss']:.2f}, TP {trade['target']:.2f})"
            )
        return "\n".join(lines) # Ensure lines are joined with newlines

    # --- NEW METHOD TO FETCH DATA AND TRIGGER PROCESSING ---
    def fetch_and_process_data(self):
        """
        Fetches the latest OHLC data and triggers the processing logic.
        This method will be scheduled to run periodically.
        """
        logger.debug("fetch_and_process_data triggered by schedule.")
        if not self.is_trading:
             logger.debug("fetch_and_process_data: Trading not active, skipping.")
             return

        try:
            # --- REAL KITE DATA FETCHING SECTION ---
            from datetime import datetime, timedelta
            import pandas as pd

            # 1. Define instrument using the token from Config
            # Ensure INSTRUMENT_TOKEN is set correctly in your .env file for the desired instrument
            instrument_token = Config.INSTRUMENT_TOKEN # e.g., 256265 for NIFTY 50 Index

            # 2. Define the timeframe for historical data (adjust the 35 as needed)
            end_time = datetime.now()
            # Fetch last 35 minutes of 1-minute data
            start_time = end_time - timedelta(minutes=35)

            # 3. Fetch historical data using the KiteConnect instance from order_executor
            # Ensure self.order_executor.kite exists and is initialized (it should be if live_mode is True or was set to live)
            if not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                 logger.error("KiteConnect instance not found in order_executor. Cannot fetch data. Is live mode enabled?")
                 return # Or handle appropriately

            logger.debug(f"Fetching historical data for instrument token: {instrument_token}")
            ohlc_data_list = self.order_executor.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_time,
                to_date=end_time,
                interval="minute" # Fetch 1-minute candles
            )

            # 4. Convert to DataFrame
            if not ohlc_data_list:
                logger.warning("fetch_and_process_data: KiteConnect returned empty data list.")
                return

            ohlc_df = pd.DataFrame(ohlc_data_list)

            # 5. Ensure the DataFrame has the correct structure
            # Kite data usually has 'date', 'open', 'high', 'low', 'close', 'volume'
            # Set 'date' as the index and ensure it's datetime
            if 'date' in ohlc_df.columns:
                ohlc_df['date'] = pd.to_datetime(ohlc_df['date'])
                ohlc_df.set_index('date', inplace=True)
                # Rename columns if necessary to match process_bar's expectations (though they usually match)
                # ohlc_df.rename(columns={'open': 'open', 'high': 'high', ...}, inplace=True)
            else:
                logger.error("Fetched data does not contain 'date' column.")
                return

            logger.debug(f"Fetched real OHLC data with {len(ohlc_df)} bars from KiteConnect.")

            # 6. Pass the fetched data to process_bar
            if ohlc_df is not None and not ohlc_df.empty:
                self.process_bar(ohlc_df)
            else:
                logger.warning("fetch_and_process_data: No data to process after fetching (real).")

        except Exception as e:
            logger.error(f"Error in fetch_and_process_data (KiteConnect fetch): {e}", exc_info=True)
        # --- END OF REAL KITE DATA FETCHING SECTION ---

    # --- END OF NEW METHOD ---

    def __repr__(self) -> str:
        return (f"<RealTimeTrader is_trading={self.is_trading} "
                f"live_mode={self.live_mode} trades_today={len(self.trades)}>")
