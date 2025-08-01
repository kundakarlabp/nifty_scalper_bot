
import logging
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

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

    def _init_order_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled. Using simulated order executor.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
            kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            logger.info("Live order executor initialized.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error("Failed to initialize live trading. Falling back to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()

    def start(self) -> bool:
        if self.is_trading:
            logger.info("Trader already running.")
            return True
        self.is_trading = True
        self._start_polling()
        try:
            self.telegram_controller.send_realtime_session_alert("START")
        except Exception:
            pass
        logger.info("Trading started.")
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            logger.info("Trader is not running.")
            return True
        self.is_trading = False
        # DO NOT stop polling here
        try:
            self.telegram_controller.send_realtime_session_alert("STOP")
        except Exception:
            pass
       logger.info("Trading stopped.")
       return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        if command == "start":
            return self.start()
        elif command == "stop":
            return self.stop()
        elif command == "mode":
            return self._set_live_mode(arg)
        logger.warning("Unknown control command: %s", command)
        return False

    def _set_live_mode(self, mode: str) -> bool:
        desired = mode.strip().lower() == "live"
        if desired == self.live_mode:
            return True
        if self.is_trading:
            logger.info("Cannot change mode while trading is active. Stop trading first.")
            return False
        if desired:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("Switched to LIVE mode.")
                return True
            except Exception as exc:
                logger.error("Failed to switch to LIVE mode: %s", exc, exc_info=True)
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("Switched to SHADOW mode.")
            return True

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        self.telegram_controller.send_startup_alert()
        self._polling_thread = threading.Thread(target=self.telegram_controller.start_polling, daemon=True)
        self._polling_thread.start()

    def _stop_polling(self) -> None:
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=2)
        self._polling_thread = None

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        if not self.is_trading or ohlc is None or len(ohlc) < 30:
            return
        try:
            ts = ohlc.index[-1]
            current_time_str = ts.strftime("%H:%M")
            if Config.TIME_FILTER_START and Config.TIME_FILTER_END:
                if current_time_str < Config.TIME_FILTER_START or current_time_str > Config.TIME_FILTER_END:
                    return

            current_price = float(ohlc.iloc[-1]["close"])
            signal = self.strategy.generate_signal(ohlc, current_price)
            if not signal or float(signal.get("confidence", 0.0)) < Config.CONFIDENCE_THRESHOLD:
                return

            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0),
            )
            if not position or position.get("quantity", 0) <= 0:
                return

            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)
            transaction_type = signal.get("signal") or signal.get("direction")

            order_id = self.order_executor.place_entry_order(
                symbol="NIFTY50",
                exchange="NFO",
                transaction_type=transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                return

            self.order_executor.setup_gtt_orders(
                entry_order_id=order_id,
                entry_price=signal.get("entry_price", current_price),
                stop_loss_price=signal.get("stop_loss", current_price),
                target_price=signal.get("target", current_price),
                symbol="NIFTY50",
                exchange="NFO",
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
        lines = [f"Total trades: {len(self.trades)}", f"PNL: {self.daily_pnl:.2f}"]
        for trade in self.trades:
            lines.append(
                f"{trade['direction']} {trade['quantity']} @ {trade['entry_price']:.2f} "
                f"(SL {trade['stop_loss']:.2f}, TP {trade['target']:.2f})"
            )
        return "\n".join(lines)
