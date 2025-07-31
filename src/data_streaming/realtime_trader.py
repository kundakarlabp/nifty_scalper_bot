from __future__ import annotations

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
        self.last_signal: Optional[Dict[str, Any]] = None
        self.current_position: Optional[Dict[str, Any]] = None

        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        )
        self.risk_manager = PositionSizing()
        self.order_executor = OrderExecutor()

        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )

        self._polling_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self.is_trading:
            logger.info("Trader is already running.")
            return True
        self.is_trading = True
        self._start_polling()
        logger.info("Trading started.")
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            logger.info("Trader is not running.")
            return True
        self.is_trading = False
        self._stop_polling()
        logger.info("Trading stopped.")
        return True

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        if not self.is_trading:
            return
        if ohlc is None or len(ohlc) < 30:
            return
        try:
            current_price = ohlc.iloc[-1]["close"]
            signal = self.strategy.generate_signal(ohlc, current_price)
            if signal and signal.get("confidence", 0) >= Config.CONFIDENCE_THRESHOLD:
                self.last_signal = signal
                position = self.risk_manager.calculate_position_size(
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    signal_confidence=signal["confidence"],
                    market_volatility=signal.get("market_volatility", 0),
                )
                if not position or position.get("quantity", 0) <= 0:
                    return
                self.current_position = position
                self.telegram_controller.send_signal_alert(0, signal, position)
                order_id = self.order_executor.place_entry_order(
                    symbol="NIFTY50",
                    exchange="NFO",
                    transaction_type=signal["signal"],
                    quantity=position["quantity"],
                )
                if not order_id:
                    return
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=signal["entry_price"],
                    stop_loss_price=signal["stop_loss"],
                    target_price=signal["target"],
                    symbol="NIFTY50",
                    exchange="NFO",
                    quantity=position["quantity"],
                    transaction_type=signal["signal"],
                )
                self.trades.append(
                    {
                        "order_id": order_id,
                        "direction": signal["signal"],
                        "quantity": position["quantity"],
                        "entry_price": signal["entry_price"],
                        "stop_loss": signal["stop_loss"],
                        "target": signal["target"],
                        "confidence": signal["confidence"],
                    }
                )
        except Exception as exc:
            logger.error("Error processing bar: %s", exc, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "last_signal": self.last_signal or {},
            "trade_history": self.trades,
            "open_position": self.current_position or {},
            "live_pnl": self.daily_pnl,
            **self.risk_manager.get_risk_status(),
        }

    def get_summary(self) -> str:
        lines = [f"Total trades: {len(self.trades)}", f"PNL: {self.daily_pnl:.2f}"]
        for trade in self.trades:
            lines.append(
                f"{trade['direction']} {trade['quantity']} @ {trade['entry_price']:.2f} "
                f"(SL {trade['stop_loss']:.2f}, TP {trade['target']:.2f})"
            )
        return "\n".join(lines)

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        self.telegram_controller.send_startup_alert()
        self._polling_thread = threading.Thread(target=self.telegram_controller.start_polling, daemon=True)
        self._polling_thread.start()

    def _stop_polling(self) -> None:
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)
        self._polling_thread = None

    def _handle_control(self, command: str) -> bool:
        if command == "start":
            return self.start()
        if command == "stop":
            return self.stop()
        if command == "resetday":
            self.risk_manager.reset_day()
            return True
        logger.warning("Unknown control command: %s", command)
        return False
