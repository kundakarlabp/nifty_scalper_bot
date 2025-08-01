# src/core/real_time_trader.py

import logging
import threading
import atexit
import signal
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """
    Core trading engine for real-time scalping strategy.
    Manages trading state, signal processing, order execution, and Telegram control.
    """

    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = Config.ENABLE_LIVE_TRADING

        # Initialize strategy
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )

        # Initialize risk manager
        self.risk_manager = PositionSizing()

        # Initialize order executor
        self.order_executor = self._init_order_executor()

        # Initialize Telegram controller with callbacks
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )

        # Polling thread
        self._polling_thread: Optional[threading.Thread] = None

        # ✅ Start Telegram polling once at startup
        self._start_polling()

        # Register graceful shutdown
        atexit.register(self.shutdown)

    def _init_order_executor(self) -> OrderExecutor:
        """Initialize order executor based on live/simulation mode."""
        if not self.live_mode:
            logger.info("Live trading disabled. Using simulated order executor.")
            return OrderExecutor()

        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
            kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            logger.info("✅ Live order executor initialized with Kite Connect.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error("Failed to initialize live trading. Falling back to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()

    def start(self) -> bool:
        """Start trading (does NOT restart polling)."""
        if self.is_trading:
            logger.info("Trader already running.")
            return True

        self.is_trading = True
        try:
            self.telegram_controller.send_realtime_session_alert("START")
            logger.info("✅ Trading started.")
        except Exception as exc:
            logger.warning("Failed to send START alert: %s", exc)

        return True

    def stop(self) -> bool:
        """Stop trading only — keeps Telegram polling alive."""
        if not self.is_trading:
            logger.info("Trader is not running.")
            return True

        self.is_trading = False
        try:
            self.telegram_controller.send_realtime_session_alert("STOP")
            logger.info("🛑 Trading stopped. Telegram polling remains active.")
        except Exception as exc:
            logger.warning("Failed to send STOP alert: %s", exc)

        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        """Handle control commands from Telegram."""
        if command == "start":
            return self.start()
        elif command == "stop":
            return self.stop()
        elif command == "mode":
            return self._set_live_mode(arg)
        else:
            logger.warning("Unknown control command: %s", command)
            return False

    def _set_live_mode(self, mode: str) -> bool:
        """Switch between live and shadow (simulation) mode."""
        desired = mode.strip().lower() == "live"
        if desired == self.live_mode:
            return True
        if self.is_trading:
            logger.warning("Cannot change mode while trading is active. Stop trading first.")
            return False

        if desired:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("🟢 Switched to LIVE mode.")
                return True
            except Exception as exc:
                logger.error("Failed to switch to LIVE mode: %s", exc, exc_info=True)
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("🛡️ Switched to SHADOW (simulation) mode.")
            return True

    def _start_polling(self) -> None:
        """Start Telegram polling in a background thread — only once."""
        if self._polling_thread and self._polling_thread.is_alive():
            logger.debug("Polling thread already running.")
            return

        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning("Failed to send startup alert: %s", e)

        self._polling_thread = threading.Thread(
            target=self.telegram_controller.start_polling,
            daemon=True  # Dies when main thread dies
        )
        self._polling_thread.start()
        logger.info("✅ Telegram polling started (daemon).")

    def _stop_polling(self) -> None:
        """
        Stop Telegram polling (only call during app shutdown).
        Do NOT use in /stop command.
        """
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def shutdown(self) -> None:
        """Gracefully shut down the bot (call this on app exit)."""
        if not self.is_trading and (not self._polling_thread or not self._polling_thread.is_alive()):
            return  # Already shut down

        logger.info("👋 Shutting down RealTimeTrader...")
        self.stop()  # Stop trading
        self._stop_polling()  # Stop Telegram polling
        logger.info("✅ RealTimeTrader shutdown complete.")

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        """Process a new OHLC bar and generate trading signals."""
        if not self.is_trading:
            return
        if ohlc is None or len(ohlc) < 30:
            logger.debug("Insufficient data to process bar.")
            return

        try:
            # Validate index type
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                logger.error("OHLC data must have DatetimeIndex.")
                return

            ts = ohlc.index[-1]
            current_time_str = ts.strftime("%H:%M")

            # Time filter
            if Config.TIME_FILTER_START and Config.TIME_FILTER_END:
                if current_time_str < Config.TIME_FILTER_START or current_time_str > Config.TIME_FILTER_END:
                    return

            current_price = float(ohlc.iloc[-1]["close"])

            # Generate signal
            signal = self.strategy.generate_signal(ohlc, current_price)
            if not signal or float(signal.get("confidence", 0.0)) < Config.CONFIDENCE_THRESHOLD:
                return

            # Calculate position size
            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0),
            )
            if not position or position.get("quantity", 0) <= 0:
                return

            # Send signal alert
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)

            # Determine transaction type
            transaction_type = signal.get("signal") or signal.get("direction")
            if not transaction_type:
                logger.warning("Missing signal direction.")
                return

            # Place entry order
            symbol = getattr(Config, "TRADE_SYMBOL", "NIFTY50")
            exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")

            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                logger.warning("Failed to place entry order.")
                return

            # Setup GTT (bracket/OCO) orders
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

            # Record trade
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
        """Get current trading status for Telegram /status command."""
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        # Merge risk manager status
        status.update(self.risk_manager.get_risk_status())
        return status

    def get_summary(self) -> str:
        """Get daily trade summary for Telegram /summary command."""
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
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<RealTimeTrader is_trading={self.is_trading} "
                f"live_mode={self.live_mode} trades_today={len(self.trades)}>")