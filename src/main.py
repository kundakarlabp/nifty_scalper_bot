# src/main.py
"""
Application entry point for the Nifty Scalper Bot.
Initializes and starts all components of the trading application.
"""

from __future__ import annotations

import logging
import signal
import sys
from threading import Event, Thread

from kiteconnect import KiteConnect

from src.config import settings
from src.data.source import LiveKiteSource
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.server.health import run as run_health_server
from src.strategies.runner import StrategyRunner
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

# Configure logging based on settings
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class Application:
    """
    The main application class. Responsible for initializing, starting,
    and gracefully shutting down all bot components.
    """

    def __init__(self):
        self._stop_event = Event()
        self.settings = settings

        # --- Initialize Components ---
        logger.info("Initializing application components...")

        # 1. Kite Connect API
        try:
            self.kite = KiteConnect(api_key=self.settings.api.zerodha_api_key)
            self.kite.set_access_token(self.settings.api.zerodha_access_token)
            logger.info("KiteConnect initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize KiteConnect: {e}", exc_info=True)
            sys.exit(1)

        # 2. Core Components (using dependency injection)
        self.data_source = LiveKiteSource(self.kite)
        self.strategy = EnhancedScalpingStrategy(self.settings.strategy)
        self.executor = OrderExecutor(self.settings.executor, self.kite)
        self.sizer = PositionSizer(self.settings.risk)

        # TODO: Get starting equity from a real source
        self.session = TradingSession(self.settings.risk, self.settings.executor, starting_equity=100000.0)

        # 3. Telegram Controller (optional)
        self.telegram_controller = None
        if self.settings.enable_telegram:
            try:
                self.telegram_controller = TelegramController(
                    config=self.settings.telegram,
                    status_callback=self.get_status,
                    control_callback=self.control_bot,
                    summary_callback=lambda: f"Daily PnL: {self.session.daily_pnl:.2f}",
                )
                logger.info("TelegramController initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize TelegramController: {e}", exc_info=True)

        # 4. Strategy Runner (the main loop)
        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=self.telegram_controller,
        )
        logger.info("All components initialized.")

    def start(self):
        """Starts all application components and waits for a shutdown signal."""
        logger.info("Starting Nifty Scalper Bot Application...")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

        # Start optional components
        if self.telegram_controller:
            self.telegram_controller.start_polling()

        health_thread = Thread(target=run_health_server, args=(self.get_status,), daemon=True)
        health_thread.start()

        # Start the main trading loop
        self.runner.start()

        # Wait for shutdown signal
        self._stop_event.wait()
        logger.info("Application shutdown process complete.")

    def shutdown(self, signum, frame):
        """Handles graceful shutdown of the application."""
        _ = frame  # Unused
        if not self._stop_event.is_set():
            logger.info(f"Received shutdown signal {signum}. Shutting down gracefully...")
            if self.runner:
                self.runner.stop()
            if self.telegram_controller:
                self.telegram_controller.stop_polling()
            self._stop_event.set()

    # --- Telegram Callbacks ---
    def get_status(self):
        return {
            "is_trading": self.runner._running if self.runner else False,
            "live_mode": self.settings.enable_live_trading,
            "open_positions": len(self.session.active_trades),
            "closed_today": len(self.session.trade_history),
            "daily_pnl": self.session.daily_pnl,
        }

    def control_bot(self, command: str, arg: str):
        logger.info(f"Received control command: {command} with arg: {arg}")
        # This is a simplified control path. A more robust implementation
        # would use a proper command pattern.
        if command == "stop":
            self.runner.stop()
            return True
        # The 'start' command from telegram is tricky as the runner is already running.
        # This needs a more sophisticated state machine, but for now we can ignore it.
        return False


def main():
    """Main function to run the bot."""
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        app = Application()
        app.start()
    else:
        print("Usage: python3 -m src.main start", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()