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
from src.risk.session import TradingSession
from src.server.health import run as run_health_server
from src.strategies.runner import StrategyRunner

# Strategy + Sizer: support either naming found in repo
try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy as StrategyClass
except Exception:
    from src.strategies.scalping_strategy import DynamicScalpingStrategy as StrategyClass  # fallback

try:
    from src.risk.position_sizing import PositionSizer as SizerClass
except Exception:
    from src.risk.position_sizing import PositionSizing as SizerClass  # fallback


# ---------------------- logging -----------------------------------------
def _setup_logging() -> None:
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("kiteconnect").setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger(__name__)


class Application:
    """
    Main application: initialize dependencies, start services, and manage lifecycle.
    """

    def __init__(self):
        self._stop_event = Event()
        self.settings = settings

        logger.info("Starting Nifty Scalper Bot | live_trading=%s  TZ=%s",
                    self.settings.enable_live_trading, self.settings.tz)

        # --- 1) KiteConnect (required for live trading) ---
        self.kite: KiteConnect | None = None
        try:
            if not self.settings.api.zerodha_api_key or not self.settings.api.zerodha_access_token:
                raise ValueError("ZERODHA_API_KEY/ACCESS_TOKEN not configured.")
            self.kite = KiteConnect(api_key=self.settings.api.zerodha_api_key)
            self.kite.set_access_token(self.settings.api.zerodha_access_token)
            logger.info("KiteConnect initialized.")
        except Exception as e:
            logger.critical("Failed to initialize KiteConnect: %s", e, exc_info=True)
            sys.exit(1)

        # --- 2) Core services (DI) ---
        self.data_source = LiveKiteSource(self.kite)
        try:
            self.data_source.connect()
        except Exception as e:
            logger.critical("Failed to connect data source: %s", e, exc_info=True)
            sys.exit(1)

        self.strategy = StrategyClass(self.settings.strategy)
        self.executor = OrderExecutor(self.settings.executor, self.kite)
        self.sizer = SizerClass(self.settings.risk)

        # TODO: wire real starting equity or broker funds
        self.session = TradingSession(
            self.settings.risk,
            self.settings.executor,
            starting_equity=100000.0,
        )

        # --- 3) Telegram Controller (optional) ---
        self.telegram_controller: TelegramController | None = None
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
                logger.error("Failed to initialize TelegramController: %s", e, exc_info=True)

        # --- 4) Strategy Runner (main loop orchestrator) ---
        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=self.telegram_controller,
        )
        logger.info("All components initialized.")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Starts components and waits for shutdown signal."""
        logger.info("Starting Nifty Scalper Bot Application...")

        # Graceful shutdown handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

        # Start optional components
        if self.telegram_controller:
            try:
                self.telegram_controller.start_polling()
            except Exception as e:
                logger.error("Telegram polling start failed: %s", e, exc_info=True)

        # Health server (daemon)
        try:
            health_thread = Thread(target=run_health_server, args=(self.get_status,), daemon=True)
            health_thread.start()
        except Exception as e:
            logger.error("Health server failed to start: %s", e, exc_info=True)

        # Main trading loop (blocking until stop)
        try:
            self.runner.start()
        except Exception as e:
            logger.critical("Runner crashed: %s", e, exc_info=True)
            self.shutdown(signal.SIGTERM, None)

        # Wait for shutdown signal
        self._stop_event.wait()
        logger.info("Application shutdown complete.")

    def shutdown(self, signum, frame) -> None:
        """Handles graceful shutdown of the application."""
        _ = frame  # Unused
        if not self._stop_event.is_set():
            logger.info("Received shutdown signal %s. Stopping components...", signum)
            if self.runner:
                try:
                    self.runner.stop()
                except Exception:
                    pass
            if self.telegram_controller:
                try:
                    self.telegram_controller.stop_polling()
                except Exception:
                    pass
            self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Telegram callbacks
    # ------------------------------------------------------------------ #
    def get_status(self) -> dict:
        return {
            "is_trading": bool(getattr(self.runner, "_running", False)),
            "live_mode": bool(self.settings.enable_live_trading),
            "open_positions": len(getattr(self.session, "active_trades", [])),
            "closed_today": len(getattr(self.session, "trade_history", [])),
            "daily_pnl": getattr(self.session, "daily_pnl", 0.0),
        }

    def control_bot(self, command: str, arg: str) -> bool:
        logger.info("Received control command: %s arg=%s", command, arg)
        if command == "stop" and self.runner:
            self.runner.stop()
            return True
        # 'start' from Telegram is ignored; runner is already active.
        return False


def main() -> None:
    """Main function to run the bot."""
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        app = Application()
        app.start()
    else:
        print("Usage: python -m src.main start", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
