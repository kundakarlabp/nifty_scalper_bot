"""
Application entry point for the Nifty Scalper Bot.
Initializes and starts all components of the trading application.
"""

from __future__ import annotations

import logging
import signal
import sys
from threading import Event, Thread
from typing import Optional

from kiteconnect import KiteConnect

from src.config import settings
from src.server.health import run as run_health_server

# Optional imports that may not exist in all setups; fail gracefully.
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore

try:
    from src.execution.order_executor import OrderExecutor  # type: ignore
except Exception:  # pragma: no cover
    OrderExecutor = None  # type: ignore

try:
    from src.notifications.telegram_controller import TelegramController  # type: ignore
except Exception:  # pragma: no cover
    TelegramController = None  # type: ignore

try:
    from src.risk.session import TradingSession  # type: ignore
except Exception:  # pragma: no cover
    TradingSession = None  # type: ignore

# Strategy + Sizer: support either naming found in repo
try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy as StrategyClass
except Exception:  # pragma: no cover
    from src.strategies.scalping_strategy import DynamicScalpingStrategy as StrategyClass  # type: ignore

try:
    from src.risk.position_sizing import PositionSizer as SizerClass  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from src.risk.position_sizing import PositionSizing as SizerClass  # type: ignore


# ---------------------- logging -----------------------------------------
def _setup_logging() -> None:
    level_name = str(getattr(settings, "log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
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

        logger.info(
            "Starting Nifty Scalper Bot | live_trading=%s",
            bool(self.settings.enable_live_trading),
        )

        # --- 1) KiteConnect (required for live trading) ---
        self.kite: Optional[KiteConnect] = None
        try:
            # Only *require* credentials when live trading is enabled.
            if self.settings.enable_live_trading:
                if not self.settings.api.zerodha_api_key or not self.settings.api.zerodha_access_token:
                    raise ValueError("ZERODHA_API_KEY/ACCESS_TOKEN not configured for live trading.")
                self.kite = KiteConnect(api_key=self.settings.api.zerodha_api_key)
                self.kite.set_access_token(self.settings.api.zerodha_access_token)
                logger.info("KiteConnect initialized for live trading.")
            else:
                # Shadow/offline mode: initialize if creds are present, otherwise skip.
                if self.settings.api.zerodha_api_key and self.settings.api.zerodha_access_token:
                    self.kite = KiteConnect(api_key=self.settings.api.zerodha_api_key)
                    self.kite.set_access_token(self.settings.api.zerodha_access_token)
                    logger.info("KiteConnect initialized (shadow mode).")
                else:
                    logger.info("KiteConnect not initialized (shadow mode, no credentials).")
        except Exception as e:
            logger.critical("Failed to initialize KiteConnect: %s", e, exc_info=True)
            if self.settings.enable_live_trading:
                sys.exit(1)

        # --- 2) Core services (DI) ---
        self.data_source = None
        if LiveKiteSource is not None and self.kite is not None:
            try:
                self.data_source = LiveKiteSource(self.kite)  # type: ignore[call-arg]
                self.data_source.connect()
                logger.info("Live data source connected.")
            except Exception as e:
                logger.critical("Failed to connect data source: %s", e, exc_info=True)
                if self.settings.enable_live_trading:
                    sys.exit(1)
        else:
            logger.warning("Data source not available (no Kite or module missing). Running without live feed.")

        # Strategy & sizer should construct with no args (they read global settings internally)
        try:
            self.strategy = StrategyClass()
        except TypeError:
            # If your StrategyClass expects a config object, fall back
            self.strategy = StrategyClass(self.settings.strategy)  # type: ignore

        try:
            self.sizer = SizerClass()
        except TypeError:
            self.sizer = SizerClass(self.settings.risk)  # type: ignore

        # Order executor (optional in shadow mode)
        self.executor = None
        if OrderExecutor is not None:
            try:
                self.executor = OrderExecutor(self.settings.executor, self.kite)  # type: ignore
            except Exception as e:
                logger.error("OrderExecutor init failed: %s", e, exc_info=True)

        # Trading session (required by runner)
        if TradingSession is not None:
            try:
                self.session = TradingSession(  # type: ignore
                    self.settings.risk,
                    getattr(self.settings, "executor", None),
                    starting_equity=100000.0,
                )
            except Exception as e:
                logger.critical("TradingSession init failed: %s", e, exc_info=True)
                sys.exit(1)
        else:  # pragma: no cover
            logger.critical("TradingSession module missing.")
            sys.exit(1)

        # --- 3) Telegram Controller (optional) ---
        self.telegram_controller = None
        if self.settings.enable_telegram and TelegramController is not None:
            try:
                self.telegram_controller = TelegramController(  # type: ignore
                    config=self.settings.telegram,
                    status_callback=self.get_status,
                    control_callback=self.control_bot,
                    summary_callback=lambda: f"Daily PnL: {getattr(self.session, 'daily_pnl', 0.0):.2f}",
                )
                logger.info("TelegramController initialized.")
            except Exception as e:
                logger.error("Failed to initialize TelegramController: %s", e, exc_info=True)

        # --- 4) Strategy Runner (main loop orchestrator) ---
        try:
            from src.strategies.runner import StrategyRunner  # local import to avoid hard failure
        except Exception as e:  # pragma: no cover
            logger.critical("StrategyRunner import failed: %s", e, exc_info=True)
            sys.exit(1)

        try:
            self.runner = StrategyRunner(
                data_source=self.data_source,
                strategy=self.strategy,
                order_executor=self.executor,
                trading_session=self.session,
                position_sizer=self.sizer,
                telegram_controller=self.telegram_controller,
            )
        except Exception as e:
            logger.critical("StrategyRunner init failed: %s", e, exc_info=True)
            sys.exit(1)

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
                self.telegram_controller.start_polling()  # type: ignore[attr-defined]
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
            if getattr(self, "runner", None):
                try:
                    self.runner.stop()
                except Exception:
                    pass
            if getattr(self, "telegram_controller", None):
                try:
                    self.telegram_controller.stop_polling()  # type: ignore[attr-defined]
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
            "daily_pnl": float(getattr(self.session, "daily_pnl", 0.0)),
        }

    def control_bot(self, command: str, arg: str) -> bool:
        logger.info("Received control command: %s arg=%s", command, arg)
        if command == "stop" and getattr(self, "runner", None):
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
