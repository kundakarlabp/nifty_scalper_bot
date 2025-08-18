# src/main.py
from __future__ import annotations

import logging
import signal
import sys
from threading import Event, Thread

# Optional import to avoid hard crash if kiteconnect not installed
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

from src.config import settings
from src.data.source import LiveKiteSource
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.server.health import run as run_health_server
from src.strategies.runner import StrategyRunner
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class Application:
    def __init__(self):
        self._stop_event = Event()
        self.settings = settings

        if KiteConnect is None:
            logger.critical("kiteconnect not installed. pip install kiteconnect")
            sys.exit(1)

        try:
            self.kite = KiteConnect(api_key=self.settings.api.zerodha_api_key)
            self.kite.set_access_token(self.settings.api.zerodha_access_token)
        except Exception as e:
            logger.critical(f"Kite init failed: {e}", exc_info=True)
            sys.exit(1)

        self.data_source = LiveKiteSource(self.kite)
        self.strategy = EnhancedScalpingStrategy(self.settings.strategy)
        self.executor = OrderExecutor(self.settings.executor, self.kite)
        self.sizer = PositionSizer(self.settings.risk)
        self.session = TradingSession(self.settings.risk, self.settings.executor, starting_equity=100000.0)

        self.telegram = None
        if self.settings.enable_telegram:
            self.telegram = TelegramController(
                config=self.settings.telegram,
                status_callback=self.get_status,
                control_callback=self.control_bot,
                summary_callback=lambda: f"Daily PnL: {self.session.daily_pnl:.2f}",
                # config callbacks are plugged by StrategyRunner init
            )

        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=self.telegram,
        )

    def start(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        health_thread = Thread(target=run_health_server, args=(self.get_status,), daemon=True)
        health_thread.start()

        if self.telegram:
            self.telegram.start_polling()

        self.runner.start()
        self._stop_event.wait()

    def shutdown(self, signum, frame):
        if not self._stop_event.is_set():
            logger.info(f"Shutting down ({signum})...")
            self.runner.stop()
            if self.telegram:
                self.telegram.stop_polling()
            self._stop_event.set()

    def get_status(self):
        return {
            "is_trading": self.runner._running if self.runner else False,
            "live_mode": self.settings.enable_live_trading,
            "open_positions": len(self.session.active_trades),
            "closed_today": len(self.session.trade_history),
            "daily_pnl": self.session.daily_pnl,
        }

    def control_bot(self, command: str, value: str) -> bool:
        # Extend as needed
        if command == "start":
            return True
        if command == "stop":
            self.shutdown("USR", None); return True
        if command == "mode":
            # wire your live/shadow toggle here
            return True
        if command == "panic":
            try:
                self.executor.cancel_all_orders(); return True
            except Exception:
                return False
        return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        app = Application()
        app.start()
    else:
        print("Usage: python3 -m src.main start", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
