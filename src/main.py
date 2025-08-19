# src/main.py
from __future__ import annotations

import logging
import os
import signal
import sys
from threading import Event, Thread
from typing import Dict, Any

from src.config import settings
from src.data.source import LiveKiteSource
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.server.health import run as run_health_server
from src.strategies.runner import StrategyRunner
from src.utils import strike_selector as sel
from src.utils.kite_auth import build_kite_from_env
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

_LOG_LEVEL = getattr(logging, str(settings.log_level).upper(), logging.INFO)
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


class Application:
    def __init__(self) -> None:
        self._stop_event = Event()
        self.settings = settings

        os.environ.setdefault("TZ", "Asia/Kolkata")

        # Build authenticated Kite client (validates token or auto-generates)
        try:
            self.kite = build_kite_from_env()
        except Exception as e:
            logger.critical("Kite init/auth failed: %s", e, exc_info=True)
            sys.exit(1)

        self.data_source = LiveKiteSource(self.kite)
        self.strategy = EnhancedScalpingStrategy(self.settings.strategy)
        self.executor = OrderExecutor(self.settings.executor, self.kite)
        self.sizer = PositionSizer(self.settings.risk)
        self.session = TradingSession(self.settings.risk, self.settings.executor, starting_equity=100_000.0)

        self.telegram = None
        if self.settings.enable_telegram and self.settings.telegram.bot_token:
            self.telegram = TelegramController(
                bot_token=self.settings.telegram.bot_token,
                chat_id=self.settings.telegram.chat_id,
            )
            self.telegram.set_context(session=self.session, executor=self.executor, health_getter=self.get_health)

        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=self.telegram,
        )

        self._runner_thread: Thread | None = None
        self._health_thread: Thread | None = None

    def start(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._health_thread = Thread(target=run_health_server, args=(self.get_health,), daemon=True)
        self._health_thread.start()

        if self.telegram:
            self.telegram.start()

        self._runner_thread = Thread(target=self.runner.start, name="runner", daemon=True)
        self._runner_thread.start()

        logger.info("Application started. Live mode: %s, TZ=%s", self.settings.enable_live_trading, os.getenv("TZ"))
        self._stop_event.wait()
        logger.info("Application exiting.")

    def _on_signal(self, signum, _frame) -> None:
        logger.info("Signal received: %s – shutting down…", signum)
        self.stop()

    def stop(self) -> None:
        if not self._stop_event.is_set():
            try: self.runner.stop()
            except Exception: pass
            try:
                if self.telegram: self.telegram.stop()
            except Exception: pass
            self._stop_event.set()

    def get_health(self) -> Dict[str, Any]:
        try:
            base = sel.health_check(self.kite)
        except Exception as e:
            base = {"overall_status": "ERROR", "message": f"health_check failed: {e}", "checks": {}}
        base["runner"] = {"running": bool(self.runner and self.runner._running), "live_mode": bool(self.settings.enable_live_trading)}
        base["session"] = {
            "open_positions": len(self.session.active_trades),
            "trades_today": self.session.trades_today,
            "daily_pnl": self.session.daily_pnl,
            "consecutive_losses": self.session.consecutive_losses,
        }
        return base


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        app = Application()
        app.start()
    else:
        print("Usage: python -m src.main start", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
