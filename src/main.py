from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.config import settings
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner

# Optional broker (paper mode works without it)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


# ---------------- Logging ----------------
def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
        root.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(ch)
    # quiet noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _now_ist() -> str:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- App ----------------
class Application:
    def __init__(self) -> None:
        _setup_logging()
        self.log = logging.getLogger("manage_bot")
        self._stop_event = threading.Event()

        # 1) Telegram is **mandatory**
        if not settings.telegram.enabled:
            raise RuntimeError("TelegramController is required (settings.telegram.enabled=False). Enable it in .env.")
        if not settings.telegram.bot_token or not settings.telegram.chat_id:
            raise RuntimeError(
                "TelegramController is required and must be configured: "
                "set TELEGRAM__BOT_TOKEN and TELEGRAM__CHAT_ID in .env"
            )

        self.tg = TelegramController(
            bot_token=settings.telegram.bot_token,
            chat_id=str(settings.telegram.chat_id),
        )
        self.tg.start_polling()

        # 2) Optional broker for live trading
        self.kite: Optional[KiteConnect] = None
        if settings.enable_live_trading:
            if not KiteConnect:
                raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect is not available in the environment.")
            self.kite = KiteConnect(api_key=settings.zerodha.api_key)
            self.kite.set_access_token(settings.zerodha.access_token)

        # 3) Runner requires Telegram
        self.runner = StrategyRunner(kite=self.kite, telegram_controller=self.tg)

        # Signals for graceful shutdown
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self.log.info("Starting trader (shadow mode by default)")
        self.tg.send_message(f"🤖 Bot starting at {_now_ist()} | live={settings.enable_live_trading}")

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self.log.info("Signal %s received, shutting down…", signum)
        self._stop_event.set()

    def run(self) -> None:
        """Simple heartbeat loop; StrategyRunner does work on its own tick/feeds."""
        hb_every = 300.0
        last_hb = 0.0
        try:
            while not self._stop_event.wait(timeout=0.5):
                # keep runner healthy even if no ticks arrive
                try:
                    self.runner.health_check()
                except Exception as e:
                    logging.getLogger(__name__).warning("Runner health warning: %s", e)

                now = time.time()
                if now - last_hb > hb_every:
                    self.log.info("⏱ heartbeat | time_ist=%s live=%s", _now_ist(), settings.enable_live_trading)
                    last_hb = now
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.runner.shutdown()
        except Exception:
            pass
        try:
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass
        self.log.info("Nifty Scalper Bot stopped.")


def main() -> int:
    app = Application()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())