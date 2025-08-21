# src/main.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner

# Optional broker SDK (keep imports safe)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

# Optional DataSource (not required if StrategyRunner manages its own source)
try:
    from src.data.source import LiveKiteSource, DataSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore
    DataSource = object  # type: ignore


log = logging.getLogger(__name__)


# ------------ Logging helper ------------
def _setup_logging() -> None:
    """Configure centralized logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------ time helper ------------
def _now_ist_naive() -> datetime:
    """Return current naive datetime in IST."""
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ------------ nested attr helper ------------
def _get_nested(obj: Any, *path: str) -> Optional[Any]:
    """Safely get a nested attribute from an object."""
    cur = obj
    for p in path:
        cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        # runner will be (re)constructed below so we can inject the event sink
        self.runner: Optional[StrategyRunner] = None
        self.tg = None  # TelegramController instance (optional)

        # for uptime in /ping
        self._start_ts = time.time()

        # graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- status payload for health/Telegram
    def _status_payload(self) -> dict:
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(getattr(settings, "enable_live_trading", False)),
            "broker": "Kite" if self.runner and self.runner._kite else "none",
            "active_orders": (len(self.runner.executor.get_active_orders()) if self.runner and self.runner.executor else 0),
        }

    def run(self) -> None:
        """Main entry point for the application."""
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", bool(getattr(settings, "enable_live_trading", False)))

        # Health server (guarded inside health module)
        health_thread = threading.Thread(
            target=health_server.run,
            kwargs={
                "callback": self._status_payload,
                "host": settings.server.host,
                "port": settings.server.port,
            },
            daemon=True,
        )
        health_thread.start()

        # --- Build runner with an event sink that forwards to Telegram (if available) ---
        def _runner_event_sink(evt: dict) -> None:
            try:
                if not self.tg:  # Telegram may not be started (no creds)
                    return
                t = str(evt.get("type"))
                if t == "ENTRY_PLACED":
                    self.tg.notify_entry(
                        symbol=evt.get("symbol", "?"),
                        side=evt.get("side", "?"),
                        qty=int(evt.get("qty", 0)),
                        price=float(evt.get("price", 0.0)),
                        record_id=str(evt.get("record_id", "")),
                    )
                elif t == "FILLS":
                    fills = evt.get("fills") or []
                    self.tg.notify_fills(fills)
            except Exception:
                log.exception("Failed to forward runner event to Telegram.")

        self.runner = StrategyRunner(event_sink=_runner_event_sink)

        # --- Telegram controller (with flat env fallbacks) ---
        try:
            from src.notifications.telegram_controller import TelegramController

            tg_enabled = (
                bool(_get_nested(settings, "telegram", "enabled")) or
                os.getenv("ENABLE_TELEGRAM", "true").lower() in ("1", "true", "yes")
            )
            bot_token = _get_nested(settings, "telegram", "bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id_raw = _get_nested(settings, "telegram", "chat_id") or os.getenv("TELEGRAM_CHAT_ID")
            chat_id = int(chat_id_raw) if chat_id_raw else None

            if tg_enabled and bot_token and chat_id:
                # patch settings at runtime so TelegramController's own settings read succeeds
                try:
                    settings.telegram.bot_token = bot_token  # type: ignore[attr-defined]
                    settings.telegram.chat_id = chat_id      # type: ignore[attr-defined]
                except Exception:
                    pass

                self.tg = TelegramController(
                    status_provider=self.runner.to_status_dict,
                    positions_provider=(self.runner.executor.get_positions_kite if self.runner and self.runner.executor else None),
                    actives_provider=(self.runner.executor.get_active_orders if self.runner and self.runner.executor else None),
                    runner_pause=self.runner.pause,
                    runner_resume=self.runner.resume,
                    cancel_all=(self.runner.executor.cancel_all_orders if self.runner and self.runner.executor else None),
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")

        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main trading loop (StrategyRunner handles cadence internally if needed)
        cadence = 0.5  # responsive sleep
        while not self._stop_event.is_set():
            try:
                result = self.runner.run_once(stop_event=self._stop_event)
                if result:
                    log.info("Signal: %s", result)
            except (NetworkException, TokenException, InputException) as e:
                log.error("Transient broker error: %s", e)
            except Exception as e:
                log.exception("Main loop error: %s", e)

            if self._stop_event.wait(timeout=cadence):
                break

        # Teardown
        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        log.info("Nifty Scalper Bot stopped.")


# ------------ cli ------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest from CSV file")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"

    if cmd == "start":
        app = Application()
        app.run()
    elif cmd == "backtest":
        log.info("Backtest command not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())