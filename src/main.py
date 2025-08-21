from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (keep imports safe)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks


log = logging.getLogger(__name__)


# ------------ time helper ------------
def _now_ist_naive() -> datetime:
    """Return current naive datetime in IST."""
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ------------ ring buffer log handler ------------
class RingBufferHandler(logging.Handler):
    """
    In-memory ring buffer of recent log lines for /logs.
    Cheap and thread-safe enough for our needs.
    """
    def __init__(self, capacity: int = 4000) -> None:
        super().__init__()
        self._buf: Deque[str] = deque(maxlen=max(100, capacity))
        self._lock = threading.Lock()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        self._formatter = logging.Formatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self._formatter.format(record)
            with self._lock:
                self._buf.append(line)
        except Exception:
            pass

    def tail(self, n: int = 200) -> str:
        n = max(1, min(n, self._buf.maxlen or 2000))
        with self._lock:
            return "\n".join(list(self._buf)[-n:])


_RING_HANDLER: Optional[RingBufferHandler] = None


def get_recent_logs(n: int = 200) -> str:
    if __RING_HANDLER:
        return __RING_HANDLER.tail(n)
    return "no log buffer"


# ------------ Logging helper ------------
def _setup_logging() -> None:
    """Configure centralized logging."""
    global __RING_HANDLER
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(root.level)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(ch)

    # Ring buffer for /logs
    __RING_HANDLER = RingBufferHandler(capacity=6000)
    __RING_HANDLER.setLevel(logging.DEBUG)  # capture everything for tailing
    root.addHandler(__RING_HANDLER)

    # Quiet noisy libs
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.INFO)


# ------------ Application ------------
class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self.runner = StrategyRunner()
        self.tg: Optional[TelegramController] = None
        self._start_ts = time.time()
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- status payload / health
    def _status_payload(self) -> dict:
        r = {}
        try:
            r = self.runner.to_status_dict()
        except Exception:
            r = {}
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(getattr(self.runner, "_live", False)),
            "paused": bool(getattr(self.runner, "_paused", False)),
            "active_orders": (len(self.runner.executor.get_active_orders()) if self.runner.executor else 0),
            "broker": "Kite" if getattr(self.runner, "_kite", None) else "none",
            "data_source": r.get("data_source"),
        }

    # ---- telegram command handler (fallback)
    def _handle_telegram_cmd(self, cmd: str) -> str:
        c = (cmd or "").strip().lower()
        if c == "ping":
            uptime = int(time.time() - self._start_ts)
            return f"Pong at {_now_ist_naive().strftime('%H:%M:%S')} (uptime {uptime}s)"
        if c == "status":
            return str(self._status_payload())
        if c == "stop":
            self._stop_event.set()
            return "Stopping…"
        return f"Unknown command: '{cmd}'"

    # ---- providers for TelegramController ----
    def _provider_positions(self) -> dict:
        if self.runner.executor:
            return self.runner.executor.get_positions_kite()
        return {}

    def _provider_actives(self):
        if self.runner.executor:
            return self.runner.executor.get_active_orders()
        return []

    def _provider_logs(self, n: int) -> str:
        return get_recent_logs(n)

    def _provider_health(self) -> dict:
        # combine health + runner status
        s = self._status_payload()
        s["uptime_sec"] = int(time.time() - self._start_ts)
        s["status"] = "ok"
        return s

    def _tick_once(self) -> dict:
        """Run one strategy cycle (on-demand). Returns a small summary."""
        try:
            res = self.runner.run_once(stop_event=self._stop_event)
            if res:
                return {"ran": True, "signal": True, "side": res.get("side"), "lots": res.get("lots"), "qty": res.get("quantity_units")}
            return {"ran": True, "signal": False}
        except Exception as e:
            log.exception("tick error: %s", e)
            return {"ran": False, "error": str(e)}

    # ---- run ----
    def run(self) -> None:
        """Main entry point for the application."""
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server
        health_thread = threading.Thread(
            target=health_server.run,
            kwargs={
                "callback": self._provider_health,
                "host": settings.server.host,
                "port": settings.server.port,
            },
            daemon=True,
        )
        health_thread.start()

        # Telegram controller
        try:
            tg_enabled = bool(getattr(settings.telegram, "enabled", True))
            bot_token = getattr(settings.telegram, "bot_token", None)
            chat_id = getattr(settings.telegram, "chat_id", None)
            if tg_enabled and bot_token and chat_id:
                self.tg = TelegramController(
                    status_provider=self._status_payload,
                    positions_provider=self._provider_positions,
                    actives_provider=self._provider_actives,
                    # control hooks
                    runner_pause=self.runner.pause,
                    runner_resume=self.runner.resume,
                    cancel_all=(self.runner.executor.cancel_all_orders if self.runner.executor else None),
                    # diagnostics
                    logs_provider=self._provider_logs,
                    health_provider=self._provider_health,
                    tick_provider=self._tick_once,
                    # live mode toggle
                    set_live_mode=lambda v: setattr(self.runner, "_live", bool(v)),
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main loop
        cadence = 0.75  # responsive sleep, no noisy heartbeats
        while not self._stop_event.is_set():
            try:
                # Only place orders during trading hours; the runner contains all gating.
                self.runner.run_once(stop_event=self._stop_event)
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
        log.info("Bot stopped.")


# ------------ cli ------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if (args.cmd or "start") == "start":
        app = Application()
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())