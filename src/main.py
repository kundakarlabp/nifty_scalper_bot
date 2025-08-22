# src/main.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # fallback if kiteconnect not installed
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception


# ---------------- Logging ----------------
class RingBufferLogHandler(logging.Handler):
    """In-memory log buffer for /logs."""
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self.buf: Deque[str] = deque(maxlen=self.capacity)
        self._fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append(self._fmt.format(record))
        except Exception:
            pass

    def tail(self, n: int) -> List[str]:
        return list(self.buf)[-n:]


def _setup_logging() -> RingBufferLogHandler:
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(ch)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    rb = RingBufferLogHandler(capacity=2000)
    root.addHandler(rb)
    return rb


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ---------------- Application ----------------
class Application:
    def __init__(self) -> None:
        self._log_handler = _setup_logging()
        self.log = logging.getLogger(__name__)

        self.live_trading = bool(settings.enable_live_trading)
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._start_ts = time.time()
        self._last_signal: Optional[Dict[str, Any]] = None
        self.runner = StrategyRunner(event_sink=self._on_event)
        self.tg: Optional[TelegramController] = None

    # ---------- status payload ----------
    def _status_payload(self) -> Dict[str, Any]:
        try:
            r = self.runner.to_status_dict()
        except Exception:
            r = {}
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(self.live_trading),
            "paused": bool(r.get("paused")),
            "broker": r.get("broker") or ("Kite" if KiteConnect else "none"),
            "data_source": r.get("data_source"),
            "active_orders": r.get("active_orders", 0),
            "uptime_sec": int(time.time() - self._start_ts),
        }

    # ---------- health server ----------
    def _start_health(self) -> None:
        # Defaults if not in .env
        host = "0.0.0.0"
        port = 8000
        t = threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": host, "port": port},
            daemon=True,
            name="health-http",
        )
        t.start()

    # ---------- telegram wiring ----------
    def _wire_telegram(self) -> None:
        tg_enabled = True
        if not getattr(settings, "telegram_bot_token", None) or not getattr(settings, "telegram_chat_id", None):
            self.log.info("Telegram not started (disabled or credentials missing).")
            return

        self.tg = TelegramController(
            status_provider=self._status_payload,
            positions_provider=lambda: getattr(self.runner.executor, "get_positions_kite", lambda: {})(),
            actives_provider=lambda: getattr(self.runner.executor, "get_active_orders", lambda: [])(),
            diag_provider=lambda: self.runner.diagnose(),
            logs_provider=self._log_handler.tail,
            last_signal_provider=lambda: self._last_signal,
        )
        self.tg.start_polling()
        self.tg.send_startup_alert()

    # ---------- runner events ----------
    def _on_event(self, evt: Dict[str, Any]) -> None:
        if not evt:
            return
        try:
            et = evt.get("type")
            if et == "ENTRY_PLACED" and self.tg:
                self.tg.notify_entry(
                    symbol=str(evt.get("symbol")),
                    side=str(evt.get("side")),
                    qty=int(evt.get("qty", 0)),
                    price=float(evt.get("price", 0.0)),
                    record_id=str(evt.get("record_id")),
                )
            elif et in ("FILL", "FILLS") and self.tg:
                fills = evt.get("fills") or []
                if fills:
                    self.tg.notify_fills(fills)

            if et in ("ENTRY_PLACED", "SIGNAL", "FILL", "FILLS"):
                self._last_signal = evt
        except Exception:
            pass

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.log.info("Signal %s received, shutting down…", signum)
        self._stop_event.set()

    # ---------- main loop ----------
    def run(self) -> None:
        self.log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)
        self._start_health()
        self._wire_telegram()

        cadence = 0.5
        hb_every = 300.0
        last_hb = 0.0

        while not self._stop_event.is_set():
            try:
                res = self.runner.run_once(self._stop_event)
                if res:
                    self._last_signal = res
            except (NetworkException, TokenException, InputException) as e:
                logging.getLogger(__name__).error("Transient broker error: %s", e)
            except Exception as e:
                logging.getLogger(__name__).exception("Main loop error: %s", e)

            now = time.time()
            if now - last_hb > hb_every:
                s = self._status_payload()
                self.log.info("⏱ heartbeat | live=%d paused=%s active=%d src=%s",
                              1 if s.get("live_trading") else 0,
                              s.get("paused"),
                              s.get("active_orders", 0),
                              s.get("data_source"))
                last_hb = now

            if self._stop_event.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        self.log.info("Nifty Scalper Bot stopped.")


# ---------------- CLI ----------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest from CSV file")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"

    if cmd == "start":
        app = Application()
        app.run()
    elif cmd == "backtest":
        logging.getLogger(__name__).info("Backtest command not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())