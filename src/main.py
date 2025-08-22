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

# Optional broker SDK (safe import)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

# Optional DataSource
try:
    from src.data.source import LiveKiteSource, DataSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore
    DataSource = object  # type: ignore


log = logging.getLogger(__name__)


# --------- In-memory log buffer for /logs ---------
class _RingLogHandler(logging.Handler):
    def __init__(self, capacity: int = 400):
        super().__init__()
        self.capacity = capacity
        self.buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append({
                "ts": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "lvl": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            })
        except Exception:
            pass

    def last(self, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        n = min(n, self.capacity)
        return list(self.buf)[-n:]


_ring_handler = _RingLogHandler(capacity=600)


def _setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    # attach ring buffer once
    if _ring_handler not in root.handlers:
        root.addHandler(_ring_handler)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


class Application:
    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self.runner = StrategyRunner(event_sink=self._on_event)
        self.tg: Optional[TelegramController] = None

        self._start_ts = time.time()
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ----- signals -----
    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s, shutting down…", signum)
        self._stop_event.set()

    # ----- events from runner -----
    def _on_event(self, evt: dict) -> None:
        et = evt.get("type")
        if not self.tg:
            return
        if et == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=evt.get("symbol", "?"),
                side=evt.get("side", "?"),
                qty=int(evt.get("qty", 0) or 0),
                price=float(evt.get("price", 0.0) or 0.0),
                record_id=str(evt.get("record_id", "")),
            )
        elif et == "FILLS":
            fills = evt.get("fills", [])
            if fills:
                self.tg.notify_fills(fills)

    # ----- telegram control shims (legacy/simple) -----
    def _control_cmd(self, cmd: str) -> str:
        c = (cmd or "").strip().lower()
        if c == "ping":
            return f"Pong {_now_ist_naive().strftime('%H:%M:%S')} uptime={int(time.time()-self._start_ts)}s"
        if c == "status":
            return str(self._status_payload())
        if c == "pause":
            self.runner.pause()
            return "Paused entries."
        if c == "resume":
            self.runner.resume()
            return "Resumed entries."
        if c == "live":
            settings.enable_live_trading = True
            self.live_trading = True
            return "Live mode enabled."
        if c == "dry":
            settings.enable_live_trading = False
            self.live_trading = False
            return "Dry mode enabled."
        if c == "stop":
            self._stop_event.set()
            return "Stopping…"
        return f"Unknown: {cmd}"

    # ----- telegram providers -----
    def _actives_provider(self) -> List[Any]:
        try:
            return self.runner.executor.get_active_orders() if self.runner.executor else []
        except Exception:
            return []

    def _positions_provider(self) -> Dict[str, Any]:
        try:
            return self.runner.executor.get_positions_kite() if self.runner.executor else {}
        except Exception:
            return {}

    def _logs_provider(self, n: int) -> List[Dict[str, Any]]:
        try:
            return _ring_handler.last(n)
        except Exception:
            return []

    def _diag_provider(self) -> Dict[str, Any]:
        # Prefer runner.diagnose if present
        diag_fn = getattr(self.runner, "diagnose", None)
        if callable(diag_fn):
            try:
                return diag_fn()
            except Exception as e:
                return {"ok": False, "error": f"runner.diagnose failed: {e}"}
        # Fallback minimal
        try:
            ok = bool(self.runner.data_source) and bool(self.runner.executor)
            return {
                "ok": ok,
                "checks": [
                    {"name": "broker", "ok": bool(getattr(self.runner, "_kite", None))},
                    {"name": "data_source", "ok": bool(self.runner.data_source)},
                    {"name": "executor", "ok": bool(self.runner.executor)},
                ],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tick_once(self) -> None:
        # Force one run_once; still respects market-hours gates inside
        try:
            self.runner.run_once(stop_event=self._stop_event)
        except Exception as e:
            log.error("Manual tick failed: %s", e)

    # ----- status payload -----
    def _status_payload(self) -> dict:
        active = 0
        try:
            active = len(self.runner.executor.get_active_orders()) if self.runner.executor else 0
        except Exception:
            pass
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite" if getattr(self.runner, "_kite", None) else "none",
            "data_source": type(self.runner.data_source).__name__ if self.runner.data_source else None,
            "active_orders": active,
            "paused": getattr(self.runner, "_paused", False),
        }

    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # health server
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

        # telegram
        try:
            tg_enabled = bool(getattr(settings.telegram, "enabled", True))
            if tg_enabled and settings.telegram.bot_token and settings.telegram.chat_id:
                self.tg = TelegramController(
                    status_callback=self._status_payload,
                    summary_callback=lambda: "No summary yet.",
                    control_callback=self._control_cmd,
                    actives_provider=self._actives_provider,
                    positions_provider=self._positions_provider,
                    logs_provider=self._logs_provider,
                    diag_provider=self._diag_provider,
                    runner_pause=self.runner.pause,
                    runner_resume=self.runner.resume,
                    cancel_all=(self.runner.executor.cancel_all_orders if self.runner.executor else None),
                    tick_callback=self._tick_once,
                    # runtime toggles (optional shims; defaults handled in controller)
                    set_risk_pct=lambda pct: setattr(settings.risk, "risk_per_trade", float(pct) / 100.0),
                    toggle_trailing=lambda v: setattr(settings.executor, "enable_trailing", bool(v)),
                    set_trailing_mult=lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)),
                    toggle_partial=lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)),
                    set_tp1_ratio=lambda pct: setattr(settings.executor, "tp1_qty_ratio", float(pct) / 100.0),
                    set_breakeven_ticks=lambda t: setattr(settings.executor, "breakeven_ticks", int(t)),
                    set_live_mode=lambda v: setattr(settings, "enable_live_trading", bool(v)),
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram disabled or not configured.")
        except Exception as e:
            log.warning("Telegram not started: %s", e)
            self.tg = None

        # main loop
        cadence = 0.5
        while not self._stop_event.is_set():
            try:
                self.runner.run_once(stop_event=self._stop_event)
            except (NetworkException, TokenException, InputException) as e:
                log.error("Broker error: %s", e)
            except Exception as e:
                log.exception("Main loop error: %s", e)

            if self._stop_event.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        log.info("Bot stopped.")


# ---- CLI ----
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest from CSV file (todo)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        app = Application()
        app.run()
    elif cmd == "backtest":
        log.info("Backtest not implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())