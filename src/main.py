from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict, List

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (safe imports)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks


log = logging.getLogger(__name__)


# ------------ Logging helper ------------
def _setup_logging() -> None:
    """Configure centralized logging."""
    lvl = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------ time helper ------------
def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        # Runner builds its own Kite/DataSource if available
        self.runner = StrategyRunner(event_sink=self._on_event)
        self.tg: Optional[TelegramController] = None

        # uptime
        self._start_ts = time.time()

        # graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- health/status payload
    def _status_payload(self) -> Dict[str, Any]:
        active_count = 0
        if self.runner and getattr(self.runner, "executor", None):
            try:
                active_count = len(self.runner.executor.get_active_orders())
            except Exception:
                active_count = 0
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(self.runner._live),
            "broker": "Kite" if self.runner._kite else "none",
            "active_orders": active_count,
            "uptime_sec": int(time.time() - self._start_ts),
        }

    # ---- providers used by Telegram
    def _positions_provider(self) -> Dict[str, Any]:
        if self.runner and self.runner.executor:
            try:
                return self.runner.executor.get_positions_kite()
            except Exception:
                return {}
        return {}

    def _actives_provider(self) -> List[Any]:
        if self.runner and self.runner.executor:
            try:
                return self.runner.executor.get_active_orders()
            except Exception:
                return []
        return []

    # ---- control hooks exposed to Telegram
    def set_live_mode(self, enable: bool) -> None:
        """Flip runner into LIVE/DRY (runtime state)."""
        try:
            self.runner._live = bool(enable)
            log.info("Live mode set to %s", self.runner._live)
        except Exception as e:
            log.warning("Failed to set live mode: %s", e)

    def _cancel_all(self) -> None:
        if self.runner and self.runner.executor:
            try:
                self.runner.executor.cancel_all_orders()
            except Exception as e:
                log.warning("cancel_all_orders failed: %s", e)

    # ---- event sink from runner -> Telegram
    def _on_event(self, evt: Dict[str, Any]) -> None:
        if not self.tg or not isinstance(evt, dict):
            return
        et = str(evt.get("type") or "").upper()
        if et == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=str(evt.get("symbol", "?")),
                side=str(evt.get("side", "?")),
                qty=int(evt.get("qty", 0) or 0),
                price=float(evt.get("price", 0.0) or 0.0),
                record_id=str(evt.get("record_id", "")),
            )
        elif et == "FILLS":
            fills = evt.get("fills") or []
            if isinstance(fills, list) and fills:
                self.tg.notify_fills(fills)

    def _start_health_server(self) -> None:
        t = threading.Thread(
            target=health_server.run,
            kwargs={
                "callback": self._status_payload,
                "host": settings.server.host,
                "port": settings.server.port,
            },
            daemon=True,
            name="health-server",
        )
        t.start()

    def _start_telegram(self) -> None:
        tg_enabled = bool(getattr(settings.telegram, "enabled", True))
        bot_token = getattr(settings.telegram, "bot_token", None)
        chat_id = getattr(settings.telegram, "chat_id", None)
        if not (tg_enabled and bot_token and chat_id):
            log.info("Telegram not started (disabled or credentials missing).")
            return

        self.tg = TelegramController(
            # status & data
            status_provider=self._status_payload,
            positions_provider=self._positions_provider,
            actives_provider=self._actives_provider,
            # runner controls
            runner_pause=self.runner.pause,
            runner_resume=self.runner.resume,
            cancel_all=self._cancel_all,
            # runtime config hooks
            set_live_mode=self.set_live_mode,
        )
        self.tg.start_polling()
        self.tg.send_startup_alert()

    def run(self) -> None:
        """Main entry point for the application."""
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.runner._live)

        # Health server for probes
        self._start_health_server()

        # Telegram (if configured)
        try:
            self._start_telegram()
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main trading loop
        cadence = 0.75
        while not self._stop_event.is_set():
            try:
                _ = self.runner.run_once(stop_event=self._stop_event)
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


# ------------ CLI ------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest (not implemented)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        app = Application()
        app.run()
    elif cmd == "backtest":
        log.info("Backtest not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())