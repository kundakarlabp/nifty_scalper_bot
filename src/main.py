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
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (keep imports safe)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

# Optional DataSource (runner can build it)
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


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self.runner = StrategyRunner(event_sink=self._on_runner_event)
        self.tg: Optional[TelegramController] = None

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

    # ---- event sink (from StrategyRunner)
    def _on_runner_event(self, evt: dict) -> None:
        """Receive ENTRY_PLACED / FILLS from runner and forward to Telegram."""
        if not self.tg or not isinstance(evt, dict):
            return
        t = str(evt.get("type", "")).upper()
        if t == "ENTRY_PLACED":
            try:
                self.tg.notify_entry(
                    symbol=str(evt.get("symbol", "?")),
                    side=str(evt.get("side", "?")),
                    qty=int(evt.get("qty", 0)),
                    price=float(evt.get("price", 0.0)),
                    record_id=str(evt.get("record_id", "")),
                )
            except Exception:
                pass
        elif t == "FILLS":
            fills = evt.get("fills") or []
            try:
                self.tg.notify_fills(fills)
            except Exception:
                pass

    # ---- status payload for health/Telegram
    def _status_payload(self) -> dict:
        active_cnt = 0
        try:
            active_cnt = len(self.runner.executor.get_active_orders()) if self.runner.executor else 0
        except Exception:
            pass
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(getattr(settings, "enable_live_trading", False)),
            "broker": "Kite" if getattr(self.runner, "_kite", None) else "none",
            "active_orders": active_cnt,
        }

    # ---- provider wrappers for Telegram
    def _positions_provider(self) -> dict:
        try:
            return self.runner.executor.get_positions_kite() if self.runner.executor else {}
        except Exception:
            return {}

    def _actives_provider(self):
        try:
            return self.runner.executor.get_active_orders() if self.runner.executor else []
        except Exception:
            return []

    # ---- control hooks
    def _pause(self) -> None:
        self.runner.pause()

    def _resume(self) -> None:
        self.runner.resume()

    def _cancel_all(self) -> None:
        if self.runner.executor:
            self.runner.executor.cancel_all_orders()

    def _enable_live(self, v: bool) -> None:
        setattr(settings, "enable_live_trading", bool(v))
        self.live_trading = bool(v)

    def run(self) -> None:
        """Main entry point for the application."""
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

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

        # Telegram controller (enabled + creds required)
        try:
            # accept both TELEGRAM__ENABLED and legacy ENABLE_TELEGRAM
            tg_enabled_env = os.getenv("ENABLE_TELEGRAM", "").lower() in ("1", "true", "yes")
            tg_enabled_cfg = bool(getattr(getattr(settings, "telegram", object()), "enabled", True))
            tg_enabled = tg_enabled_cfg or tg_enabled_env

            bot_token = getattr(getattr(settings, "telegram", object()), "bot_token", None)
            chat_id = getattr(getattr(settings, "telegram", object()), "chat_id", None)

            if tg_enabled and bot_token and chat_id:
                self.tg = TelegramController(
                    status_provider=self._status_payload,
                    positions_provider=self._positions_provider,
                    actives_provider=self._actives_provider,
                    runner_pause=self._pause,
                    runner_resume=self._resume,
                    cancel_all=self._cancel_all,
                    set_risk_pct=lambda pct: setattr(settings.risk, "risk_per_trade", float(pct) / 100.0),
                    toggle_trailing=lambda v: setattr(settings.executor, "enable_trailing", bool(v)),
                    set_trailing_mult=lambda x: setattr(settings.executor, "trailing_atr_multiplier", float(x)),
                    toggle_partial=lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)),
                    set_tp1_ratio=lambda pct: setattr(settings.executor, "tp1_qty_ratio", float(pct) / 100.0),
                    set_breakeven_ticks=lambda n: setattr(settings.executor, "breakeven_ticks", int(n)),
                    set_live_mode=self._enable_live,
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