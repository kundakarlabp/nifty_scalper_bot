# src/main.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (guarded)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks


log = logging.getLogger(__name__)


# ---------------- Logging ----------------
def _setup_logging() -> None:
    level = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # keep 3rd party libs quiet
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------- Time helper ----------------
def _now_ist_str() -> str:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


class Application:
    """
    Main app: builds StrategyRunner, hooks Telegram providers & setters,
    starts the health server, and runs the trading loop.
    """

    def __init__(self) -> None:
        _setup_logging()

        # Build runner with an event sink (for Telegram notifications)
        self.runner = StrategyRunner(event_sink=self._on_event)
        # Track uptime for /status
        self._start_ts = time.time()
        # Graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)

        # Telegram (optional, enabled via settings.telegram)
        self.tg: Optional[TelegramController] = None

    # -------- signal handling --------
    def _sig(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s: shutting down.", signum)
        self._stop_event.set()

    # -------- Telegram event bridge --------
    def _on_event(self, evt: Dict[str, Any]) -> None:
        """Forward runner events to Telegram notifications."""
        if not self.tg or not isinstance(evt, dict):
            return
        typ = str(evt.get("type", "")).upper()
        if typ == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=str(evt.get("symbol", "?")),
                side=str(evt.get("side", "?")),
                qty=int(evt.get("qty", 0)),
                price=float(evt.get("price", 0.0)),
                record_id=str(evt.get("record_id", "")),
            )
        elif typ == "FILLS":
            fills = evt.get("fills") or []
            if isinstance(fills, list) and fills:
                self.tg.notify_fills(fills)

    # -------- Telegram providers / controls --------
    def _status_provider(self) -> Dict[str, Any]:
        status = self.runner.to_status_dict()
        status.setdefault("time_ist", _now_ist_str())
        status.setdefault("uptime_sec", int(time.time() - self._start_ts))
        return status

    def _positions_provider(self) -> Dict[str, Any]:
        if getattr(self.runner, "executor", None):
            try:
                return self.runner.executor.get_positions_kite()  # type: ignore[attr-defined]
            except Exception:
                return {}
        return {}

    def _actives_provider(self) -> List[Any]:
        if getattr(self.runner, "executor", None):
            try:
                return self.runner.executor.get_active_orders()  # type: ignore[attr-defined]
            except Exception:
                return []
        return []

    def _runner_pause(self) -> None:
        self.runner.pause()

    def _runner_resume(self) -> None:
        self.runner.resume()

    def _cancel_all(self) -> None:
        if getattr(self.runner, "executor", None):
            try:
                self.runner.executor.cancel_all_orders()  # type: ignore[attr-defined]
            except Exception as e:
                log.warning("cancel_all_orders failed: %s", e)

    # Runtime mutators exposed to TelegramController
    def _set_risk_pct(self, pct: float) -> None:
        # pct is like 0.5 → 0.5%; convert to fraction
        settings.risk.risk_per_trade = float(pct) / 100.0

    def _toggle_trailing(self, val: bool) -> None:
        settings.executor.enable_trailing = bool(val)

    def _set_trailing_mult(self, v: float) -> None:
        settings.executor.trailing_atr_multiplier = float(v)

    def _toggle_partial(self, val: bool) -> None:
        settings.executor.partial_tp_enable = bool(val)

    def _set_tp1_ratio(self, pct: float) -> None:
        # pct like 40 → 0.40
        settings.executor.tp1_qty_ratio = max(0.0, min(1.0, float(pct) / 100.0))

    def _set_breakeven_ticks(self, ticks: int) -> None:
        settings.executor.breakeven_ticks = int(ticks)

    def _set_live_mode(self, live: bool) -> None:
        settings.enable_live_trading = bool(live)
        # Keep runner in sync with latest toggle
        try:
            # StrategyRunner exposes internal flag; we sync it
            setattr(self.runner, "_live", bool(live))
        except Exception:
            pass
        log.info("Live mode set to %s.", "True" if live else "False")

    # -------- boot helpers --------
    def _start_health_server(self) -> None:
        t = threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_provider, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        )
        t.start()

    def _start_telegram(self) -> None:
        tg_cfg = settings.telegram
        # Both token and chat_id must be present
        if not tg_cfg or not tg_cfg.enabled or not tg_cfg.bot_token or not tg_cfg.chat_id:
            log.info("Telegram disabled or creds missing; skipping TelegramController.")
            return
        # Build controller with providers + setters
        self.tg = TelegramController(
            status_provider=self._status_provider,
            positions_provider=self._positions_provider,
            actives_provider=self._actives_provider,
            runner_pause=self._runner_pause,
            runner_resume=self._runner_resume,
            cancel_all=self._cancel_all,
            set_risk_pct=self._set_risk_pct,
            toggle_trailing=self._toggle_trailing,
            set_trailing_mult=self._set_trailing_mult,
            toggle_partial=self._toggle_partial,
            set_tp1_ratio=self._set_tp1_ratio,
            set_breakeven_ticks=self._set_breakeven_ticks,
            set_live_mode=self._set_live_mode,
        )
        self.tg.start_polling()
        # Friendly hello
        try:
            self.tg.send_startup_alert()
        except Exception:
            pass

    # -------- main loop --------
    def run(self) -> None:
        log.info("Starting Nifty Scalper Bot | live_trading=%s", settings.enable_live_trading)

        self._start_health_server()
        self._start_telegram()

        # Tight loop for quick responsiveness; we don't spam logs here.
        cadence = 0.6
        while not self._stop_event.is_set():
            try:
                self.runner.run_once(self._stop_event)
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


# ---------------- CLI ----------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        app = Application()
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())