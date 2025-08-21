# src/main.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict

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


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._start_ts = time.time()
        self.live_trading = bool(settings.enable_live_trading)

        # Runner with event sink wired to Telegram (installed later)
        self.tg: Optional[TelegramController] = None
        self.runner = StrategyRunner(event_sink=self._on_runner_event)

    # ---------------- lifecycle ----------------
    def _on_signal(self, signum: int, _frame: Any) -> None:
        log.info("Received signal %s, shutting down…", signum)
        self._stop_event.set()

    # ---------------- event bridge from runner -> Telegram ----------------
    def _on_runner_event(self, event: Dict[str, Any]) -> None:
        etype = (event or {}).get("type")
        if not self.tg:
            return
        if etype == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=event.get("symbol", "?"),
                side=event.get("side", "?"),
                qty=int(event.get("qty", 0) or 0),
                price=float(event.get("price", 0.0) or 0.0),
                record_id=str(event.get("record_id", "?")),
            )
        elif etype == "FILLS":
            self.tg.notify_fills(event.get("fills", []) or [])

    # ---------------- providers/hooks for Telegram ----------------
    def _status_payload(self) -> Dict[str, Any]:
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": self.runner.is_live(),
            "broker": "Kite" if self.runner.has_broker() else "none",
            "active_orders": len(self.runner.get_active_orders() or []),
            "paused": self.runner.is_paused(),
        }

    def _positions_provider(self) -> Dict[str, Any]:
        return self.runner.get_positions() or {}

    def _actives_provider(self):
        return self.runner.get_active_orders() or []

    def _pause(self) -> None:
        self.runner.pause()

    def _resume(self) -> None:
        self.runner.resume()

    def _cancel_all(self) -> None:
        self.runner.cancel_all()

    def _set_risk_pct(self, pct: float) -> None:
        # Settings are live objects; runner reads them per-tick.
        settings.risk.risk_per_trade = float(pct) / 100.0

    def _toggle_trailing(self, on: bool) -> None:
        settings.executor.enable_trailing = bool(on)

    def _set_trailing_mult(self, v: float) -> None:
        settings.executor.trailing_atr_multiplier = float(v)

    def _toggle_partial(self, on: bool) -> None:
        settings.executor.partial_tp_enable = bool(on)

    def _set_tp1_ratio(self, pct: float) -> None:
        settings.executor.tp1_qty_ratio = float(pct) / 100.0

    def _set_breakeven_ticks(self, ticks: int) -> None:
        settings.executor.breakeven_ticks = int(ticks)

    def _set_live_mode(self, on: bool) -> None:
        settings.enable_live_trading = bool(on)
        self.runner.set_live(bool(on))
        log.info("Live mode set to %s.", bool(on))

    # ---------------- main ----------------
    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

        # Telegram (if credentials exist)
        try:
            tg_cfg = settings.telegram
            if tg_cfg.enabled and tg_cfg.bot_token and tg_cfg.chat_id:
                self.tg = TelegramController(
                    status_provider=self._status_payload,
                    positions_provider=self._positions_provider,
                    actives_provider=self._actives_provider,
                    runner_pause=self._pause,
                    runner_resume=self._resume,
                    cancel_all=self._cancel_all,
                    set_risk_pct=self._set_risk_pct,
                    toggle_trailing=self._toggle_trailing,
                    set_trailing_mult=self._set_trailing_mult,
                    toggle_partial=self._toggle_partial,
                    set_tp1_ratio=self._set_tp1_ratio,
                    set_breakeven_ticks=self._set_breakeven_ticks,
                    set_live_mode=self._set_live_mode,  # <- critical: flips runner live immediately
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main loop
        cadence = 1.0
        while not self._stop_event.is_set():
            try:
                self.runner.run_once(stop_event=self._stop_event)
            except (NetworkException, TokenException, InputException) as e:
                log.error("Transient broker error: %s", e)
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
        Application().run()
    else:
        log.info("Backtest not implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())