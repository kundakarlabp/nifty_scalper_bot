from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (keep imports safe)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

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


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        # Wire the runner with an event sink that forwards to Telegram
        self.runner = StrategyRunner(event_sink=self._on_runner_event)
        self.tg: Optional[TelegramController] = None

        # for uptime in /status
        self._start_ts = time.time()

        # graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, _frame: Any) -> None:
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- runner -> telegram event bridge
    def _on_runner_event(self, evt: dict) -> None:
        """Forward StrategyRunner events to Telegram."""
        if not self.tg or not isinstance(evt, dict):
            return
        et = (evt.get("type") or "").upper()
        if et == "ENTRY_PLACED":
            try:
                self.tg.notify_entry(
                    symbol=str(evt.get("symbol")),
                    side=str(evt.get("side")),
                    qty=int(evt.get("qty")),
                    price=float(evt.get("price")),
                    record_id=str(evt.get("record_id")),
                )
            except Exception:
                pass
        elif et == "FILLS":
            fills = evt.get("fills") or []
            try:
                self.tg.notify_fills(fills)
            except Exception:
                pass

    # ---- status payload for health/Telegram
    def _status_payload(self) -> dict:
        active = []
        try:
            if self.runner.executor:
                active = self.runner.executor.get_active_orders()
        except Exception:
            active = []
        return {
            "time_ist": time.strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite" if isinstance(getattr(self.runner, "_kite", None), KiteConnect) else "none",
            "active_orders": len(active),
        }

    # ---- providers/hooks for Telegram ----
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

    def _runner_pause(self) -> None:
        self.runner.pause()

    def _runner_resume(self) -> None:
        self.runner.resume()

    def _cancel_all(self) -> None:
        if self.runner.executor:
            self.runner.executor.cancel_all_orders()

    def _set_risk_pct(self, pct: float) -> None:
        # pct is like 0.5 => 0.5%
        settings.risk.risk_per_trade = float(pct) / 100.0  # runtime mutation ok

    def _toggle_trailing(self, val: bool) -> None:
        settings.executor.enable_trailing = bool(val)

    def _set_trailing_mult(self, v: float) -> None:
        settings.executor.trailing_atr_multiplier = float(v)

    def _toggle_partial(self, val: bool) -> None:
        settings.executor.partial_tp_enable = bool(val)

    def _set_tp1_ratio(self, pct: float) -> None:
        settings.executor.tp1_qty_ratio = float(pct) / 100.0

    def _set_breakeven_ticks(self, ticks: int) -> None:
        settings.executor.breakeven_ticks = int(ticks)

    def _set_live_mode(self, val: bool) -> None:
        settings.enable_live_trading = bool(val)
        self.live_trading = bool(val)
        log.info("Live mode set to %s.", "True" if val else "False")

    # ---- Telegram bootstrap ----
    def _start_telegram(self) -> None:
        tg_conf = getattr(settings, "telegram", object())
        bot_token = getattr(tg_conf, "bot_token", None)
        chat_id = getattr(tg_conf, "chat_id", None)
        enabled = bool(getattr(tg_conf, "enabled", True))
        if not (enabled and bot_token and chat_id):
            log.info("Telegram not started (disabled or credentials missing).")
            return

        self.tg = TelegramController(
            status_provider=self._status_payload,
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
        self.tg.send_startup_alert()

    def run(self) -> None:
        """Main entry point for the application."""
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server (threaded)
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

        # Telegram (if enabled)
        try:
            self._start_telegram()
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main trading loop
        cadence = 0.5  # seconds
        while not self._stop_event.is_set():
            try:
                _ = self.runner.run_once(stop_event=self._stop_event)
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
    sub.add_parser("backtest", help="Run backtest from CSV file (todo)")
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