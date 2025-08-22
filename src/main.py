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

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# Optional DataSource (runner builds it if not provided)
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore


log = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class Application:
    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self.runner = StrategyRunner()
        self.tg: Optional[TelegramController] = None
        self._start_ts = time.time()
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- Telegram glue ----
    def _status_payload(self) -> dict:
        base = self.runner.to_status_dict()
        base["uptime_sec"] = int(time.time() - self._start_ts)
        return base

    def _actives_provider(self):
        return self.runner.executor.get_active_orders() if self.runner.executor else []

    def _positions_provider(self):
        return self.runner.executor.get_positions_kite() if self.runner.executor else {}

    def _event_sink(self, evt: dict) -> None:
        if not self.tg:
            return
        t = evt.get("type")
        if t == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=evt.get("symbol", "?"),
                side=evt.get("side", "?"),
                qty=int(evt.get("qty", 0) or 0),
                price=float(evt.get("price", 0.0) or 0.0),
                record_id=str(evt.get("record_id")),
            )
        elif t == "FILLS":
            self.tg.notify_fills(evt.get("fills", []))

    # ---- command hooks for Telegram ----
    def _runner_pause(self) -> None:
        self.runner.pause()

    def _runner_resume(self) -> None:
        self.runner.resume()

    def _cancel_all(self) -> None:
        if self.runner.executor:
            self.runner.executor.cancel_all_orders()

    def _set_risk_pct(self, pct: float) -> None:
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
        self.live_trading = bool(on)
        log.info("Live mode set to %s.", "True" if on else "False")

    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server
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

        # Wire event sink
        self.runner.set_event_sink(self._event_sink)

        # Telegram (if enabled)
        try:
            tg_enabled = bool(getattr(settings.telegram, "enabled", True))
            if tg_enabled and settings.telegram.bot_token and settings.telegram.chat_id:
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
                    diag_provider=self.runner.diag_snapshot,
                    tick_callback=self.runner.tick_once,
                    http_timeout=20.0,
                    attach_log_buffer=True,
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main loop — lean logging; rely on /status, /diag, /logs on demand
        cadence = 1.0
        while not self._stop_event.is_set():
            try:
                self.runner.run_once(stop_event=self._stop_event)
            except Exception as e:
                log.exception("Main loop error: %s", e)
            if self._stop_event.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        log.info("Nifty Scalper Bot stopped.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        app = Application()
        app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())