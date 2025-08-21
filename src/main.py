from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, List

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

log = logging.getLogger(__name__)

# ---- ring buffer for /logs
_LOG_RING: deque[str] = deque(maxlen=400)


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        _LOG_RING.append(msg)


def _setup_logging() -> None:
    """Configure centralized logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers[:] = [sh]

    rh = _RingHandler()
    rh.setFormatter(fmt)
    root.addHandler(rh)

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


def _get_recent_logs(n: int = 60) -> List[str]:
    n = max(1, min(200, n))
    return list(_LOG_RING)[-n:]


class Application:
    """Main application for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self.runner = StrategyRunner(event_sink=self._on_event)
        self.tg: Optional[TelegramController] = None

        self._start_ts = time.time()
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # heartbeat cadence for logs
        self._last_hb = 0.0
        self._hb_interval = 60.0  # 1 minute to keep deploy logs clean

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- events -> Telegram
    def _on_event(self, evt: dict) -> None:
        if not self.tg:
            return
        t = str(evt.get("type", "").upper())
        if t == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=evt.get("symbol"),
                side=evt.get("side"),
                qty=int(evt.get("qty", 0)),
                price=float(evt.get("price", 0.0)),
                record_id=str(evt.get("record_id") or evt.get("order_record_id", "")),
            )
        elif t == "FILLS":
            fills = evt.get("fills") or []
            try:
                self.tg.notify_fills([(str(rid), float(px)) for rid, px in fills])
            except Exception:
                pass

    # ---- status payload
    def _status_payload(self) -> dict:
        rstat = {}
        try:
            rstat = self.runner.to_status_dict()
        except Exception:
            rstat = {}
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(self.live_trading),
            "runner": rstat or {},
        }

    # ---- telegram
    def _setup_telegram(self) -> None:
        try:
            tg_enabled = bool(getattr(settings.telegram, "enabled", True))
            bot_token = getattr(settings.telegram, "bot_token", None)
            chat_id = getattr(settings.telegram, "chat_id", None)
            if not (tg_enabled and bot_token and chat_id):
                log.info("Telegram disabled or credentials missing.")
                return

            # Providers
            def positions_provider():
                ex = getattr(self.runner, "executor", None)
                return ex.get_positions_kite() if ex else {}

            def actives_provider():
                ex = getattr(self.runner, "executor", None)
                return ex.get_active_orders() if ex else []

            # Diagnostic provider (runner can expose one; otherwise no-op)
            diag_provider = getattr(self.runner, "diagnose_flow", None)

            self.tg = TelegramController(
                status_provider=lambda: self._status_payload().get("runner", {}),
                positions_provider=positions_provider,
                actives_provider=actives_provider,
                runner_pause=self.runner.pause,
                runner_resume=self.runner.resume,
                cancel_all=(getattr(self.runner, "executor", None).cancel_all_orders
                            if getattr(self.runner, "executor", None) else None),
                set_risk_pct=lambda pct: setattr(settings.risk, "risk_per_trade", float(pct) / 100.0),
                toggle_trailing=lambda on: setattr(settings.executor, "enable_trailing", bool(on)),
                set_trailing_mult=lambda x: setattr(settings.executor, "trailing_atr_multiplier", float(x)),
                toggle_partial=lambda on: setattr(settings.executor, "partial_tp_enable", bool(on)),
                set_tp1_ratio=lambda pct: setattr(settings.executor, "tp1_qty_ratio", float(pct) / 100.0),
                set_breakeven_ticks=lambda n: setattr(settings.executor, "breakeven_ticks", int(n)),
                set_live_mode=self._set_live_mode,
                diag_provider=diag_provider,
                log_provider=_get_recent_logs,
                http_timeout=20.0,
            )
            self.tg.start_polling()
            self.tg.send_startup_alert()
            log.info("Telegram polling started.")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

    def _set_live_mode(self, val: bool) -> None:
        self.live_trading = bool(val)
        setattr(settings, "enable_live_trading", self.live_trading)
        state = "True" if self.live_trading else "False"
        log.info("Live mode set to %s.", state)

    # ---- main loop ----
    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server
        threading.Thread(
            target=health_server.run,
            kwargs={
                "callback": self._status_payload,
                "host": settings.server.host,
                "port": settings.server.port,
            },
            daemon=True,
        ).start()

        # Telegram
        self._setup_telegram()

        cadence = 0.5
        while not self._stop_event.is_set():
            try:
                result = self.runner.run_once(stop_event=self._stop_event)
                if result:
                    log.info("Signal: %s", result)

                # lightweight heartbeat to deploy logs
                now = time.time()
                if now - self._last_hb >= 60.0:
                    src = type(self.runner.data_source).__name__ if self.runner.data_source else None
                    active = 0
                    ex = getattr(self.runner, "executor", None)
                    if ex:
                        try:
                            active = len(ex.get_active_orders())
                        except Exception:
                            active = 0
                    log.info("⏱ heartbeat | live=%d paused=%s active=%d src=%s",
                             1 if self.live_trading else 0, self.runner._paused, active, src)
                    self._last_hb = now

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