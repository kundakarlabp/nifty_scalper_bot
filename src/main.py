# src/main.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.config import settings
from src.server import health as health_server
from src.notifications.telegram_controller import TelegramController
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.strategies.runner import StrategyRunner

# Optional broker SDK (keep imports safe)
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


# ------------ logging ------------
def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------ time helper ------------
def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ------------ kite session wiring ------------
def _build_kite() -> Optional[KiteConnect]:
    if KiteConnect is None:
        log.info("kiteconnect not installed; running in shadow mode.")
        return None

    ak = settings.zerodha.api_key
    sk = settings.zerodha.api_secret
    at = settings.zerodha.access_token
    enctoken = settings.zerodha.enctoken

    if enctoken:
        # ENCTOKEN bootstrap (no api_key/secret needed)
        k = KiteConnect(api_key="enc")  # dummy; library requires a string
        try:
            k.set_session_expiry_hook(lambda *a, **kw: None)  # avoid crashes on expiry hook
        except Exception:
            pass
        k.set_headers({"Authorization": f"enctoken {enctoken}"})
        log.info("Kite session initialized via ENCTOKEN.")
        return k

    if not ak or not at:
        log.info("Kite credentials missing (API key/access token). Shadow mode.")
        return None

    k = KiteConnect(api_key=ak)
    try:
        k.set_access_token(at)
        log.info("Kite session initialized via API key + access token.")
    except Exception as e:
        log.warning("Failed to set access token: %s", e)
        return None
    return k


def _build_sources(kite: Optional[KiteConnect]) -> tuple[Optional[DataSource], Optional[DataSource]]:
    """Return (option_source, spot_source). For now both are LiveKiteSource if kite is present."""
    if kite and LiveKiteSource:
        try:
            opt_src = LiveKiteSource(kite)
            opt_src.connect()
            spot_src = LiveKiteSource(kite)  # separate instance (simple)
            spot_src.connect()
            return opt_src, spot_src
        except Exception as e:
            log.warning("LiveKiteSource init/connect failed: %s", e)
    return None, None


class Application:
    """Main application class for the Nifty Scalper Bot."""

    def __init__(self) -> None:
        self.live_trading = bool(settings.enable_live_trading)
        self._stop_event = threading.Event()
        self._start_ts = time.time()

        # OS signals
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Build broker + data sources
        self.kite = _build_kite()
        opt_src, spot_src = _build_sources(self.kite)

        # Strategy + Runner wiring (runner expects explicit components)
        self.strategy = EnhancedScalpingStrategy()
        self.runner = StrategyRunner(
            strategy=self.strategy,
            data_source=opt_src,     # may be None in shadow mode; runner should handle
            spot_source=spot_src,    # may be None in shadow mode; runner should handle
        )

        self.tg: Optional[TelegramController] = None

    # ---- signals / shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:
        log.info("Received signal %s; initiating graceful shutdown…", signum)
        self._stop_event.set()

    # ---- telegram command handler
    def _handle_telegram_cmd(self, cmd: str) -> str:
        c = (cmd or "").strip().lower()
        if c == "ping":
            return f"Pong at {_now_ist_naive().strftime('%H:%M:%S')} (uptime: {int(time.time()-self._start_ts)}s)"
        if c == "status":
            return str(self._status_payload())
        if c == "stop":
            self._stop_event.set()
            return "Stopping…"
        return f"Unknown command: '{cmd}'"

    # ---- status payload for health/Telegram
    def _status_payload(self) -> dict:
        d = {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": self.live_trading,
            "runner": "StrategyRunner",
        }
        try:
            # If your runner exposes a health_check() dict, prefer that
            if hasattr(self.runner, "health_check"):
                d.update(self.runner.health_check())  # type: ignore[attr-defined]
        except Exception:
            pass
        return d

    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Health server
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

        # Telegram controller (guarded by settings.telegram)
        try:
            if settings.telegram.enabled and settings.telegram.bot_token and settings.telegram.chat_id:
                self.tg = TelegramController(
                    status_callback=self._status_payload,
                    control_callback=self._handle_telegram_cmd,
                    summary_callback=lambda: "No summary yet.",
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main trading loop (cooperative sleep + stop_event)
        cadence = 0.5
        while not self._stop_event.is_set():
            try:
                result = self.runner.run_once(stop_event=self._stop_event)
                if result:
                    log.info("Signal: %s", result)
                    # If you later wire OrderExecutor + Telegram fills, emit here
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