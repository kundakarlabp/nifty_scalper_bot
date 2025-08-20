from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Optional import of KiteConnect
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


def _build_kite_or_die() -> Optional[Any]:
    """
    Build a KiteConnect client if env/config present. Returns None if kiteconnect
    isn’t installed. If keys are present but login is bad, we fail fast so you notice.
    """
    if KiteConnect is None:
        log.warning("kiteconnect not installed; running without live feed.")
        return None

    api_key = settings.zerodha.api_key or getattr(settings, "ZERODHA_API_KEY", None)
    access_token = settings.zerodha.access_token or getattr(settings, "KITE_ACCESS_TOKEN", None)

    if not api_key or not access_token:
        log.warning("Kite credentials missing; running without live feed.")
        return None

    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # light sanity: get margins (cheap) to prove token is valid
        try:
            kite.margins()
        except Exception:
            # If margins scope isn’t allowed, try profile as fallback
            kite.profile()
        log.info("KiteConnect OK (token valid).")
        return kite
    except Exception as e:
        log.critical("Failed to initialize KiteConnect with provided creds: %s", e, exc_info=False)
        return None


def _load_data_source(kite: Optional[Any]) -> Optional[Any]:
    """
    Use concrete LiveKiteSource from src.data.source (requires connect()).
    We fail gracefully and log the exact reason.
    """
    try:
        from src.data.source import LiveKiteSource  # type: ignore
    except Exception as e:
        log.warning("LiveKiteSource import failed: %s", e)
        return None

    if kite is None:
        return None

    try:
        ds = LiveKiteSource(kite=kite)  # your class requires a KiteConnect object
        ds.connect()  # will set is_connected and raise on failure
        log.info("LiveKiteSource connected.")
        return ds
    except Exception as e:
        log.warning("Data source connect failed: %s", e)
        return None


class Application:
    """
    Orchestrator: health server + StrategyRunner loop + Telegram control.
    Works in shadow mode if live deps are absent.
    """

    def __init__(self) -> None:
        self.settings = settings
        self.live_trading = bool(self.settings.enable_live_trading)
        self.quality = "AUTO"
        self.regime = "auto"

        self._started_ts = time.time()
        self._shutdown = False

        # Dependencies
        self.kite = _build_kite_or_die()
        self.data_source = _load_data_source(self.kite)
        if self.data_source is None:
            log.warning("Data source not available. Running without live feed.")

        self.runner = StrategyRunner(data_source=self.data_source, kite=self.kite)

        # Health + Telegram
        self._health_thread: Optional[threading.Thread] = None
        self.tg: Optional[TelegramController] = None

    # ---- status payload (used by health + Telegram /status) ----
    def _status_payload(self) -> Dict[str, Any]:
        uptime = time.time() - self._started_ts
        return {
            "ok": True,
            "is_trading": bool(self.live_trading),
            "live_mode": bool(self.live_trading),
            "quality": self.quality,
            "regime": self.regime,
            "open_positions": 0,
            "closed_today": 0,
            "daily_pnl": 0.0,
            "session_date": _now_ist_naive().strftime("%Y-%m-%d"),
            "uptime_seconds": float(uptime),
            "kite": bool(self.kite is not None),
            "data_source": bool(self.data_source is not None),
            "preferred_exit_mode": self.settings.preferred_exit_mode,
        }

    # ---- Telegram control handler ----
    def _handle_telegram_cmd(self, command: str, arg: str) -> bool:
        try:
            if command == "mode":
                if arg == "live":
                    self.live_trading = True
                elif arg == "shadow":
                    self.live_trading = False
                else:
                    return False
                return True

            if command == "quality":
                v = arg.upper()
                if v not in ("AUTO", "CONSERVATIVE", "AGGRESSIVE"):
                    return False
                self.quality = v
                return True

            if command == "start":
                self.live_trading = True
                return True

            if command == "stop":
                self.live_trading = False
                return True

            if command == "refresh":
                # hook for cache refresh / re-login, if required
                return True

            if command == "health":
                return True

            if command == "emergency":
                # TODO: cancel open orders/flatten positions if you wire trading layer
                self.live_trading = False
                return True

            return False
        except Exception as e:
            log.error("Telegram control handler error: %s", e)
            return False

    # ---- health server ----
    def _health_payload(self) -> Dict[str, Any]:
        return {**self._status_payload()}

    def start_health(self) -> None:
        def _run():
            health_server.run(callback=self._health_payload,
                              host=self.settings.server.host,
                              port=self.settings.server.port)
        self._health_thread = threading.Thread(target=_run, name="health", daemon=True)
        self._health_thread.start()

    # ---- signals ----
    def _install_signals(self) -> None:
        def _sig(signum, _frame):
            log.info("Received signal %s -> shutting down...", signum)
            self._shutdown = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)
            except Exception:
                pass

    # ---- main loop ----
    def run(self) -> None:
        self._install_signals()
        self.start_health()

        # Telegram (only if token available)
        try:
            self.tg = TelegramController(
                status_callback=self._status_payload,
                control_callback=self._handle_telegram_cmd,
                summary_callback=lambda: "No summary yet.",
            )
            self.tg.start_polling()
            self.tg.send_startup_alert()
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)

        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        while not self._shutdown:
            try:
                result = self.runner.run_once()
                if result:
                    log.info("Signal: %s", result)
                time.sleep(2.0)
            except Exception as e:
                log.exception("Main loop error: %s", e)
                time.sleep(2.0)

        if self.tg:
            self.tg.stop_polling()


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
    print("Unknown command:", cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main())