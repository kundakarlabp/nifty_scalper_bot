from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Optional imports guarded
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# Try to import your data source if present
def _load_data_source() -> Optional[Any]:
    """
    Tries several known data sources; returns instance or None.
    """
    # Try: src.data.source: LiveKiteSource
    try:
        from src.data.source import LiveKiteSource  # type: ignore
        return LiveKiteSource()
    except Exception:
        pass

    # Try: src.data.live (alternative module naming used in some repos)
    try:
        from src.data.live import LiveKiteSource  # type: ignore
        return LiveKiteSource()
    except Exception:
        pass

    # If nothing found, return None (shadow mode)
    return None


def _build_kite() -> Optional[Any]:
    """
    Build a KiteConnect instance if creds available; else None.
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
        return kite
    except Exception as e:
        log.critical("Failed to initialize KiteConnect: %s", e)
        return None


class Application:
    """
    Lightweight orchestrator: health server + StrategyRunner loop.
    Works in shadow mode if no Kite/data source.
    """

    def __init__(self) -> None:
        self.settings = settings
        self.live_trading = bool(self.settings.enable_live_trading)
        self._shutdown = False

        # Dependencies (optional)
        self.kite = _build_kite()
        self.data_source = _load_data_source()
        if self.data_source is None:
            log.warning("Data source not available (no Kite or module missing). Running without live feed.")

        self.runner = StrategyRunner(
            data_source=self.data_source,
            kite=self.kite,
        )

        # Health server
        self._health_thread: Optional[threading.Thread] = None

    # ---- health ----
    def _health_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "live_trading": self.live_trading,
            "kite": bool(self.kite is not None),
            "data_source": bool(self.data_source is not None),
            "preferred_exit_mode": self.settings.preferred_exit_mode,
        }

    def start_health(self) -> None:
        def _run():
            health_server.run(callback=self._health_payload,
                              host=self.settings.server.host,
                              port=self.settings.server.port)

        self._health_thread = threading.Thread(target=_run, name="health", daemon=True)
        self._health_thread.start()

    # ---- shutdown ----
    def _install_signals(self) -> None:
        def _sig_handler(signum, _frame):
            log.info("Received signal %s -> shutting down...", signum)
            self._shutdown = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig_handler)
            except Exception:
                pass

    # ---- main loop ----
    def run(self) -> None:
        """
        Super-simple cadence loop: call runner.run_once(), sleep.
        Adapt cadence as needed.
        """
        self._install_signals()
        self.start_health()

        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # If you prefer to short-circuit when missing deps, uncomment:
        # if not self.kite or not self.data_source:
        #     log.warning("Missing live deps; staying in health-only mode.")
        #     while not self._shutdown:
        #         time.sleep(1)
        #     return

        # main cadence loop
        while not self._shutdown:
            try:
                result = self.runner.run_once()
                if result:
                    log.info("Signal: %s", result)
                time.sleep(2.0)  # adjust to taste
            except Exception as e:
                log.exception("Main loop error: %s", e)
                time.sleep(2.0)


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