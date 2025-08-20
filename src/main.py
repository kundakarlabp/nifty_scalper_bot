# src/main.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.config import settings
from src.server import health as health_server

# Optional import of KiteConnect
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# Local imports for runner wiring
from src.data.source import LiveKiteSource
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor
from src.risk.session import TradingSession
from src.risk.position_sizing import PositionSizing
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ------------ time helpers ------------
def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ------------ env helpers ------------
def _first_nonempty(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


# ------------ kite + data source wiring ------------
def _build_kite_or_none() -> Optional[Any]:
    if KiteConnect is None:
        log.warning("kiteconnect not installed; running without live feed.")
        return None

    api_key = _first_nonempty("ZERODHA_API_KEY", "KITE_API_KEY") or settings.zerodha.api_key
    access_token = _first_nonempty("ZERODHA_ACCESS_TOKEN", "KITE_ACCESS_TOKEN", "ACCESS_TOKEN") or settings.zerodha.access_token
    api_secret = _first_nonempty("ZERODHA_API_SECRET", "KITE_API_SECRET") or settings.zerodha.api_secret

    log.info("Kite env present: api_key=%s access_token=%s (secret=%s)",
             "Y" if api_key else "N", "Y" if access_token else "N", "Y" if api_secret else "N")

    if not api_key or not access_token:
        log.warning("Kite credentials missing; running without live feed.")
        return None

    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # cheap sanity check
        try:
            kite.margins()
        except Exception:
            kite.profile()
        log.info("KiteConnect OK (token valid).")
        return kite
    except Exception as e:
        log.critical("Failed to initialize KiteConnect with provided creds: %s", e, exc_info=False)
        return None


def _load_live_source(kite: Optional[Any]) -> Optional[LiveKiteSource]:
    if kite is None:
        return None
    try:
        ds = LiveKiteSource(kite=kite)
        ds.connect()
        log.info("LiveKiteSource connected.")
        return ds
    except Exception as e:
        log.warning("Data source connect failed: %s", e)
        return None


# ------------ application ------------
class Application:
    def __init__(self) -> None:
        env_live = os.getenv("ENABLE_LIVE_TRADING")
        self.live_trading = (env_live.strip().lower() not in ("0", "false", "no")) if env_live is not None else bool(settings.enable_live_trading)
        self.quality = "AUTO"
        self.regime = "auto"
        self._started_ts = time.time()
        self._shutdown = False

        self.kite = _build_kite_or_none()
        self.data_source = _load_live_source(self.kite)

        # Build all Runner deps here
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite)
        self.session = TradingSession(equity_default=getattr(settings.risk, "default_equity", 30000.0))
        self.sizer = PositionSizing()
        self.telegram: Optional[TelegramController] = None

        # Health thread
        self._health_thread: Optional[threading.Thread] = None

        # Compose the runner as the module expects
        self.runner = StrategyRunner(
            data_source=self.data_source,
            strategy=self.strategy,
            order_executor=self.executor,
            trading_session=self.session,
            position_sizer=self.sizer,
            telegram_controller=None,  # we attach the instance after we create TelegramController below
        )

    # ---- status for health/Telegram ----
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
            "kite": bool(self.kite),
            "data_source": bool(self.data_source),
        }

    # ---- Telegram control handler ----
    def _handle_telegram_cmd(self, command: str, arg: str) -> bool:
        try:
            if command == "mode":
                if arg == "live": self.live_trading = True;  return True
                if arg == "shadow": self.live_trading = False; return True
                return False
            if command == "quality":
                v = arg.upper()
                if v in ("AUTO", "CONSERVATIVE", "AGGRESSIVE"):
                    self.quality = v; return True
                return False
            if command == "start": self.live_trading = True;  return True
            if command == "stop":  self.live_trading = False; return True
            if command == "refresh": return True
            if command == "health":  return True
            if command == "emergency": self.live_trading = False; return True
            return False
        except Exception as e:
            log.error("Telegram control handler error: %s", e)
            return False

    # ---- health server ----
    def start_health(self) -> None:
        def _run():
            health_server.run(callback=self._status_payload,
                              host=getattr(settings.server, "host", "0.0.0.0"),
                              port=getattr(settings.server, "port", 8000))
        self._health_thread = threading.Thread(target=_run, name="health", daemon=True)
        self._health_thread.start()

    # ---- signals ----
    def _install_signals(self) -> None:
        def _sig(signum, _frame):
            log.info("Received signal %s -> shutting down...", signum)
            self._shutdown = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try: signal.signal(s, _sig)
            except Exception: pass

    # ---- main loop (responsive sleep) ----
    def run(self) -> None:
        self._install_signals()
        self.start_health()

        try:
            self.telegram = TelegramController(
                status_callback=self._status_payload,
                control_callback=self._handle_telegram_cmd,
                summary_callback=lambda: "No summary yet.",
            )
            self.telegram.start_polling()
            self.telegram.send_startup_alert()
            # attach it to runner for signal/status messages
            self.runner.telegram = self.telegram  # type: ignore[attr-defined]
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)

        log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)

        # Responsive loop: check shutdown every 100ms
        while not self._shutdown:
            try:
                self.runner.tick()
            except Exception as e:
                log.exception("Main loop error: %s", e)
            # split 2.0s into short waits
            for _ in range(20):
                if self._shutdown: break
                time.sleep(0.1)

        if self.telegram:
            self.telegram.stop_polling()


# ------------ cli ------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        Application().run()
        return 0
    print("Unknown command:", cmd);  return 2


if __name__ == "__main__":
    sys.exit(main())
