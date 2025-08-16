# src/main.py
from __future__ import annotations

"""
Main entrypoint for the real-time trader.

Features
- .env loader (no external dependency)
- Config-driven logging
- Optional health server runner
- CLI: run/start/stop/status/auth-check with extra flags
- Clean shutdown (SIGINT/SIGTERM) with thread joins
"""

import argparse
import logging
import os
import signal
import sys
from threading import Event, Thread
from typing import Optional

# --- lightweight .env loader (optional) ---
def _load_env_from_file(path: str = ".env") -> None:
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # do not overwrite if already set
                os.environ.setdefault(k, v)
    except Exception:
        # env loading must never crash startup
        pass

_load_env_from_file()

# Local imports AFTER env is loaded
from src.config import Config  # noqa: E402
from src.data_streaming.realtime_trader import RealTimeTrader  # noqa: E402
from src.server.health import run as run_health  # noqa: E402

# Optional: Zerodha auth preflight
try:
    from src.auth.zerodha_auth import check_live_credentials
except Exception:  # keep optional
    def check_live_credentials():
        return True, []

# ---- logging ----
def _setup_logging() -> None:
    level = getattr(logging, os.environ.get("LOG_LEVEL", getattr(Config, "LOG_LEVEL", "INFO")), logging.INFO)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

_setup_logging()
logger = logging.getLogger("main")

# ---- globals & lifecycle ----
stop_event = Event()
_trader: Optional[RealTimeTrader] = None
_trader_thread: Optional[Thread] = None
_health_thread: Optional[Thread] = None

def _graceful_exit(*_):
    logger.info("Shutdown signal received.")
    try:
        if _trader:
            _trader.stop()
    except Exception:
        pass
    stop_event.set()

def _start_health_server(port: int) -> Optional[Thread]:
    if port <= 0:
        return None
    t = Thread(target=run_health, kwargs={"port": port}, name="HealthServer", daemon=True)
    t.start()
    logger.info("Health server listening on :%d", port)
    return t

def _start_trader_loop(trader: RealTimeTrader) -> Thread:
    t = Thread(target=trader.run, name="TraderLoop", daemon=True)
    t.start()
    logger.info("Trader main loop started.")
    return t

# ---- commands ----
def cmd_run(args) -> int:
    global _trader, _trader_thread, _health_thread

    # Apply runtime toggles
    if args.shadow:
        os.environ["ENABLE_LIVE_TRADING"] = "false"
    if args.live:
        os.environ["ENABLE_LIVE_TRADING"] = "true"
    if args.no_telegram:
        os.environ["ENABLE_TELEGRAM"] = "false"

    # Instantiate
    _trader = RealTimeTrader()

    # Handle signals
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)

    # Health server (optional)
    _health_thread = _start_health_server(args.health_port)

    # Start trader
    _trader_thread = _start_trader_loop(_trader)

    # Keep process alive
    stop_event.wait()

    # Join threads briefly on exit (best-effort)
    try:
        if _trader_thread and _trader_thread.is_alive():
            _trader_thread.join(timeout=2)
        if _health_thread and _health_thread.is_alive():
            _health_thread.join(timeout=1)
    except Exception:
        pass

    logger.info("Stopped.")
    return 0

def cmd_start(args) -> int:
    # start toggles only apply if you run in this process; typically you use `run`
    return cmd_run(args)

def cmd_stop(_args) -> int:
    # Local process stop helper; in real deployments you'd signal the supervisor (systemd/docker)
    logger.info("Stop requested; send SIGTERM/SIGINT to the process.")
    return 0

def cmd_status(_args) -> int:
    try:
        t = RealTimeTrader()
        status = t.get_status()
        logger.info("Status: %s", status)
        return 0
    except Exception as e:
        logger.error("Status error: %s", e, exc_info=True)
        return 1

def cmd_auth_check(_args) -> int:
    ok, missing = check_live_credentials()
    if ok:
        print("✅ Live credentials found.")
        return 0
    print("❌ Missing credentials:", ", ".join(missing))
    return 2

# ---- CLI ----
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nifty scalper real-time trader")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--health-port", type=int, default=8000, help="Health server port (0 to disable)")
    mode = common.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Force LIVE mode")
    mode.add_argument("--shadow", action="store_true", help="Force SHADOW (paper) mode")
    common.add_argument("--no-telegram", action="store_true", help="Disable Telegram on startup")

    sub.add_parser("run", parents=[common], help="Run the trader (recommended)")
    sub.add_parser("start", parents=[common], help="Alias of run")
    sub.add_parser("stop", help="Signal a managed process to stop (informational)")
    sub.add_parser("status", help="Print a quick status snapshot")
    sub.add_parser("auth-check", help="Verify Zerodha credentials in env")

    return p

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Fast fail on auth if explicitly requesting LIVE at startup
    if args.cmd in ("run", "start") and (args.live or os.getenv("ENABLE_LIVE_TRADING", "").lower() in ("1", "true", "yes")):
        ok, missing = check_live_credentials()
        if not ok:
            logger.error("Missing live credentials: %s", ", ".join(missing))
            return 2

    dispatch = {
        "run": cmd_run,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "auth-check": cmd_auth_check,
    }
    return dispatch[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())