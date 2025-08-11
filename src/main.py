from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
from threading import Event, Thread

from src.data_streaming.realtime_trader import RealTimeTrader
from src.server.health import run as run_health

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
stop_event = Event()
_trader: RealTimeTrader | None = None

def _graceful_exit(*_):
    try:
        if _trader:
            _trader.stop()
    except Exception:
        pass
    stop_event.set()

def run():
    global _trader
    _trader = RealTimeTrader()
    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)
    # Health server in background
    Thread(target=run_health, daemon=True).start()
    # Start trading loop (non-blocking run loop)
    Thread(target=_trader.run, daemon=True).start()
    # Keep process alive until stop signal
    stop_event.wait()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run", "start", "stop", "status"], help="Control trader")
    args = parser.parse_args()

    global _trader
    if args.cmd == "run":
        run()
    else:
        # For start/stop/status we need a lightweight instance for Telegram control paths
        _trader = RealTimeTrader()
        if args.cmd == "start":
            _trader.start()
        elif args.cmd == "stop":
            _trader.stop()
        elif args.cmd == "status":
            _trader.get_status()

if __name__ == "__main__":
    cli()