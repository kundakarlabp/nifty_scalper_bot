# src/main.py
"""
Entry point for the Nifty Scalper Bot.

CLI:
  python -m src.main start   -> init trader, start trading, run schedule loop
  python -m src.main stop    -> stop trading (polling stays alive)
  python -m src.main status  -> print current status
  python -m src.main health  -> run health check and print

The RealTimeTrader starts Telegram polling in a daemon thread on init.
The main thread only runs schedule.run_pending() without blocking Telegram.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Optional

import schedule

from src.data_streaming.realtime_trader import RealTimeTrader

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- singleton ----------
_trader: Optional[RealTimeTrader] = None


def get_trader() -> RealTimeTrader:
    """Lazy-init singleton. Also starts Telegram polling (inside trader.__init__)."""
    global _trader
    if _trader is None:
        logger.info("🧠 Initializing RealTimeTrader…")
        _trader = RealTimeTrader()
    return _trader


# ---------- graceful shutdown ----------
def _graceful_shutdown(signum, _frame):
    logger.info("🛑 Received signal %s. Shutting down…", signum)
    try:
        if _trader:
            _trader.shutdown()  # stops trading + stops telegram polling daemons
    finally:
        # give logs a moment to flush in containers
        time.sleep(0.3)
        sys.exit(0)


# ---------- CLI ----------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status|health]")
        sys.exit(1)

    command = sys.argv[1].strip().lower()

    # Register signals early
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    trader = get_trader()  # starts Telegram poller inside

    if command == "start":
        logger.info("🚀 Starting trading loop…")
        trader.start()  # flip is_trading = True (safe if already started)

        # Foreground loop: run schedule jobs created by trader (e.g., _smart_fetch_and_process)
        logger.info("🟢 Main loop active. Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(0.5)  # small sleep avoids CPU burn
        except KeyboardInterrupt:
            _graceful_shutdown(signal.SIGINT, None)

    elif command == "stop":
        logger.info("🛑 Stopping trading logic (Telegram polling stays alive)…")
        trader.stop()
        sys.exit(0)

    elif command == "status":
        logger.info("📊 Fetching bot status…")
        status = trader.get_status()
        print("📊 Bot Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    elif command == "health":
        logger.info("🩺 Running health check…")
        trader._run_health_check()  # uses Telegram + logs; keep as is for parity
        sys.exit(0)

    else:
        print(f"❌ Unknown command: {command}")
        print("Usage: python -m src.main [start|stop|status|health]")
        sys.exit(1)


if __name__ == "__main__":
    main()
