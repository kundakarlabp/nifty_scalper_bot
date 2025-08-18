from __future__ import annotations
import os
import sys
import time
import logging

# Ensure local project imports work if PYTHONPATH not set
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_streaming.realtime_trader import RealTimeTrader  # noqa
from src.utils.config import load_env  # if your config loader differs, keep existing

log = logging.getLogger("main")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def start():
    load_env()  # prints “Loaded environment …” in your build
    trader = RealTimeTrader()
    try:
        trader.start()  # schedules jobs, starts Telegram polling
        log.info("🧭 Entering main loop (Ctrl+C to stop)...")
        while True:
            # If your trader exposes a tick/scheduler runner, call it here.
            # Otherwise just sleep; internal scheduler threads will run jobs.
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("👋 KeyboardInterrupt received.")
    finally:
        log.info("🔻 Shutting down…")
        trader.shutdown()
        log.info("✅ Exit complete.")

if __name__ == "__main__":
    # Support `python3 -m src.main start`
    if len(sys.argv) == 1 or sys.argv[1] == "start":
        start()
    else:
        print("Usage: python -m src.main start")
