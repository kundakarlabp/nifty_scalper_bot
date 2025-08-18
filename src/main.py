from __future__ import annotations
import os, sys, time, logging

# Ensure project root in sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- config loader ----
try:
    # If utils.config exists
    from src.utils.config import load_env  # type: ignore
except ModuleNotFoundError:
    # fallback: use load_dotenv from python-dotenv
    from dotenv import load_dotenv as load_env

from src.data_streaming.realtime_trader import RealTimeTrader

log = logging.getLogger("main")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def start():
    # Load environment variables
    load_env()
    log.info("✅ Environment loaded")
    trader = RealTimeTrader()
    try:
        trader.start()
        log.info("🧭 Entering main loop (Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("👋 KeyboardInterrupt received.")
    finally:
        trader.shutdown()
        log.info("✅ Exit complete.")

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "start":
        start()
    else:
        print("Usage: python -m src.main start")
