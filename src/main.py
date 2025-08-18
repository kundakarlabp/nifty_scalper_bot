from __future__ import annotations
import os, sys, time, logging

# Ensure project root is on sys.path (works in Docker, Railway, Codespaces)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- config loader (support both layouts) ----
try:
    # Preferred (new layout)
    from src.utils.config import load_env  # type: ignore
except ModuleNotFoundError:
    # Fallback (existing layout)
    from src.config import load_env  # type: ignore

from src.data_streaming.realtime_trader import RealTimeTrader

log = logging.getLogger("main")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def start():
    load_env()  # prints "Loaded environment from ..."
    trader = RealTimeTrader()
    try:
        trader.start()
        log.info("🧭 Entering main loop (Ctrl+C to stop)...")
        while True:
            time.sleep(1)  # keep process alive
    except KeyboardInterrupt:
        log.info("👋 KeyboardInterrupt received.")
    finally:
        log.info("🔻 Shutting down…")
        trader.shutdown()
        log.info("✅ Exit complete.")

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "start":
        start()
    else:
        print("Usage: python -m src.main start")
