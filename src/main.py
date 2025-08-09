"""
CLI entry point for the Nifty Scalper Bot.

Usage:
  python -m src.main start   -> init bot, start Telegram, enter main loop (schedule)
  python -m src.main stop    -> stop only trading logic (polling stays alive)
  python -m src.main status  -> print current bot status
"""

from __future__ import annotations

import logging
import sys
import time
import signal
from pathlib import Path

# --- Load .env automatically (works in Docker/Railway and locally) ---
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("⚠️  No .env file found — using system environment variables")
except Exception as _e:
    # Not fatal: Config also attempts to load .env itself.
    pass

import schedule  # ensure installed
from src.data_streaming.realtime_trader import RealTimeTrader

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Singleton instance
_trader: RealTimeTrader | None = None


def get_trader() -> RealTimeTrader:
    """Lazy-initialize the RealTimeTrader singleton."""
    global _trader
    if _trader is None:
        logger.info("🧠 Initializing RealTimeTrader...")
        _trader = RealTimeTrader()  # starts Telegram polling in its own daemon thread
    return _trader


def graceful_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    logger.info(f"🛑 Received signal {signum}. Shutting down...")
    try:
        if _trader:
            _trader.shutdown()
    finally:
        sys.exit(0)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)

    command = sys.argv[1].lower().strip()
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    trader = get_trader()  # initializes the trader and starts Telegram polling in __init__

    if command == "start":
        logger.info("🚀 Starting Nifty Scalper Bot main processing loop...")
        # Trader schedules its own tasks (e.g., smart_fetch_and_process) in __init__.
        # We just run the scheduler tick.
        try:
            logger.info("🟢 Entering main processing loop. Press Ctrl+C to stop.")
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 KeyboardInterrupt received in main loop.")
            graceful_shutdown(signal.SIGINT, None)

    elif command == "stop":
        logger.info("🛑 Stopping trading logic...")
        trader.stop()  # sets is_trading = False; Telegram polling remains active
        logger.info("Trading logic stopped. Telegram polling remains active.")
        sys.exit(0)

    elif command == "status":
        logger.info("📊 Fetching bot status...")
        status = trader.get_status()
        print("📊 Bot Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    else:
        print(f"❌ Unknown command: {command}")
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()