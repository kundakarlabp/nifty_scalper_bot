# main.py

"""
Entry point for the Nifty Scalper Bot.

Provides a CLI to control the bot:
    python -m src.main start    → Initialize bot, start Telegram, and begin main processing loop
    python -m src.main stop     → Stop only trading logic (polling stays alive)
    python -m src.main status   → Print current bot status

The bot will respond to Telegram commands like /start, /stop, /mode live.
Always run this as a module to ensure correct imports.
"""

from __future__ import annotations

import logging
import sys
import time
import signal
import schedule # Import schedule if your trader uses it

# ✅ Correct import path based on your actual file
from src.data_streaming.realtime_trader import RealTimeTrader

# Configure logging
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
        _trader = RealTimeTrader() # This initializes and starts Telegram polling in a daemon thread
    return _trader


def graceful_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    logger.info(f"🛑 Received signal {signum}. Shutting down...")
    if _trader:
        _trader.shutdown()
    sys.exit(0)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)

    command = sys.argv[1].lower().strip()
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    trader = get_trader() # This initializes the trader and starts Telegram polling in __init__

    if command == "start":
        logger.info("🚀 Starting Nifty Scalper Bot main processing loop...")
        # Trader is initialized, Telegram polling started in __init__ (daemon thread)
        # Now run the main processing loop in the main thread.
        # This loop is responsible for running scheduled tasks or periodic checks.
        try:
            logger.info("🟢 Entering main processing loop. Press Ctrl+C to stop.")
            # The main loop runs scheduled tasks or other periodic logic.
            # If you have scheduled tasks set up elsewhere (e.g., in trader.__init__ or process_data_and_trade),
            # this loop will execute them.
            while True:
                # Run any scheduled tasks (e.g., data fetching, signal checking)
                # Make sure you have scheduled tasks set up, e.g.:
                # schedule.every(1).minutes.do(trader.some_periodic_method)
                schedule.run_pending()
                time.sleep(1) # Check for scheduled tasks every second
        except KeyboardInterrupt:
            logger.info("🛑 KeyboardInterrupt received in main loop.")
            graceful_shutdown(signal.SIGINT, None)

    elif command == "stop":
        logger.info("🛑 Stopping trading logic...")
        trader.stop() # This sets trader.is_trading = False
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
