"""
Entry point for the Nifty Scalper Bot.
"""

from __future__ import annotations

import logging
import sys
import time
import signal
import threading

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
        _trader = RealTimeTrader()
    return _trader


def graceful_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    logger.info(f"🛑 Received signal {signum}. Shutting down...")
    if _trader:
        _trader.shutdown()
    sys.exit(0)


def wait_for_commands():
    """Keep the main thread alive to handle Telegram messages."""
    try:
        logger.info("✅ Bot is running. Awaiting commands via Telegram...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)

    command = sys.argv[1].lower().strip()
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    trader = get_trader()

    if command == "start":
        logger.info("🚀 Starting Nifty Scalper Bot...")

        # 🔁 Start trading loop in background thread
        logger.info("Starting trading loop...")
        trading_thread = threading.Thread(target=trader.run, daemon=True)
        trading_thread.start()
        logger.info("Trading loop started.")

        # 📞 Start Telegram bot in main thread (blocking)
        logger.info("Starting Telegram...")
        trader.start_telegram()

        # Keep main thread alive for Telegram
        wait_for_commands()

    elif command == "stop":
        logger.info("🛑 Stopping trading engine...")
        trader.stop()
        logger.info("Trading stopped. Telegram polling remains active.")
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