"""
Entry point for the Nifty scalper bot.

This module provides a simple CLI interface to start or stop the bot
manually. When run with:

    python -m src.main start    # start the bot and polling
    python -m src.main stop     # stop a running bot
    python -m src.main status   # print current status

It will spin up a `RealTimeTrader` instance, begin Telegram polling and await incoming
market data via the `process_bar` method. In the absence of a live data feed,
the bot will simply idle and respond to Telegram commands.

In typical deployment the bot will be managed via `manage_bot.sh` or
the Render `render.yaml` rather than invoking this module directly.
"""

from __future__ import annotations

import logging
import sys
import time
import signal

from src.core.real_time_trader import RealTimeTrader  # Updated path

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Singleton trader instance
_trader: RealTimeTrader | None = None


def get_trader() -> RealTimeTrader:
    """
    Return a singleton instance of `RealTimeTrader`.
    Initializes only once, and avoids sending duplicate startup alerts.
    """
    global _trader
    if _trader is None:
        logger.info("🧠 Initializing RealTimeTrader singleton...")
        _trader = RealTimeTrader()
    return _trader


def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"🛑 Received signal {signum}. Shutting down...")
    if _trader:
        _trader.shutdown()
    sys.exit(0)


def wait_for_commands():
    """
    Keep the main thread alive and responsive to Telegram.
    Unlike trading loop, this runs regardless of is_trading.
    """
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        graceful_shutdown(signal.SIGINT, None)


def main() -> None:
    """Command-line interface for starting/stopping the bot."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)

    command = sys.argv[1].lower()

    # Register signal handlers early
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Always get trader instance (safe even for status)
    trader = get_trader()

    if command == "start":
        logger.info("🚀 Starting Nifty Scalper Bot...")
        trader.start()
        logger.info("Bot started. Awaiting commands via Telegram...")
        wait_for_commands()  # Stay alive forever (or until signal)

    elif command == "stop":
        logger.info("🛑 Stopping Nifty Scalper Bot...")
        trader.stop()
        # Do NOT call shutdown() here — just stop trading
        logger.info("Bot stopped. Telegram polling remains active for further commands.")

    elif command == "status":
        logger.info("📊 Fetching bot status...")
        status = trader.get_status()
        print("📊 Bot Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    else:
        print(f"❌ Unknown command: {command}")
        print("Usage: python -m src.main [start|stop|status]")
        sys.exit(1)


if __name__ == "__main__":
    main()