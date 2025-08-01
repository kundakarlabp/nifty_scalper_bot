"""
Entry point for the Nifty scalper bot.

This module provides a simple CLI interface to start or stop the bot
manually. When run with ``python -m src.main start`` it will spin up a
``RealTimeTrader`` instance, begin Telegram polling and await incoming
market data via the ``process_bar`` method. In the absence of a live
data feed the bot will simply idle and respond to Telegram commands.

Usage::

    python -m src.main start    # start the bot and polling
    python -m src.main stop     # stop a running bot
    python -m src.main status   # print current status

In typical deployment the bot will be managed via ``manage_bot.sh`` or
the Render `render.yaml` rather than invoking this module directly.
"""

from __future__ import annotations

import logging
import sys
import time
from src.data_streaming.realtime_trader import RealTimeTrader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_trader: RealTimeTrader | None = None


def get_trader() -> RealTimeTrader:
    """Return a singleton instance of ``RealTimeTrader``."""
    global _trader
    if _trader is None:
        _trader = RealTimeTrader()
    return _trader


def main() -> None:
    """Command-line interface for starting/stopping the bot."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [start|stop|status]")
        return

    command = sys.argv[1].lower()
    trader = get_trader()

    if command == "start":
        trader.start()
        try:
            while trader.is_trading:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received. Stopping bot...")
            trader.stop()

    elif command == "stop":
        trader.stop()

    elif command == "status":
        status = trader.get_status()
        print("📊 Bot Status:")
        for k, v in status.items():
            print(f"{k}: {v}")

    else:
        print(f"❓ Unknown command: {command}")
        print("Usage: python -m src.main [start|stop|status]")


if __name__ == "__main__":
    main()