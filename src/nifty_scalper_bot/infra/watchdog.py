# src/nifty_scalper_bot/infra/watchdog.py
import threading
import time
import logging
import os

def start_watchdog(market_data_manager):
    """Start the data health monitor in a background thread."""
    logger = logging.getLogger("nifty_scalper_bot.watchdog")
    
    def _monitor():
        logger.info("✅ Watchdog Started")
        while True:
            time.sleep(60)
            last_tick = getattr(market_data_manager, "last_tick_time", 0)
            if last_tick > 0 and (time.time() - last_tick > 180):
                logger.critical("🚨 FATAL: No data. Exiting.")
                os._exit(1)
    
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return thread
