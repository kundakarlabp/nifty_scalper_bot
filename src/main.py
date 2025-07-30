# src/main.py
"""
Main entry point for the Nifty 50 Scalper Bot.
Handles configuration, initialization of core components,
instrument selection, and starts the trading loop.
"""
import sys
import os
# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import monkey_patch_twisted_signals

import argparse
import logging
from pathlib import Path
# Import logging.handlers for log rotation
import logging.handlers
# CRITICAL FIX: Import time module for main loop
import time
import threading

# Import KiteConnect
from kiteconnect import KiteConnect

# Import configuration using the Config class for consistency
from config import Config

# Import core modules
from src.data_streaming.realtime_trader import RealTimeTrader
# Import the new OrderExecutor
from src.execution.order_executor import OrderExecutor
# Import utility functions for instrument selection
from src.utils.expiry_selector import get_next_weekly_expiry
from src.utils.strike_selector import select_nifty_option_strikes

# --- Setup Enhanced Logging with Rotation ---
def setup_logging():
    """Configures the logging system with file rotation."""
    # Ensure the logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define the log file path
    log_file_path = logs_dir / "trading_bot.log"

    # Configure logging only if it hasn't been configured yet
    if not logging.getLogger().hasHandlers():
        # Create a rotating file handler (e.g., 5 files, 10MB each)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024, # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG) # Capture detailed logs in file

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) # Show INFO and above on console

        # Create a formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Get the root logger and add handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # Set root level to DEBUG to allow filtering by handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Get a logger for this specific module
        logger = logging.getLogger(__name__)
        logger.info("📜 Logging configured with rotation.")

# Call the setup function at the start
setup_logging()

# Now get the logger for main.py after setup
logger = logging.getLogger(__name__)

# --- End of Logging Setup ---


def select_instruments(kite_client: KiteConnect) -> list:
    """
    Selects the Nifty 50 option instruments to trade based on expiry and strike logic.

    Args:
        kite_client (KiteConnect): An authenticated KiteConnect instance.

    Returns:
        list: A list of instrument tokens to add to the trader.
    """
    logger.info("🔍 Starting instrument selection...")
    selected_tokens = []

    try:
        # 1. Determine Next Expiry
        next_expiry_date = get_next_weekly_expiry()
        expiry_str = next_expiry_date.strftime('%Y-%m-%d')
        logger.info(f"🎯 Selected Expiry Date: {expiry_str}")

        # 2. Select Strikes
        # Example: Select ATM Call and Put for the next expiry
        logger.info("🔍 Selecting ATM Call strike...")
        atm_ce_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="CE",
            strike_criteria="ATM"
            # instrument_mapping=pre_fetched_mapping # Optional optimization
        )
        if atm_ce_tokens:
            selected_tokens.extend(atm_ce_tokens)
            logger.info(f"✅ Selected ATM CE Token: {atm_ce_tokens[0]}")
        else:
            logger.error("❌ Failed to select ATM CE strike.")

        logger.info("🔍 Selecting ATM Put strike...")
        atm_pe_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="PE",
            strike_criteria="ATM"
        )
        if atm_pe_tokens:
            selected_tokens.extend(atm_pe_tokens)
            logger.info(f"✅ Selected ATM PE Token: {atm_pe_tokens[0]}")
        else:
            logger.error("❌ Failed to select ATM PE strike.")

    except Exception as e:
        logger.error(f"❌ Error during instrument selection: {e}", exc_info=True)

    # 3. Fallback logic if selection fails
    if not selected_tokens:
        logger.warning("⚠️ No instruments selected via strike selector.")
        # --- CRITICAL: Decide on Fallback Strategy ---
        # Option 1: Exit if no instruments are selected (Recommended for live trading)
        logger.error("❌ Instrument selection failed. Exiting application.")
        return [] # Return empty list to signal failure

        # Option 2: Use a default/fallback token for testing (Comment out Option 1 if using this)
        # logger.info("ℹ️ Using fallback Nifty 50 Index token (256265) for testing.")
        # selected_tokens = [256265] # Nifty 50 Index Token

    if selected_tokens:
        logger.info(f"✅ Final selected instrument tokens: {selected_tokens}")
    else:
        logger.warning("⚠️ Instrument selection returned an empty list.")

    return selected_tokens


def main():
    """Main function to initialize and run the bot."""
    parser = argparse.ArgumentParser(description="Nifty Scalper Bot")
    parser.add_argument('--mode', choices=['realtime', 'signal'], default='realtime',
                        help='Operational mode')
    parser.add_argument('--trade', action='store_true',
                        help='Enable live trade execution (requires valid Kite credentials)')
    args = parser.parse_args()

    logger.info("🚀 Initializing Nifty 50 Scalper Bot...")
    logger.info(f"🔁 Mode: {args.mode}, Trade Execution: {'ENABLED' if args.trade else 'DISABLED'}")

    # --- 1. Initialize Kite Connect ---
    if not Config.ZERODHA_API_KEY or not Config.KITE_ACCESS_TOKEN:
        logger.error("❌ Zerodha API credentials (ZERODHA_API_KEY, KITE_ACCESS_TOKEN) missing in config.")
        return

    kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
    try:
        kite.set_access_token(Config.KITE_ACCESS_TOKEN)
        logger.info("✅ Zerodha Kite Connect client initialized and authenticated.")
    except Exception as e:
        logger.error(f"❌ Failed to authenticate Kite client: {e}")
        return

    # --- 2. Select Instruments ---
    instrument_tokens_to_trade = select_instruments(kite)

    # Check if instrument selection was successful (based on your chosen fallback strategy)
    # If you chose Option 1 (exit on failure), this check is crucial.
    # If you chose Option 2 (fallback token), this might just log a warning.
    if not instrument_tokens_to_trade:
         logger.error("❌ No instruments available to trade. Exiting.")
         return # Exit the main function

    # --- 3. Initialize Order Executor ---
    # Pass the same authenticated kite instance
    order_executor = OrderExecutor(kite=kite)
    logger.info("✅ OrderExecutor initialized.")

    # --- 4. Initialize RealTime Trader ---
    # Pass the OrderExecutor instance to the trader
    # Ensure RealTimeTrader.__init__ accepts order_executor: Optional[OrderExecutor] = None
    trader = RealTimeTrader(order_executor=order_executor)

    # Enable/disable live execution based on --trade flag
    trader.enable_trading(enable=args.trade)
    logger.info(f"{'✅' if args.trade else '⚠️'} Trading execution is {'ENABLED' if args.trade else 'DISABLED'}.")

    # --- 5. Add Selected Instruments to Trader ---
    successfully_added = 0
    # Iterate through the list of selected instruments (dicts or ints)
    # Assuming select_nifty_option_strikes returns a list of token integers
    for token in instrument_tokens_to_trade:
        # If it returns dicts with token, symbol, exchange:
        # token = instrument_data['token']
        # symbol = instrument_data['symbol']
        # exchange = instrument_data['exchange']
        # For now, assuming it returns token integers
        # You need to get symbol/exchange from strike_selector or instrument data
        # Placeholder: You need to implement this mapping correctly
        # This is a critical missing piece from the logs.
        # For now, using placeholder symbol/exchange
        symbol = f"TOKEN_{token}_PLACEHOLDER" # REPLACE with actual symbol from selection
        exchange = "NFO" # REPLACE with actual exchange from selection

        # Pass all three required arguments to add_trading_instrument
        if trader.add_trading_instrument(token, symbol, exchange):
            logger.info(f"➕ Successfully added instrument: Token={token}, Symbol={symbol}, Exchange={exchange}")
            successfully_added += 1
        else:
            logger.error(f"❌ Failed to add instrument: Token={token}, Symbol={symbol}")

    if successfully_added == 0:
        logger.error("❌ Failed to add any instruments to the trader. Exiting.")
        return
    elif successfully_added < len(instrument_tokens_to_trade):
        logger.warning(f"⚠️ Only {successfully_added}/{len(instrument_tokens_to_trade)} instruments were added successfully.")

    # --- 6. Start Trading ---
    if args.mode == 'realtime':
        logger.info("🚀 Starting RealTime Trading Engine...")
        if trader.start_trading():
            logger.info("✅ RealTimeTrader started successfully.")
            logger.info("⏳ Bot is now running. Press Ctrl+C to stop.")
            # --- 7. Keep the main thread alive correctly ---
            try:
                # Use a simple loop that checks a flag
                # This keeps the main thread alive without blocking it
                # and allows Twisted/KiteTicker to run in the background
                while trader.is_trading: # Check a flag that stop_trading sets to False
                    # You could add periodic status checks or other main-loop tasks here
                    # For now, just sleep to prevent a busy loop
                    # CRITICAL FIX: time.sleep was undefined
                    time.sleep(1) # Check every second
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard Interrupt received. Stopping trader...")
            finally:
                trader.stop_trading()
                logger.info("🛑 Trader stopped. Bot shutdown complete.")
        else:
            logger.error("❌ Failed to start RealTimeTrader.")
    else:
        logger.info("🔄 Signal generation mode is selected but not implemented in this script.")


if __name__ == "__main__":
    main()
