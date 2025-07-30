# src/main.py
"""
Main entry point for the Nifty 50 Scalper Bot.
Handles configuration, initialization of core components,
instrument selection, and starts the trading loop.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import logging.handlers
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure correct path resolution for imports if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiteconnect import KiteConnect

# Import configuration using the Config class (aligns with provided order_executor.py)
from config import Config

# Import core modules
from src.data_streaming.realtime_trader import RealTimeTrader
from src.execution.order_executor import OrderExecutor
# Import utility functions for instrument selection
from src.utils.expiry_selector import get_next_weekly_expiry
from src.utils.strike_selector import select_nifty_option_strikes

# --- Setup Enhanced Logging with Rotation ---
def setup_logging():
    """Configures the logging system with file rotation."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / "trading_bot.log"

    if not logging.getLogger().hasHandlers():
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger = logging.getLogger(__name__)
        logger.info("📜 Logging configured with rotation.")

# Call the setup function at the start
setup_logging()

# Get the logger for this module
logger = logging.getLogger(__name__)
# --- End of Logging Setup ---


# --- Instrument Selection ---
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
        # logger.error("❌ Instrument selection failed. Exiting application.")
        # return [] # Return empty list to signal failure

        # Option 2: Use a default/fallback token for testing (Comment out Option 1 if using this)
        logger.info("ℹ️ Using fallback Nifty 50 Index token (256265) for testing.")
        selected_tokens = [256265] # Nifty 50 Index Token

    if selected_tokens:
        logger.info(f"✅ Final selected instrument tokens: {selected_tokens}")
    else:
        logger.warning("⚠️ Instrument selection returned an empty list.")

    return selected_tokens


# --- Main Bot Logic ---
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
    if not instrument_tokens_to_trade: # If you chose Option 1 (exit on failure)
         logger.error("❌ No instruments available to trade. Exiting.")
         return # Exit the main function

    # --- 3. Initialize Order Executor ---
    order_executor = OrderExecutor(kite=kite)
    logger.info("✅ OrderExecutor initialized.")

    # --- 4. Initialize RealTime Trader ---
    # Pass the OrderExecutor instance to the trader's constructor
    # (Assuming RealTimeTrader.__init__ was updated to accept order_executor)
    trader = RealTimeTrader(order_executor=order_executor)

    # Enable/disable live execution based on --trade flag
    trader.enable_trading(enable=args.trade)
    logger.info(f"{'✅' if args.trade else '⚠️'} Trading execution is {'ENABLED' if args.trade else 'DISABLED'}.")

    # --- 5. Add Selected Instruments to Trader ---
    successfully_added = 0
    for token in instrument_tokens_to_trade:
        if trader.add_trading_instrument(token):
            logger.info(f"➕ Successfully added instrument token {token} to trader.")
            successfully_added += 1
        else:
            logger.error(f"❌ Failed to add instrument token {token}.")

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
            # --- 7. Keep the main thread alive ---
            try:
                # Simple loop to keep the script running
                # The trader runs its streaming/processing in background threads
                while trader.is_trading: # Check a flag that stop_trading sets to False
                    # You could add periodic status checks or other main-loop tasks here
                    # For now, just sleep to prevent a busy loop
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
