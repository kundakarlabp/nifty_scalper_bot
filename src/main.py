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

import argparse
import logging
from pathlib import Path
# Import logging.handlers for log rotation
import logging.handlers

# Import KiteConnect
from kiteconnect import KiteConnect

# Import configuration using the Config class for consistency
from config import Config

# Import core modules
from src.data_streaming.realtime_trader import RealTimeTrader
# Import the new OrderExecutor
from src.execution.order_executor import OrderExecutor
# Import utility functions for instrument selection
# Make sure these files exist and functions are implemented
from src.utils.expiry_selector import get_next_weekly_expiry # , get_nearest_expiry, get_monthly_expiry
from src.utils.strike_selector import select_nifty_option_strikes

# --- Setup Logging with Rotation ---
def setup_logging():
    """Configures the logging system with file rotation."""
    # Ensure the logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define the log file path
    log_file_path = logs_dir / "trading_bot.log"

    # Configure logging only if it hasn't been configured yet
    # This prevents issues if main.py is imported or run multiple times
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
    Returns a list of dictionaries containing token, symbol, and exchange.

    Args:
        kite_client (KiteConnect): An authenticated KiteConnect instance.

    Returns:
        list: A list of dicts like [{'token': int, 'symbol': str, 'exchange': str}, ...]
    """
    logger.info("🔍 Starting instrument selection...")
    selected_instruments = [] # List to hold dicts with token, symbol, exchange

    try:
        # 1. Determine Next Expiry
        next_expiry_date = get_next_weekly_expiry()
        expiry_str = next_expiry_date.strftime('%Y-%m-%d')
        logger.info(f"🎯 Selected Expiry Date: {expiry_str}")

        # 2. Select Strikes
        # Example: Select ATM Call and Put for the next expiry
        logger.info("🔍 Selecting ATM Call strike...")
        # The select_nifty_option_strikes function needs to return token, symbol, exchange
        # This example assumes it returns a list of tokens. You might need to modify
        # select_nifty_option_strikes or this logic to get symbol/exchange.
        # Let's assume select_nifty_option_strikes can be modified or we get the info differently.
        # For now, let's assume it returns a list of tokens, and we have a way to get details.
        # A more robust way is to modify select_nifty_option_strikes to return full details.

        # --- MODIFIED APPROACH: Assume select_nifty_option_strikes returns full details ---
        # This requires modifying strike_selector.py to return a list like:
        # [{'token': 123, 'symbol': 'NIFTY...', 'exchange': 'NFO'}, ...]
        # If it doesn't, you need to fetch the details based on the token.
        # Placeholder for fetching full instrument details
        def get_instrument_details(token_list):
             # This is a simplified placeholder.
             # In a real scenario, you'd use kite.instruments() or a pre-fetched map.
             # For now, let's assume strike_selector provides enough info or we fetch it.
             # If strike_selector returned tokens only, you'd need to look up symbol/exchange.
             # Let's assume for this example it returns full details or we can get them.
             # We'll simulate getting details. You need to replace this logic.
             details_list = []
             # Example: Fetch all NFO instruments once and create a map
             # all_nfo_instruments = kite_client.instruments("NFO")
             # instrument_map = {inst['instrument_token']: inst for inst in all_nfo_instruments}
             # for token in token_list:
             #     inst_data = instrument_map.get(token)
             #     if inst_data:
             #         details_list.append({
             #             'token': token,
             #             'symbol': inst_data['tradingsymbol'],
             #             'exchange': inst_data['exchange']
             #         })
             # return details_list

             # Since we don't have the full logic here, let's assume strike_selector
             # can be made to return this structure. For now, we'll handle tokens only
             # and fetch details. This is a common approach.
             if not token_list:
                 return []

             # Fetch instrument details for all selected tokens
             # This is a more robust way than assuming strike_selector provides symbol/exchange
             logger.debug("📥 Fetching instrument details for selected tokens...")
             try:
                 # Fetch all NFO instruments (or filter more cleverly if possible)
                 # Fetching all NFO can be slow. If strike_selector can provide exchange/symbol
                 # or a smaller list to fetch, that's better. For now, this works.
                 all_nfo_instruments = kite_client.instruments("NFO")
                 instrument_map = {inst['instrument_token']: inst for inst in all_nfo_instruments}
                 logger.debug(f"📥 Fetched {len(instrument_map)} NFO instruments.")

                 for token in token_list:
                     inst_data = instrument_map.get(token)
                     if inst_data:
                         details_list.append({
                             'token': token,
                             'symbol': inst_data['tradingsymbol'],
                             'exchange': inst_data['exchange']
                         })
                         logger.debug(f"📥 Found details for token {token}: {inst_data['tradingsymbol']}")
                     else:
                         logger.warning(f"⚠️ Could not find instrument details for token {token}")
             except Exception as e:
                 logger.error(f"❌ Error fetching instrument details: {e}", exc_info=True)
                 return [] # Return empty list on fetch error

             return details_list

        # --- Call strike selector for CE ---
        atm_ce_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="CE",
            strike_criteria="ATM"
            # instrument_mapping=pre_fetched_mapping # Optional optimization
        )
        if atm_ce_tokens:
            # Get full details for CE tokens
            ce_details = get_instrument_details(atm_ce_tokens)
            selected_instruments.extend(ce_details)
            for detail in ce_details:
                logger.info(f"✅ Selected ATM CE: Token={detail['token']}, Symbol={detail['symbol']}")
        else:
            logger.error("❌ Failed to select ATM CE strike.")

        # --- Call strike selector for PE ---
        logger.info("🔍 Selecting ATM Put strike...")
        atm_pe_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="PE",
            strike_criteria="ATM"
        )
        if atm_pe_tokens:
            # Get full details for PE tokens
            pe_details = get_instrument_details(atm_pe_tokens)
            selected_instruments.extend(pe_details)
            for detail in pe_details:
                logger.info(f"✅ Selected ATM PE: Token={detail['token']}, Symbol={detail['symbol']}")
        else:
            logger.error("❌ Failed to select ATM PE strike.")

    except Exception as e:
        logger.error(f"❌ Error during instrument selection: {e}", exc_info=True)

    # 3. Fallback logic if selection fails
    if not selected_instruments:
        logger.warning("⚠️ No instruments selected via strike selector.")
        # --- CRITICAL: Decide on Fallback Strategy ---
        # Option 1: Exit if no instruments are selected (Recommended for live trading)
        # logger.error("❌ Instrument selection failed. Exiting application.")
        # return [] # Return empty list to signal failure

        # Option 2: Use a default/fallback token for testing (Comment out Option 1 if using this)
        # logger.info("ℹ️ Using fallback Nifty 50 Index token (256265) for testing.")
        # selected_instruments = [{
        #     'token': 256265,
        #     'symbol': 'NIFTY 50', # Placeholder, might need real symbol
        #     'exchange': 'NSE'     # Placeholder
        # }]

    if selected_instruments:
        logger.info(f"✅ Final selected instruments: {selected_instruments}")
    else:
        logger.warning("⚠️ Instrument selection returned an empty list.")

    return selected_instruments # Return list of dicts


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
    # This now returns a list of dictionaries with token, symbol, exchange
    selected_instruments = select_instruments(kite)

    # Check if instrument selection was successful
    if not selected_instruments:
        logger.error("❌ No instruments available to trade. Exiting.")
        return

    # --- 3. Initialize Order Executor ---
    # Pass the same authenticated kite instance
    order_executor = OrderExecutor(kite=kite)
    logger.info("✅ OrderExecutor initialized.")

    # --- 4. Initialize RealTime Trader ---
    # Pass the OrderExecutor instance to the trader
    trader = RealTimeTrader(order_executor=order_executor)

    # Enable/disable live execution based on --trade flag
    trader.enable_trading(enable=args.trade)
    logger.info(f"{'✅' if args.trade else '⚠️'} Trading execution is {'ENABLED' if args.trade else 'DISABLED'}.")

    # --- 5. Add Selected Instruments to Trader ---
    successfully_added = 0
    # Iterate through the list of selected instruments (dicts)
    for instrument_data in selected_instruments:
        token = instrument_data['token']
        symbol = instrument_data['symbol']
        exchange = instrument_data['exchange']

        # Pass all three required arguments to add_trading_instrument
        if trader.add_trading_instrument(token, symbol, exchange):
            logger.info(f"➕ Successfully added instrument: Token={token}, Symbol={symbol}, Exchange={exchange}")
            successfully_added += 1
        else:
            logger.error(f"❌ Failed to add instrument: Token={token}, Symbol={symbol}")

    if successfully_added == 0:
        logger.error("❌ Failed to add any instruments to the trader. Exiting.")
        return
    elif successfully_added < len(selected_instruments):
        logger.warning(f"⚠️ Only {successfully_added}/{len(selected_instruments)} instruments were added successfully.")

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
                import time
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
