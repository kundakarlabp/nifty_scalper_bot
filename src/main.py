# src/main.py
"""
Main entry point for the Nifty 50 Scalper Bot.
Handles configuration, initialization of core components,
instrument selection, and starts the trading loop.
"""
import sys
import os
# --- Twisted Signal Handling ---
# Import the signal blocker before any Twisted imports
import twisted_signal_blocker
from twisted.internet import reactor

# Patch reactor.run to disable signal handlers by default
original_run = reactor.run

def patched_start_reactor(*args, **kwargs):
    """Patched reactor.run to prevent default signal handler installation."""
    if not reactor.running:
        # Force installSignalHandlers=False
        kwargs['installSignalHandlers'] = False
        original_run(*args, **kwargs)

reactor.run = patched_start_reactor
# --- End Twisted Signal Handling ---

# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
# Import logging.handlers for log rotation
import logging.handlers
# CRITICAL FIX: Import time module for main loop
import time
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
from src.utils.strike_selector import select_nifty_option_strikes # Corrected import

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


def select_instruments(kite_client: KiteConnect):
    """
    Selects Nifty 50 option instruments to trade based on expiry and strike logic.
    Fetches full instrument details (token, symbol, exchange) for the selected tokens.

    Args:
        kite_client (KiteConnect): An authenticated KiteConnect instance.

    Returns:
        list: A list of tuples (token, symbol, exchange) for selected instruments.
              Returns an empty list if selection fails critically.
    """
    logger.info("🔍 Starting instrument selection...")
    selected_instruments = [] # List to hold (token, symbol, exchange) tuples

    try:
        # 1. Determine Next Expiry
        next_expiry_date = get_next_weekly_expiry()
        if not next_expiry_date:
            logger.error("❌ Could not determine next weekly expiry date.")
            return []

        expiry_str = next_expiry_date.strftime('%Y-%m-%d')
        logger.info(f"🎯 Selected Expiry Date: {expiry_str}")

        # 2. Select Strikes
        logger.info("🔍 Selecting ATM Call strike...")
        atm_ce_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="CE",
            strike_criteria="ATM"
        )

        logger.info("🔍 Selecting ATM Put strike...")
        atm_pe_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="PE",
            strike_criteria="ATM"
        )

        all_selected_tokens = []
        if atm_ce_tokens:
            all_selected_tokens.extend(atm_ce_tokens)
            logger.info(f"✅ Selected ATM CE Tokens: {atm_ce_tokens}")
        else:
            logger.error("❌ Failed to select ATM CE strike.")

        if atm_pe_tokens:
            all_selected_tokens.extend(atm_pe_tokens)
            logger.info(f"✅ Selected ATM PE Tokens: {atm_pe_tokens}")
        else:
            logger.error("❌ Failed to select ATM PE strike.")

        # 3. Get full instrument details (symbol, exchange) for each token
        # Fetch instrument list once for efficiency
        if all_selected_tokens:
            logger.info("🔍 Fetching instrument details from Kite...")
            # Fetch all NFO instruments (or filter more specifically if needed)
            nfo_instruments = kite_client.instruments(exchange="NFO")
            # Create a map for quick lookup
            instrument_map = {inst['instrument_token']: inst for inst in nfo_instruments}

            for token in all_selected_tokens:
                instrument_data = instrument_map.get(token)
                if instrument_data:
                    selected_instruments.append((
                        instrument_data['instrument_token'],
                        instrument_data['tradingsymbol'],
                        instrument_data['exchange']
                    ))
                    logger.info(f"📄 Added Instrument - Token: {instrument_data['instrument_token']}, Symbol: {instrument_data['tradingsymbol']}, Exchange: {instrument_data['exchange']}")
                else:
                    logger.warning(f"⚠️ Details for token {token} not found in instrument list. Skipping.")

    except Exception as e:
        logger.error(f"❌ Unexpected error during instrument selection: {e}", exc_info=True)
        # Depending on strategy, you might want to fail here or continue with partial list
        # For strict live trading, failing might be safer.

    # 4. Final Check & Fallback/Exit Logic
    if not selected_instruments:
        logger.error("❌ Instrument selection failed - no valid instruments found. Exiting application.")
        # Returning empty list signals failure to the main function
        return []
    else:
        logger.info(f"✅ Final selected instruments: {selected_instruments}")

    return selected_instruments


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
        logger.critical("❌ Zerodha API credentials (ZERODHA_API_KEY, KITE_ACCESS_TOKEN) missing in config.")
        sys.exit(1) # Standard exit code for error

    kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
    try:
        kite.set_access_token(Config.KITE_ACCESS_TOKEN)
        # Optional: Verify connection with a simple API call
        # profile = kite.profile()
        logger.info("✅ Zerodha Kite Connect client initialized and authenticated.")
    except Exception as e:
        logger.critical(f"❌ Failed to authenticate Kite client: {e}")
        sys.exit(1) # Standard exit code for error

    # --- 2. Select Instruments ---
    # This now returns a list of tuples (token, symbol, exchange)
    selected_instruments = select_instruments(kite)

    if not selected_instruments:
        logger.critical("❌ No instruments available to trade. Exiting.")
        sys.exit(1) # Standard exit code for error

    # --- 3. Initialize Order Executor ---
    order_executor = OrderExecutor(kite=kite)
    logger.info("✅ OrderExecutor initialized.")

    # --- 4. Initialize RealTime Trader ---
    trader = RealTimeTrader(order_executor=order_executor)
    trader.enable_trading(enable=args.trade)
    logger.info(f"{'✅' if args.trade else '⚠️'} Trading execution is {'ENABLED' if args.trade else 'DISABLED'}.")

    # --- 5. Add Selected Instruments to Trader ---
    successfully_added = 0
    for token, symbol, exchange in selected_instruments:
        if trader.add_trading_instrument(token, symbol, exchange):
            logger.info(f"➕ Successfully added instrument: {symbol} ({token}) on {exchange}")
            successfully_added += 1
        else:
            logger.error(f"❌ Failed to add instrument: {symbol} ({token})")

    if successfully_added == 0:
        logger.critical("❌ Failed to add any instruments to the trader. Exiting.")
        trader.stop_trading() # Ensure clean shutdown if startup fails
        sys.exit(1) # Standard exit code for error
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
                # Main loop checks trader status
                while getattr(trader, 'is_trading', False): # Safer attribute access
                    time.sleep(1) # Check every second
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard Interrupt received. Stopping trader...")
            except Exception as e:
                logger.error(f"❌ Unexpected error in main loop: {e}", exc_info=True)
            finally:
                # Ensure trader is stopped on exit
                if trader:
                    trader.stop_trading()
                    logger.info("🛑 Trader stopped. Bot shutdown complete.")
        else:
            logger.critical("❌ Failed to start RealTimeTrader.")
            sys.exit(1) # Standard exit code for error
    else:
        logger.info("🔄 Signal generation mode selected. Implementation needed.")
        # Add signal generation logic here if required

    sys.exit(0) # Standard exit code for success

if __name__ == "__main__":
    main()
