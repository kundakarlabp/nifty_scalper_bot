# src/main.py
"""
Main entry point for the Nifty 50 Scalper Bot.
Handles configuration, initialization of core components,
instrument selection, and starts the trading loop.
Optionally starts the Telegram command listener.
"""
import sys
import os
# Ensure correct path resolution for imports
# This might be handled by your project structure or Docker setup
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
# Import the OrderExecutor
from src.execution.order_executor import OrderExecutor
# Import the Telegram Command Listener
from src.notifications.telegram_command_listener import TelegramCommandListener
# Import utility functions for instrument selection
from src.utils.expiry_selector import get_next_weekly_expiry
from src.utils.strike_selector import select_nifty_option_strikes

# Import threading to run the Telegram listener in the background
import threading


# --- Setup Enhanced Logging with Rotation ---
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
        logger.info("🔍 Selecting ATM Call strike...")
        # Assume select_nifty_option_strikes returns a list of tokens
        atm_ce_tokens = select_nifty_option_strikes(
            kite=kite_client,
            expiry=expiry_str,
            option_type="CE",
            strike_criteria="ATM"
        )
        if atm_ce_tokens:
            # Fetch full instrument details for CE tokens
            try:
                # Efficiently fetch details for selected tokens
                # Get all NFO instruments once
                all_nfo_instruments = kite_client.instruments("NFO")
                instrument_map = {inst['instrument_token']: inst for inst in all_nfo_instruments}
                
                for token in atm_ce_tokens:
                    inst_data = instrument_map.get(token)
                    if inst_data:
                        selected_instruments.append({
                            'token': token,
                            'symbol': inst_data['tradingsymbol'],
                            'exchange': inst_data['exchange']
                        })
                        logger.info(f"✅ Selected ATM CE: Token={token}, Symbol={inst_data['tradingsymbol']}")
                    else:
                        logger.warning(f"⚠️ Could not find details for CE token {token}")
            except Exception as e:
                logger.error(f"❌ Error fetching CE instrument details: {e}", exc_info=True)
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
             # Fetch full instrument details for PE tokens
            try:
                # Re-use the instrument_map if it was successfully created for CE
                # If not, fetch it again (less efficient, but robust)
                if 'all_nfo_instruments' not in locals() or 'instrument_map' not in locals():
                     all_nfo_instruments = kite_client.instruments("NFO")
                     instrument_map = {inst['instrument_token']: inst for inst in all_nfo_instruments}

                for token in atm_pe_tokens:
                    inst_data = instrument_map.get(token)
                    if inst_data:
                        selected_instruments.append({
                            'token': token,
                            'symbol': inst_data['tradingsymbol'],
                            'exchange': inst_data['exchange']
                        })
                        logger.info(f"✅ Selected ATM PE: Token={token}, Symbol={inst_data['tradingsymbol']}")
                    else:
                         logger.warning(f"⚠️ Could not find details for PE token {token}")
            except Exception as e:
                logger.error(f"❌ Error fetching PE instrument details: {e}", exc_info=True)
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
    parser = argparse.ArgumentParser(
        description="Nifty Scalper Bot",
        epilog="Example: python src/main.py --mode realtime --trade --telegram"
    )
    parser.add_argument('--mode', choices=['realtime', 'signal'], default='realtime',
                        help='Operational mode (default: realtime)')
    parser.add_argument('--trade', action='store_true',
                        help='Enable live trade execution (requires valid Kite credentials)')
    parser.add_argument('--telegram', action='store_true',
                        help='Enable Telegram command listener (requires valid Telegram credentials)')
    args = parser.parse_args()

    logger.info("🚀 Initializing Nifty 50 Scalper Bot...")
    logger.info(f"🔁 Mode: {args.mode}, Trade Execution: {'ENABLED' if args.trade else 'DISABLED'}, Telegram Listener: {'ENABLED' if args.telegram else 'DISABLED'}")

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

    # --- 6. Initialize and Start Telegram Command Listener (Optional) ---
    telegram_listener_thread = None
    if args.telegram:
        if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_USER_ID:
            logger.error("❌ Telegram credentials missing. Cannot start listener.")
            # Optionally, continue without Telegram
            # Or exit: return
        else:
            try:
                # Create the Telegram listener instance, passing the *main* trader instance
                telegram_listener = TelegramCommandListener(
                    bot_token=Config.TELEGRAM_BOT_TOKEN,
                    chat_id=str(Config.TELEGRAM_USER_ID), # Ensure chat_id is a string
                    trader_instance=trader # Pass the main trader
                )
                # Start the listener in a separate thread so the main thread can run the trader
                telegram_listener_thread = threading.Thread(target=telegram_listener.start_listening, daemon=True)
                telegram_listener_thread.start()
                logger.info("✅ Telegram Command Listener started in background thread.")
            except Exception as e:
                logger.error(f"❌ Failed to start Telegram Command Listener: {e}", exc_info=True)
                # Optionally, continue without Telegram
                # Or exit: return
    else:
        logger.info("ℹ️ Telegram Command Listener is disabled (--telegram flag not set).")

    # --- 7. Start Trading Engine ---
    if args.mode == 'realtime':
        logger.info("🚀 Starting RealTime Trading Engine...")
        if trader.start_trading():
            logger.info("✅ RealTimeTrader started successfully.")
            logger.info("⏳ Bot is now running. Press Ctrl+C to stop.")
            # --- 8. Keep the main thread alive ---
            try:
                # Main loop: The trader runs its streaming/processing in background threads
                # The Telegram listener (if enabled) also runs in its own thread
                # This main thread can monitor or wait.
                while trader.is_trading: # Check a flag that stop_trading sets to False
                    # You could add periodic status checks or other main-loop tasks here
                    # For now, just sleep to prevent a busy loop
                    time.sleep(1) # Check every second
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard Interrupt received. Stopping trader...")
            finally:
                # Signal the trader to stop
                trader.stop_trading()
                # Signal the Telegram listener to stop (if it was started)
                if args.telegram and 'telegram_listener' in locals():
                    telegram_listener.stop_listening()
                    # Optionally wait for the thread to finish
                    # if telegram_listener_thread and telegram_listener_thread.is_alive():
                    #     telegram_listener_thread.join(timeout=5)
                logger.info("🛑 Trader and Listener stopped. Bot shutdown complete.")
        else:
            logger.error("❌ Failed to start RealTimeTrader.")
    else:
        logger.info("🔄 Signal generation mode is selected but not implemented in this script.")


if __name__ == "__main__":
    main()
