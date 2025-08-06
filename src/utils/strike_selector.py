# src/utils/strike_selector.py
"""
Utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# --- Helper function to get the correct LTP symbol ---
def _get_spot_ltp_symbol():
    """Determines the correct symbol string for fetching Nifty 50 LTP."""
    # Fetch the confirmed correct symbol directly from Config
    from src.config import Config
    return Config.SPOT_SYMBOL # This will now be "NSE:NIFTY 50" from .env
# --- End Helper ---

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    """
    Finds the next expiry date that has available instruments.
    This function now cross-references with actual NFO instruments to find a valid expiry.
    Returns the date in YYYY-MM-DD format.
    """
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required to find expiry.")
        return ""
    
    try:
        # Fetch all NFO instruments to get available expiries
        all_nfo_instruments = kite_instance.instruments("NFO")
        
        # --- Find the relevant symbol prefix ---
        # Use Config.SPOT_SYMBOL to derive the base name for filtering.
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL # e.g., "NSE:NIFTY 50"

        # Derive base symbol name for filtering (as corrected previously)
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1] # Get part after first ':'
        else:
            potential_name_with_spaces = spot_symbol_config

        # Now, get the base name (e.g., 'NIFTY' from 'NIFTY 50')
        base_symbol_for_search = potential_name_with_spaces.split()[0]
        # Examples:
        # "NSE:NIFTY 50" -> "NIFTY 50" -> "NIFTY"
        # "NIFTY 50" -> "NIFTY"
        # "BANKNIFTY" -> "BANKNIFTY"
        # "NSE:BANKNIFTY" -> "BANKNIFTY"
        logger.debug(f"[get_next_expiry_date] Derived base symbol for NFO search: '{base_symbol_for_search}' from Config.SPOT_SYMBOL: '{spot_symbol_config}'")

        # Filter instruments for the specific index to get its expiries
        logger.debug(f"[get_next_expiry_date] Filtering instruments for name='{base_symbol_for_search}'...")
        index_instruments = [inst for inst in all_nfo_instruments if inst['name'] == base_symbol_for_search]

        if not index_instruments:
            logger.error(f"[get_next_expiry_date] No instruments found for index base name '{base_symbol_for_search}' on NFO (derived from Config.SPOT_SYMBOL='{spot_symbol_config}').")
            return ""

        # Get unique expiry dates for this index and sort them
        unique_expiries = set(inst['expiry'] for inst in index_instruments)
        sorted_expiries = sorted(unique_expiries)
        logger.debug(f"[get_next_expiry_date] Found {len(sorted_expiries)} unique expiries for '{base_symbol_for_search}'.")

        if not sorted_expiries:
             logger.error(f"[get_next_expiry_date] No expiries found for index base name '{base_symbol_for_search}'.")
             return ""

        today = datetime.today().date() # Compare by date only
        logger.debug(f"[get_next_expiry_date] Today's date: {today}")

        # Find the first expiry that is today or in the future
        for expiry_date in sorted_expiries:
            # Kite returns expiry as datetime.date or sometimes string
            if isinstance(expiry_date, str):
                try:
                    expiry_date_obj = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"[get_next_expiry_date] Could not parse expiry date string: {expiry_date}")
                    continue
            else:
                # It's already a datetime.date object
                expiry_date_obj = expiry_date

            logger.debug(f"[get_next_expiry_date] Checking expiry date: {expiry_date_obj} (Type: {type(expiry_date_obj)})")
            if expiry_date_obj >= today:
                selected_expiry_str = expiry_date_obj.strftime('%Y-%m-%d')
                logger.info(f"[get_next_expiry_date] ✅ Selected expiry date: {selected_expiry_str}")
                return selected_expiry_str

        # If no future expiry found (should be rare if instruments exist)
        logger.warning("[get_next_expiry_date] No future expiry found for the given symbol based on Kite data.")
        # Return the latest expiry if no future one is found
        latest_expiry = sorted_expiries[-1]
        latest_expiry_str = latest_expiry.strftime('%Y-%m-%d') if hasattr(latest_expiry, 'strftime') else str(latest_expiry)
        logger.info(f"[get_next_expiry_date] Fallback: Returning latest expiry: {latest_expiry_str}")
        return latest_expiry_str

    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return ""


def _format_expiry_for_symbol(expiry_str: str) -> str:
    """
    Formats a YYYY-MM-DD expiry string into the NSE weekly option symbol format (YYMONDD).
    E.g., '2025-08-07' -> '25AUG07'
    This format is confirmed correct by Zerodha's instrument example: NIFTY25AUG0724650CE
    """
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        # Format: YYMONDD (Confirmed correct format for weekly index options)
        return expiry_date.strftime("%y%b%d").upper() # This produces YYMONDD (e.g., 25AUG07)
    except ValueError as e:
        logger.error(f"[_format_expiry_for_symbol] Error formatting expiry date '{expiry_str}': {e}")
        return ""

def get_instrument_tokens(
    symbol: str = "NIFTY",
    offset: int = 0,
    kite_instance: KiteConnect = None,
    cached_nfo_instruments: list = None, # New parameter for cached NFO data
    cached_nse_instruments: list = None  # New parameter for cached NSE data
):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the *nearest valid* expiry.
    Offset allows getting ITM/OTM strikes.
    Note: This function uses Config.SPOT_SYMBOL for fetching the spot price LTP symbol
    and derives the correct base symbol name for filtering NFO instruments.
    It requires cached instrument lists to avoid rate limits.
    """
    # Validate inputs
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None
        
    if cached_nfo_instruments is None or cached_nse_instruments is None:
        logger.error("[get_instrument_tokens] Cached NFO and NSE instrument lists are required to prevent rate limits.")
        # Returning None here is critical to prevent the calling function from proceeding with invalid data
        # which would likely lead to errors or incorrect behavior.
        return None

    try:
        # Import Config here to get the latest SPOT_SYMBOL
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL # e.g., "NSE:NIFTY 50"

        # --- Derive the correct base symbol name for filtering NFO instruments ---
        # This is crucial for matching the 'name' field in Kite instrument data.
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1] # Get part after first ':'
        else:
            potential_name_with_spaces = spot_symbol_config
        base_symbol_for_search = potential_name_with_spaces.split()[0]
        # This should correctly resolve to 'NIFTY' for 'NSE:NIFTY 50'
        logger.debug(f"[get_instrument_tokens] Derived base symbol for NFO search: '{base_symbol_for_search}' from Config.SPOT_SYMBOL: '{spot_symbol_config}'")
        # --- End Derivation ---

        # Use cached instruments directly
        instruments = cached_nfo_instruments
        spot_instruments = cached_nse_instruments

        # 1. Get Spot Price using the confirmed correct symbol via helper
        spot_symbol_ltp = _get_spot_ltp_symbol() # This will return Config.SPOT_SYMBOL (e.g., "NSE:NIFTY 50")
        logger.debug(f"[get_instrument_tokens] Fetching spot price using LTP symbol: {spot_symbol_ltp}")
        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
             logger.error(f"[get_instrument_tokens] Failed to fetch spot price for LTP symbol '{spot_symbol_ltp}'. Check if symbol is correct and market is open.")
             return None # Return None on failure to fetch critical data
        else:
             logger.info(f"[get_instrument_tokens] Spot price fetched: {spot_price} using {spot_symbol_ltp}")

        # 2. Calculate Strike
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        # Get the nearest valid expiry date using the cached data
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)

        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine a valid expiry date.")
            return None # Return None if expiry cannot be determined

        # Format expiry for symbol construction (YYMONDD - Confirmed Correct Format)
        expiry_for_symbol = _format_expiry_for_symbol(expiry_yyyy_mm_dd)

        if not expiry_for_symbol:
            logger.error("[get_instrument_tokens] Could not format expiry date for symbol construction.")
            return None # Return None if expiry formatting fails

        logger.debug(f"[get_instrument_tokens] Searching for {spot_symbol_config} options. Spot Price: {spot_price}, Calculated ATM Strike: {atm_strike}, Expiry (YYYY-MM-DD): {expiry_yyyy_mm_dd}, Expiry (Symbol Format): {expiry_for_symbol}")

        # 3. Find Symbols and Tokens
        # --- Crucial Change: Filter instruments for the specific index first ---
        # Use the correctly derived base_symbol_for_search and the cached instruments list
        nifty_index_instruments = [inst for inst in instruments if inst['name'] == base_symbol_for_search]
        logger.debug(f"[get_instrument_tokens] Found {len(nifty_index_instruments)} instruments for base symbol '{base_symbol_for_search}' on NFO.")
        # --- End Crucial Change ---

        ce_symbol = None
        pe_symbol = None
        ce_token = None
        pe_token = None
        spot_token = None # Might be needed if you want the spot instrument token too

        # Find Spot Token (optional, if needed for historical data on spot)
        # The LTP symbol 'NSE:NIFTY 50' might not directly match an instrument symbol.
        # The instrument symbol might just be 'NIFTY 50'. Let's try to find it.
        # for inst in spot_instruments:
        #      if inst['tradingsymbol'] == potential_name_with_spaces: # e.g., 'NIFTY 50'
        #           spot_token = inst['instrument_token']
        #           logger.debug(f"[get_instrument_tokens] Found spot instrument token: {spot_token} for symbol: {inst['tradingsymbol']}")
        #           break
        # If you need the spot token for historical data, uncomment above. For LTP, token isn't needed.

        # Find Option Tokens using the correctly formatted symbol
        # Construct the expected full trading symbols using base_symbol_for_search and corrected expiry format (YYMONDD)
        expected_ce_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}CE"
        expected_pe_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}PE"
        logger.info(f"[get_instrument_tokens] Constructed CE symbol to find: {expected_ce_symbol}")
        logger.info(f"[get_instrument_tokens] Constructed PE symbol to find: {expected_pe_symbol}")

        # --- Diagnostic Logging Start ---
        # Log details of instruments that match name and expiry to help debugging
        if logger.isEnabledFor(logging.DEBUG): # Only log this extra info in DEBUG mode
            logger.debug(f"[get_instrument_tokens] [DIAGNOSTIC] Instruments for '{base_symbol_for_search}' expiring {expiry_yyyy_mm_dd}:")
            count = 0
            for inst in nifty_index_instruments:
                inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
                if inst_expiry_str == expiry_yyyy_mm_dd:
                    count += 1
                    if count <= 20: # Limit output to prevent log spam
                        logger.debug(f"  [DIAGNOSTIC] - Symbol: {inst['tradingsymbol']}, Strike: {inst.get('strike', 'N/A')}, Type: {inst.get('instrument_type', 'N/A')}")
                    elif count == 21:
                         logger.debug("  [DIAGNOSTIC] - ... (output limited)")
            if count == 0:
                logger.debug(f"  [DIAGNOSTIC] - No instruments found for '{base_symbol_for_search}' expiring {expiry_yyyy_mm_dd} in filtered list.")
            
            # --- Enhanced Diagnostic: Show Cache Stats ---
            # Provide more context about the overall cache state
            total_cached_nfo = len(instruments) if instruments else 0
            total_nifty_filtered = len(nifty_index_instruments)
            unique_expiries_in_nifty = set()
            for inst in nifty_index_instruments:
                 inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
                 unique_expiries_in_nifty.add(inst_expiry_str)
            sorted_unique_expiries = sorted(list(unique_expiries_in_nifty))
            
            logger.debug(f"  [DIAGNOSTIC] - Cache Stats: Total NFO Instruments: {total_cached_nfo}")
            logger.debug(f"  [DIAGNOSTIC] - Cache Stats: Filtered NIFTY Instruments: {total_nifty_filtered}")
            logger.debug(f"  [DIAGNOSTIC] - Cache Stats: Unique Expiries in Filtered NIFTY: {len(sorted_unique_expiries)}")
            logger.debug(f"  [DIAGNOSTIC] - Cache Stats: First 10 Unique Expiries: {sorted_unique_expiries[:10]}")
            # --- End Enhanced Diagnostic ---
            
        # --- Diagnostic Logging End ---
        
        # Iterate only through the filtered list of instruments for this index
        for inst in nifty_index_instruments:
            # Check expiry match (Kite returns datetime.date or string)
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str == expiry_yyyy_mm_dd:
                # Match the full trading symbol exactly
                if inst['tradingsymbol'] == expected_ce_symbol:
                    ce_symbol = inst['tradingsymbol']
                    ce_token = inst['instrument_token']
                    logger.info(f"[get_instrument_tokens] ✅ Found CE Token: {ce_token} for {ce_symbol}")
                elif inst['tradingsymbol'] == expected_pe_symbol:
                    pe_symbol = inst['tradingsymbol']
                    pe_token = inst['instrument_token']
                    logger.info(f"[get_instrument_tokens] ✅ Found PE Token: {pe_token} for {pe_symbol}")

        if not ce_token and not pe_token:
            logger.warning(f"[get_instrument_tokens] ❌ Could not find tokens for {spot_symbol_config} Expiry:{expiry_yyyy_mm_dd} Strike:{atm_strike} CE/PE. Expected Symbols: {expected_ce_symbol}, {expected_pe_symbol}")

        # Return the results, even if some tokens are None (e.g., if only CE or PE was found)
        # The calling function (realtime_trader.py) should check ce_token and pe_token.
        return {
            "spot_price": spot_price,
            "spot_token": spot_token, # Will be None unless logic to find it is added
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token,
            "expiry": expiry_yyyy_mm_dd # Return the standard YYYY-MM-DD expiry
        }
    except Exception as e:
        logger.error(f"[get_instrument_tokens] Error: {e}", exc_info=True)
        # Return None on any unexpected exception
        return None
