# src/utils/strike_selector.py

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
    Finds the next expiry date (Thursday) that has available instruments.
    This function now cross-references with actual NFO instruments to find a valid expiry.
    Returns the date in YYYY-MM-DD format.
    """
    if not kite_instance:
        logger.error("KiteConnect instance is required to find expiry.")
        return ""

    try:
        # Fetch all NFO instruments to get available expiries
        all_nfo_instruments = kite_instance.instruments("NFO")
        
        # --- Find the relevant symbol prefix ---
        # Use Config.SPOT_SYMBOL to derive the base name for filtering.
        # Config.SPOT_SYMBOL is 'NSE:NIFTY 50'. We need to extract 'NIFTY'.
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL # e.g., "NSE:NIFTY 50"
        
        # Remove potential exchange prefix (everything before and including ':')
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
        
        # Filter instruments for the specific index to get its expiries
        index_instruments = [inst for inst in all_nfo_instruments if inst['name'] == base_symbol_for_search]
        
        if not index_instruments:
            logger.error(f"No instruments found for index base name '{base_symbol_for_search}' on NFO (derived from Config.SPOT_SYMBOL='{spot_symbol_config}').")
            return ""

        # Get unique expiry dates for this index and sort them
        unique_expiries = set(inst['expiry'] for inst in index_instruments)
        sorted_expiries = sorted(unique_expiries)

        if not sorted_expiries:
             logger.error(f"No expiries found for index base name '{base_symbol_for_search}'.")
             return ""

        today = datetime.today().date() # Compare by date only
        
        # Find the first expiry that is today or in the future
        for expiry_date in sorted_expiries:
            # Kite returns expiry as datetime.date
            if isinstance(expiry_date, str):
                # If it's a string, parse it (shouldn't happen with Kite, but safe)
                try:
                    expiry_date_obj = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Could not parse expiry date string: {expiry_date}")
                    continue
            else:
                # It's already a datetime.date object
                expiry_date_obj = expiry_date

            if expiry_date_obj >= today:
                logger.debug(f"Selected expiry date: {expiry_date_obj.strftime('%Y-%m-%d')}")
                return expiry_date_obj.strftime('%Y-%m-%d')

        # If no future expiry found (unlikely if instruments exist)
        logger.warning("No future expiry found for the given symbol.")
        # Return the latest expiry if no future one is found
        latest_expiry = sorted_expiries[-1]
        return latest_expiry.strftime('%Y-%m-%d') if hasattr(latest_expiry, 'strftime') else str(latest_expiry)

    except Exception as e:
        logger.error(f"Error in get_next_expiry_date: {e}", exc_info=True)
        return ""


def _format_expiry_for_symbol(expiry_str: str) -> str:
    """
    Formats a YYYY-MM-DD expiry string into the NSE option symbol format (YYMONDD).
    E.g., '2025-08-07' -> '25AUG07'
    """
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        # Format: YYMONDD
        return expiry_date.strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"Error formatting expiry date '{expiry_str}': {e}")
        return ""

def get_instrument_tokens(symbol: str = "NIFTY", offset: int = 0, kite_instance: KiteConnect = None):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the *nearest valid* expiry.
    Offset allows getting ITM/OTM strikes.
    Note: This function uses Config.SPOT_SYMBOL for fetching the spot price LTP symbol
    and derives the correct base symbol name for filtering NFO instruments.
    """
    if not kite_instance:
        logger.error("KiteConnect instance is required.")
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
        logger.debug(f"Derived base symbol for NFO search: '{base_symbol_for_search}' from Config.SPOT_SYMBOL: '{spot_symbol_config}'")
        # --- End Derivation ---

        # Fetch instruments
        instruments = kite_instance.instruments("NFO")
        spot_instruments = kite_instance.instruments("NSE") # For spot price (if needed for token)

        # 1. Get Spot Price using the confirmed correct symbol via helper
        spot_symbol_ltp = _get_spot_ltp_symbol() # This will return Config.SPOT_SYMBOL (e.g., "NSE:NIFTY 50")
        logger.debug(f"[get_instrument_tokens] Fetching spot price using LTP symbol: {spot_symbol_ltp}")
        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
             logger.error(f"[get_instrument_tokens] Failed to fetch spot price for LTP symbol '{spot_symbol_ltp}'. Check if symbol is correct and market is open.")
             return None
        else:
             logger.info(f"[get_instrument_tokens] Spot price fetched: {spot_price} using {spot_symbol_ltp}")

        # 2. Calculate Strike
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        # Get the nearest valid expiry date
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)
        
        if not expiry_yyyy_mm_dd:
            logger.error("Could not determine a valid expiry date.")
            return None
            
        # Format expiry for symbol construction (YYMONDD)
        expiry_for_symbol = _format_expiry_for_symbol(expiry_yyyy_mm_dd)

        if not expiry_for_symbol:
            logger.error("Could not format expiry date for symbol construction.")
            return None

        logger.debug(f"Searching for {spot_symbol_config} options. Spot Price: {spot_price}, Calculated ATM Strike: {atm_strike}, Expiry (YYYY-MM-DD): {expiry_yyyy_mm_dd}, Expiry (Symbol Format): {expiry_for_symbol}")

        # 3. Find Symbols and Tokens
        # --- Crucial Change: Filter instruments for the specific index first ---
        # Use the correctly derived base_symbol_for_search
        nifty_index_instruments = [inst for inst in instruments if inst['name'] == base_symbol_for_search]
        logger.debug(f"Found {len(nifty_index_instruments)} instruments for base symbol '{base_symbol_for_search}' on NFO.")
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
        #           logger.debug(f"Found spot instrument token: {spot_token} for symbol: {inst['tradingsymbol']}")
        #           break
        # If you need the spot token for historical data, uncomment above. For LTP, token isn't needed.

        # Find Option Tokens using the correctly formatted symbol
        # Construct the expected full trading symbols using base_symbol_for_search
        expected_ce_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}CE"
        expected_pe_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}PE"
        logger.debug(f"Constructed CE symbol to find: {expected_ce_symbol}")
        logger.debug(f"Constructed PE symbol to find: {expected_pe_symbol}")

        # Iterate only through the filtered list of instruments for this index
        for inst in nifty_index_instruments:
            # Check expiry match (Kite returns datetime.date)
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str == expiry_yyyy_mm_dd:
                # Match the full trading symbol exactly
                if inst['tradingsymbol'] == expected_ce_symbol:
                    ce_symbol = inst['tradingsymbol']
                    ce_token = inst['instrument_token']
                    logger.debug(f"✅ Found CE Token: {ce_token} for {ce_symbol}")
                elif inst['tradingsymbol'] == expected_pe_symbol:
                    pe_symbol = inst['tradingsymbol']
                    pe_token = inst['instrument_token']
                    logger.debug(f"✅ Found PE Token: {pe_token} for {pe_symbol}")

        if not ce_token and not pe_token:
            logger.warning(f"❌ Could not find tokens for {spot_symbol_config} Expiry:{expiry_yyyy_mm_dd} Strike:{atm_strike} CE/PE. Expected Symbols: {expected_ce_symbol}, {expected_pe_symbol}")

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
        logger.error(f"Error in get_instrument_tokens: {e}", exc_info=True)
        return None
