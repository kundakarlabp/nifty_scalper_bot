# src/utils/strike_selector.py
"""
Optimized utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

This version includes robust symbol resolution, multiple fallback strategies,
and enhanced diagnostics.
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import calendar # Import calendar for week calculation

logger = logging.getLogger(__name__)

# --- Helper function to get the correct LTP symbol ---
def _get_spot_ltp_symbol():
    """Determines the correct symbol string for fetching Nifty 50 LTP."""
    from src.config import Config
    return Config.SPOT_SYMBOL # Should be "NSE:NIFTY 50"
# --- End Helper ---

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    """
    Finds the next expiry date (Thursday) that has available instruments.
    Cross-references with actual NFO instruments.
    Returns the date in YYYY-MM-DD format.
    """
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return ""

    try:
        all_nfo_instruments = kite_instance.instruments("NFO")
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL

        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
        base_symbol_for_search = potential_name_with_spaces.split()[0]

        logger.debug(f"[get_next_expiry_date] Searching for base symbol: '{base_symbol_for_search}'")

        index_instruments = [inst for inst in all_nfo_instruments if inst['name'] == base_symbol_for_search]

        if not index_instruments:
            logger.error(f"[get_next_expiry_date] No instruments found for '{base_symbol_for_search}'.")
            return ""

        unique_expiries = set(inst['expiry'] for inst in index_instruments)
        sorted_expiries = sorted(unique_expiries)
        logger.debug(f"[get_next_expiry_date] Found {len(sorted_expiries)} unique expiries.")

        if not sorted_expiries:
             logger.error(f"[get_next_expiry_date] No expiries found for '{base_symbol_for_search}'.")
             return ""

        today = datetime.today().date()
        logger.debug(f"[get_next_expiry_date] Today's date: {today}")

        for expiry_date in sorted_expiries:
            if isinstance(expiry_date, str):
                try:
                    expiry_date_obj = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"[get_next_expiry_date] Could not parse expiry: {expiry_date}")
                    continue
            else:
                expiry_date_obj = expiry_date

            logger.debug(f"[get_next_expiry_date] Checking expiry: {expiry_date_obj}")
            if expiry_date_obj >= today:
                selected_expiry_str = expiry_date_obj.strftime('%Y-%m-%d')
                logger.info(f"[get_next_expiry_date] ✅ Selected expiry: {selected_expiry_str}")
                return selected_expiry_str

        logger.warning("[get_next_expiry_date] No future expiry found.")
        latest_expiry = sorted_expiries[-1]
        latest_expiry_str = latest_expiry.strftime('%Y-%m-%d') if hasattr(latest_expiry, 'strftime') else str(latest_expiry)
        logger.info(f"[get_next_expiry_date] Fallback to latest expiry: {latest_expiry_str}")
        return latest_expiry_str

    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return ""

# --- Optimized Symbol Formatting and Resolution ---

def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """
    Primary format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')
    This is the confirmed correct format for Nifty 50 weekly options.
    """
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"[_format_expiry_for_symbol_primary] Error: {e}")
        return ""

def _construct_symbol_variants(base_symbol: str, expiry_str: str, strike: int, opt_type: str) -> list:
    """
    Generate a list of possible symbol formats to try.
    Prioritizes the known correct format.
    """
    variants = []
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        # 1. Primary Known Format: YYMONDD (e.g., NIFTY25AUG0724550CE)
        primary_format = _format_expiry_for_symbol_primary(expiry_str)
        if primary_format:
            variants.append(f"{base_symbol}{primary_format}{strike}{opt_type}")

        # 2. Alternative: YYMON (e.g., NIFTY25AUG24550CE) - Less likely for weekly, but possible for monthly or if Kite has inconsistencies
        alt_format_1 = expiry_date.strftime("%y%b").upper()
        variants.append(f"{base_symbol}{alt_format_1}{strike}{opt_type}")

        # 3. Numerical Format: YYMMDD (e.g., NIFTY25080724550CE) - Was previously incorrect, kept as low prio fallback
        # num_format = expiry_date.strftime("%y%m%d")
        # variants.append(f"{base_symbol}{num_format}{strike}{opt_type}")

        # Remove duplicates
        unique_variants = list(set(variants))
        logger.debug(f"[_construct_symbol_variants] Generated variants for {base_symbol} {expiry_str} {strike}{opt_type}: {unique_variants}")
        return unique_variants

    except Exception as e:
        logger.error(f"[_construct_symbol_variants] Error generating variants: {e}")
        # Fallback to primary format only
        primary_format = _format_expiry_for_symbol_primary(expiry_str)
        if primary_format:
             return [f"{base_symbol}{primary_format}{strike}{opt_type}]
        return []

def _fuzzy_find_instrument(instruments_list: list, base_symbol: str, expiry_yyyy_mm_dd: str, strike: int, opt_type: str) -> dict:
    """
    Fuzzy search for an instrument if direct symbol matching fails.
    Matches based on name, expiry, strike, and instrument_type.
    """
    logger.debug(f"[_fuzzy_find_instrument] Fuzzy searching for {base_symbol} Exp:{expiry_yyyy_mm_dd} Strike:{strike} Type:{opt_type}")
    try:
        for inst in instruments_list:
            # Check base symbol/name match
            if inst.get('name') != base_symbol:
                continue

            # Check expiry match
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str != expiry_yyyy_mm_dd:
                continue

            # Check instrument type (CE/PE)
            if inst.get('instrument_type') != opt_type:
                continue

            # Check strike match (allow for float/int comparison)
            inst_strike = inst.get('strike')
            if inst_strike is not None:
                try:
                    if int(float(inst_strike)) == int(strike):
                        logger.info(f"[_fuzzy_find_instrument] ✅ Found via fuzzy search: {inst['tradingsymbol']} ({inst['instrument_token']})")
                        return {
                            "symbol": inst['tradingsymbol'],
                            "token": inst['instrument_token'],
                            "strike": int(strike),
                            "type": opt_type,
                            "expiry": expiry_yyyy_mm_dd
                        }
                except (ValueError, TypeError):
                    continue # Skip if strike comparison fails
            # If strike is not available in the instrument data, we cannot fuzzy match reliably
            # In this case, symbol matching is the only reliable way.

        logger.debug(f"[_fuzzy_find_instrument] ❌ No fuzzy match found.")
        return {}
    except Exception as e:
        logger.error(f"[_fuzzy_find_instrument] Error during fuzzy search: {e}")
        return {}

# --- Main Instrument Token Retrieval Function ---

def get_instrument_tokens(
    symbol: str = "NIFTY",
    offset: int = 0,
    kite_instance: KiteConnect = None,
    cached_nfo_instruments: list = None,
    cached_nse_instruments: list = None
):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the *nearest valid* expiry.
    Implements robust symbol resolution with multiple fallbacks.
    """
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None

    if cached_nfo_instruments is None or cached_nse_instruments is None:
        logger.error("[get_instrument_tokens] Cached instrument lists are required.")
        return None

    try:
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL

        # Derive base symbol name for filtering (e.g., 'NIFTY' from 'NSE:NIFTY 50')
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
        base_symbol_for_search = potential_name_with_spaces.split()[0]
        logger.debug(f"[get_instrument_tokens] Base symbol for search: '{base_symbol_for_search}'")

        # 1. Get Spot Price
        spot_symbol_ltp = _get_spot_ltp_symbol()
        logger.debug(f"[get_instrument_tokens] Fetching spot LTP for: {spot_symbol_ltp}")
        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
             logger.error(f"[get_instrument_tokens] Failed to fetch spot price for '{spot_symbol_ltp}'.")
             return None
        else:
             logger.info(f"[get_instrument_tokens] Spot price fetched: {spot_price}")

        # 2. Calculate Strike and Expiry
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)

        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine expiry date.")
            return None

        logger.info(f"[get_instrument_tokens] Target - Expiry: {expiry_yyyy_mm_dd}, ATM Strike: {atm_strike}")

        # 3. Prepare for Symbol Search
        # Filter NFO instruments for the specific index (e.g., 'NIFTY')
        nifty_index_instruments = [inst for inst in cached_nfo_instruments if inst['name'] == base_symbol_for_search]
        logger.debug(f"[get_instrument_tokens] Filtered {len(nifty_index_instruments)} '{base_symbol_for_search}' instruments.")

        # --- Diagnostic: Show available expiries for this index ---
        if logger.isEnabledFor(logging.DEBUG):
            try:
                available_expiries = set()
                for inst in nifty_index_instruments:
                    inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
                    available_expiries.add(inst_expiry_str)
                sorted_expiries = sorted(list(available_expiries))
                logger.debug(f"[DIAGNOSTIC] Available expiries for '{base_symbol_for_search}': {sorted_expiries}")
                if expiry_yyyy_mm_dd not in sorted_expiries:
                     logger.debug(f"[DIAGNOSTIC] Target expiry '{expiry_yyyy_mm_dd}' NOT found in available expiries.")
                else:
                     logger.debug(f"[DIAGNOSTIC] Target expiry '{expiry_yyyy_mm_dd}' IS present in available expiries.")
            except Exception as diag_e:
                logger.debug(f"[DIAGNOSTIC] Error during expiry diagnostic: {diag_e}")
        # --- End Diagnostic ---

        results = {
            "spot_price": spot_price,
            "atm_strike": atm_strike,
            "expiry": expiry_yyyy_mm_dd,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None,
            "spot_token": None # Add if needed
        }

        # 4. Attempt to Find Tokens for CE and PE
        for opt_type in ['CE', 'PE']:
            target_strike = atm_strike
            logger.debug(f"[get_instrument_tokens] Searching for {opt_type} with strike {target_strike}...")

            # a. Construct Symbol Variants
            symbol_variants = _construct_symbol_variants(base_symbol_for_search, expiry_yyyy_mm_dd, target_strike, opt_type)
            found_by_variant = False

            # b. Try Direct Symbol Match with Variants
            for variant in symbol_variants:
                logger.debug(f"[get_instrument_tokens] Trying symbol variant: {variant}")
                for inst in nifty_index_instruments:
                    inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
                    if inst_expiry_str == expiry_yyyy_mm_dd and inst['tradingsymbol'] == variant:
                        logger.info(f"[get_instrument_tokens] ✅ Found {opt_type} via symbol match: {variant} ({inst['instrument_token']})")
                        if opt_type == 'CE':
                            results['ce_symbol'] = variant
                            results['ce_token'] = inst['instrument_token']
                        elif opt_type == 'PE':
                            results['pe_symbol'] = variant
                            results['pe_token'] = inst['instrument_token']
                        found_by_variant = True
                        break # Found for this option type
                if found_by_variant:
                    break # Stop trying variants for this option type

            # c. Fallback: Fuzzy Search (if direct match failed)
            if not found_by_variant:
                logger.debug(f"[get_instrument_tokens] Direct symbol match failed for {opt_type}. Trying fuzzy search...")
                fuzzy_result = _fuzzy_find_instrument(nifty_index_instruments, base_symbol_for_search, expiry_yyyy_mm_dd, target_strike, opt_type)
                if fuzzy_result:
                    # Assign fuzzy result
                    if opt_type == 'CE':
                        results['ce_symbol'] = fuzzy_result['symbol']
                        results['ce_token'] = fuzzy_result['token']
                    elif opt_type == 'PE':
                        results['pe_symbol'] = fuzzy_result['symbol']
                        results['pe_token'] = fuzzy_result['token']
                else:
                     logger.warning(f"[get_instrument_tokens] ❌ Could not find {opt_type} for Strike:{target_strike} Expiry:{expiry_yyyy_mm_dd} using any method.")


        # Check if we found at least one token
        if not results['ce_token'] and not results['pe_token']:
            logger.warning(f"[get_instrument_tokens] ❌ Failed to find tokens for any strike on expiry {expiry_yyyy_mm_dd}.")
            # Optional: Return results anyway, caller decides if partial is ok
            # Or return None if both are critical
            # return None # Uncomment if both CE & PE are mandatory

        return results

    except Exception as e:
        logger.error(f"[get_instrument_tokens] Unexpected error: {e}", exc_info=True)
        return None

# --- End of Optimized Script ---
