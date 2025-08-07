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

        # 2. Alternative: YYMON (e.g., NIFTY25AUG24550CE) - Less likely for weekly, but possible for monthly
        alt_format_1 = expiry_date.strftime("%y%b").upper()
        variants.append(f"{base_symbol}{alt_format_1}{strike}{opt_type}")

        # 3. Weekly format: YYMONGW (e.g., NIFTY25AUG1W24550CE for weekly expiries)
        week_num = (expiry_date.day - 1) // 7 + 1
        weekly_format = f"{expiry_date.strftime('%y%b').upper()}{week_num}W"
        variants.append(f"{base_symbol}{weekly_format}{strike}{opt_type}")

        # 4. Numerical Format: YYMMDD (e.g., NIFTY25080724550CE) - Low priority fallback
        num_format = expiry_date.strftime("%y%m%d")
        variants.append(f"{base_symbol}{num_format}{strike}{opt_type}")

        # Remove duplicates
        unique_variants = list(set(variants))
        logger.debug(f"[_construct_symbol_variants] Generated variants for {base_symbol} {expiry_str} {strike}{opt_type}: {unique_variants}")
        return unique_variants

    except Exception as e:
        logger.error(f"[_construct_symbol_variants] Error generating variants: {e}")
        # Fallback to primary format only - FIXED SYNTAX ERROR
        primary_format = _format_expiry_for_symbol_primary(expiry_str)
        if primary_format:
             return [f"{base_symbol}{primary_format}{strike}{opt_type}"]
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
    cached_nse_instruments: list = None,
    strike_range: int = 3  # Try ±3 strikes (150 points) if exact not found
):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the *nearest valid* expiry.
    Implements robust symbol resolution with multiple fallbacks and strike range search.
    
    Args:
        symbol: Base symbol (default "NIFTY")
        offset: Strike offset in multiples of 50 (default 0 for ATM)
        kite_instance: KiteConnect instance
        cached_nfo_instruments: List of NFO instruments
        cached_nse_instruments: List of NSE instruments (not used currently)
        strike_range: Number of strikes to try on each side if exact not found
    """
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None

    if cached_nfo_instruments is None:
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
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
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)

        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine expiry date.")
            return None

        logger.info(f"[get_instrument_tokens] Target - Expiry: {expiry_yyyy_mm_dd}, Target Strike: {target_strike}, Spot: {spot_price}")

        # 3. Prepare for Symbol Search
        # Filter NFO instruments for the specific index (e.g., 'NIFTY')
        nifty_index_instruments = [inst for inst in cached_nfo_instruments if inst['name'] == base_symbol_for_search]
        logger.debug(f"[get_instrument_tokens] Filtered {len(nifty_index_instruments)} '{base_symbol_for_search}' instruments.")

        # Filter for target expiry
        expiry_instruments = []
        for inst in nifty_index_instruments:
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str == expiry_yyyy_mm_dd:
                expiry_instruments.append(inst)

        if not expiry_instruments:
            logger.error(f"[get_instrument_tokens] No instruments found for expiry {expiry_yyyy_mm_dd}")
            # Show available expiries for debugging
            available_expiries = set()
            for inst in nifty_index_instruments:
                exp_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
                available_expiries.add(exp_str)
            logger.info(f"Available expiries: {sorted(available_expiries)}")
            return None

        logger.info(f"[get_instrument_tokens] Found {len(expiry_instruments)} instruments for target expiry")

        # --- Diagnostic: Show available strikes for debugging ---
        if logger.isEnabledFor(logging.DEBUG):
            try:
                available_strikes = set()
                for inst in expiry_instruments:
                    if inst.get('strike'):
                        available_strikes.add(int(float(inst['strike'])))
                sorted_strikes = sorted(list(available_strikes))
                logger.debug(f"[DIAGNOSTIC] Available strikes for '{expiry_yyyy_mm_dd}': {sorted_strikes}")
                logger.debug(f"[DIAGNOSTIC] Target strike {target_strike} {'IS' if target_strike in sorted_strikes else 'NOT'} in available strikes.")
            except Exception as diag_e:
                logger.debug(f"[DIAGNOSTIC] Error during strike diagnostic: {diag_e}")
        # --- End Diagnostic ---

        results = {
            "spot_price": spot_price,
            "target_strike": target_strike,
            "actual_strikes": {},  # Will store actual strikes found
            "expiry": expiry_yyyy_mm_dd,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None
        }

        # 4. Attempt to Find Tokens for CE and PE with Strike Range
        for opt_type in ['CE', 'PE']:
            found = False
            logger.debug(f"[get_instrument_tokens] Searching for {opt_type}...")
            
            # Try strikes in order: exact target, then expanding outward
            strike_attempts = [target_strike]
            for i in range(1, strike_range + 1):
                strike_attempts.extend([target_strike + i * 50, target_strike - i * 50])
            
            for attempt_strike in strike_attempts:
                if found:
                    break
                    
                logger.debug(f"[get_instrument_tokens] Trying {opt_type} strike: {attempt_strike}")

                # a. Try Symbol Variants
                symbol_variants = _construct_symbol_variants(base_symbol_for_search, expiry_yyyy_mm_dd, attempt_strike, opt_type)
                
                for variant in symbol_variants:
                    # Find matching instrument by symbol
                    matching_inst = next(
                        (inst for inst in expiry_instruments 
                         if inst['tradingsymbol'] == variant and inst.get('instrument_type') == opt_type), 
                        None
                    )
                    
                    if matching_inst:
                        logger.info(f"[get_instrument_tokens] ✅ Found {opt_type} via symbol match: {variant} (token: {matching_inst['instrument_token']})")
                        if opt_type == 'CE':
                            results['ce_symbol'] = variant
                            results['ce_token'] = matching_inst['instrument_token']
                            results['actual_strikes']['ce'] = attempt_strike
                        else:
                            results['pe_symbol'] = variant
                            results['pe_token'] = matching_inst['instrument_token']
                            results['actual_strikes']['pe'] = attempt_strike
                        found = True
                        break
                
                # b. Fallback: Fuzzy Search for this strike
                if not found:
                    logger.debug(f"[get_instrument_tokens] Symbol variants failed for {opt_type} strike {attempt_strike}. Trying fuzzy search...")
                    fuzzy_result = _fuzzy_find_instrument(expiry_instruments, base_symbol_for_search, expiry_yyyy_mm_dd, attempt_strike, opt_type)
                    if fuzzy_result:
                        if opt_type == 'CE':
                            results['ce_symbol'] = fuzzy_result['symbol']
                            results['ce_token'] = fuzzy_result['token']
                            results['actual_strikes']['ce'] = fuzzy_result['strike']
                        else:
                            results['pe_symbol'] = fuzzy_result['symbol']
                            results['pe_token'] = fuzzy_result['token']
                            results['actual_strikes']['pe'] = fuzzy_result['strike']
                        logger.info(f"[get_instrument_tokens] ✅ Found {opt_type} via fuzzy search: {fuzzy_result['symbol']}")
                        found = True
                        break

            if not found:
                logger.warning(f"[get_instrument_tokens] ❌ Could not find {opt_type} for any strike in range {target_strike}±{strike_range*50}")

        # Final validation and results
        success_count = sum([1 for x in [results['ce_token'], results['pe_token']] if x])
        
        if success_count == 0:
            logger.error("[get_instrument_tokens] ❌ No options found for any strike in the range")
            return None
        
        logger.info(f"[get_instrument_tokens] ✅ Successfully found {success_count}/2 option tokens")
        
        # Log actual strikes used if different from target
        if results['actual_strikes'].get('ce') != target_strike:
            logger.info(f"[get_instrument_tokens] CE strike adjusted from {target_strike} to {results['actual_strikes'].get('ce')}")
        if results['actual_strikes'].get('pe') != target_strike:
            logger.info(f"[get_instrument_tokens] PE strike adjusted from {target_strike} to {results['actual_strikes'].get('pe')}")

        return results

    except Exception as e:
        logger.error(f"[get_instrument_tokens] Unexpected error: {e}", exc_info=True)
        return None

# --- Additional Utility Functions ---

def diagnose_symbol_patterns(kite_instance: KiteConnect, target_expiry: str = None):
    """
    Diagnostic function to understand available symbol patterns
    """
    try:
        logger.info("[diagnose_symbol_patterns] Starting symbol pattern diagnosis...")
        nfo_instruments = kite_instance.instruments("NFO")
        
        # Filter NIFTY instruments
        nifty_instruments = [inst for inst in nfo_instruments if inst['name'] == 'NIFTY']
        
        if target_expiry:
            nifty_instruments = [
                inst for inst in nifty_instruments
                if (inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])) == target_expiry
            ]
        
        if not nifty_instruments:
            logger.error(f"❌ No NIFTY instruments found for expiry: {target_expiry}")
            return
        
        logger.info(f"📊 Found {len(nifty_instruments)} NIFTY instruments")
        
        # Show sample symbols
        logger.info("🔍 Sample trading symbols:")
        for i, inst in enumerate(nifty_instruments[:10]):
            logger.info(f"  {inst['tradingsymbol']} | Strike: {inst.get('strike', 'N/A')} | Type: {inst.get('instrument_type', 'N/A')}")
        
        # Analyze patterns
        ce_symbols = [inst['tradingsymbol'] for inst in nifty_instruments if inst.get('instrument_type') == 'CE']
        pe_symbols = [inst['tradingsymbol'] for inst in nifty_instruments if inst.get('instrument_type') == 'PE']
        
        logger.info(f"📈 Stats: CE Options: {len(ce_symbols)}, PE Options: {len(pe_symbols)}")
        
        if ce_symbols:
            logger.info(f"🎯 Sample CE Symbol: {ce_symbols[0]}")
        
    except Exception as e:
        logger.error(f"❌ Symbol pattern diagnosis failed: {e}")

def test_token_resolution(kite_instance: KiteConnect, test_spot_price: float = None):
    """
    Test function to validate token resolution
    """
    logger.info("🧪 Testing token resolution...")
    
    try:
        # Get cached instruments
        nfo_instruments = kite_instance.instruments("NFO")
        
        # Test the main function
        result = get_instrument_tokens(
            kite_instance=kite_instance,
            cached_nfo_instruments=nfo_instruments
        )
        
        if result:
            logger.info("✅ Token resolution test successful!")
            logger.info(f"  Spot Price: {result.get('spot_price')}")
            logger.info(f"  Target Strike: {result.get('target_strike')}")
            logger.info(f"  Expiry: {result.get('expiry')}")
            logger.info(f"  CE: {result.get('ce_symbol')} (Token: {result.get('ce_token')})")
            logger.info(f"  PE: {result.get('pe_symbol')} (Token: {result.get('pe_token')})")
            
            if result.get('actual_strikes'):
                logger.info(f"  Actual Strikes Used: {result['actual_strikes']}")
        else:
            logger.error("❌ Token resolution test failed")
            # Run diagnostic
            logger.info("🔍 Running diagnostic...")
            diagnose_symbol_patterns(kite_instance)
    
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

# --- End of Complete Script ---
