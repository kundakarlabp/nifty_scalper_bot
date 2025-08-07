# src/utils/strike_selector.py
"""
Complete utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

This version includes robust symbol resolution, multiple fallback strategies,
enhanced diagnostics, comprehensive rate limiting protection, and error-free execution.
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import calendar
import time
import threading
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple
import sys

logger = logging.getLogger(__name__)

# Global rate limiting and caching
_last_api_call = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # Minimum 500ms between API calls

def _rate_limited_api_call(func, *args, **kwargs):
    """Rate-limited API call wrapper with retry logic"""
    with _api_call_lock:
        call_key = func.__name__
        now = time.time()
        
        if call_key in _last_api_call:
            elapsed = now - _last_api_call[call_key]
            if elapsed < _MIN_API_INTERVAL:
                sleep_time = _MIN_API_INTERVAL - elapsed
                logger.debug(f"Rate limiting {call_key}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        try:
            result = func(*args, **kwargs)
            _last_api_call[call_key] = time.time()
            return result
        except Exception as e:
            if "Too many requests" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit for {call_key}, waiting 2 seconds...")
                time.sleep(2)
                # Retry once
                try:
                    result = func(*args, **kwargs)
                    _last_api_call[call_key] = time.time()
                    return result
                except Exception as retry_e:
                    logger.error(f"Retry also failed for {call_key}: {retry_e}")
                    raise retry_e
            raise

def _get_spot_ltp_symbol():
    """Determines the correct symbol string for fetching Nifty 50 LTP."""
    try:
        from src.config import Config
        return getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50')
    except ImportError:
        logger.warning("Could not import Config, using default SPOT_SYMBOL")
        return 'NSE:NIFTY 50'

def _calculate_next_thursday() -> str:
    """Calculate next Thursday as fallback expiry"""
    try:
        today = datetime.today()
        days_ahead = 3 - today.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        return next_thursday.strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"Error calculating next Thursday: {e}")
        # Ultimate fallback
        return (datetime.today() + timedelta(days=7)).strftime("%Y-%m-%d")

@lru_cache(maxsize=32)
def _get_cached_expiry_date(base_symbol: str, current_date_str: str) -> str:
    """Cached expiry date calculation to avoid repeated API calls"""
    return ""  # Will be populated by the main function

def get_next_expiry_date(kite_instance: KiteConnect, use_cache: bool = True) -> str:
    """
    Finds the next expiry date (Thursday) that has available instruments.
    Cross-references with actual NFO instruments with enhanced error handling.
    Returns the date in YYYY-MM-DD format.
    """
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return _calculate_next_thursday()

    try:
        spot_symbol_config = _get_spot_ltp_symbol()
        
        # Extract base symbol
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
        base_symbol_for_search = potential_name_with_spaces.split()[0]

        logger.debug(f"[get_next_expiry_date] Searching for base symbol: '{base_symbol_for_search}'")

        # Rate-limited instruments fetch
        try:
            all_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        except Exception as api_error:
            logger.error(f"[get_next_expiry_date] Failed to fetch NFO instruments: {api_error}")
            return _calculate_next_thursday()

        index_instruments = [inst for inst in all_nfo_instruments if inst.get('name') == base_symbol_for_search]

        if not index_instruments:
            logger.error(f"[get_next_expiry_date] No instruments found for '{base_symbol_for_search}'.")
            return _calculate_next_thursday()

        # Extract and sort expiries
        unique_expiries = set()
        for inst in index_instruments:
            expiry = inst.get('expiry')
            if expiry:
                if hasattr(expiry, 'strftime'):
                    unique_expiries.add(expiry.strftime("%Y-%m-%d"))
                else:
                    unique_expiries.add(str(expiry))

        sorted_expiries = sorted(unique_expiries)
        logger.debug(f"[get_next_expiry_date] Found {len(sorted_expiries)} unique expiries.")

        if not sorted_expiries:
            logger.error(f"[get_next_expiry_date] No expiries found for '{base_symbol_for_search}'.")
            return _calculate_next_thursday()

        today = datetime.today().date()
        logger.debug(f"[get_next_expiry_date] Today's date: {today}")

        # Find next valid expiry
        for expiry_str in sorted_expiries:
            try:
                expiry_date_obj = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                logger.debug(f"[get_next_expiry_date] Checking expiry: {expiry_date_obj}")
                
                if expiry_date_obj >= today:
                    logger.info(f"[get_next_expiry_date] ✅ Selected expiry: {expiry_str}")
                    return expiry_str
            except ValueError:
                logger.warning(f"[get_next_expiry_date] Could not parse expiry: {expiry_str}")
                continue

        # Fallback to latest available expiry
        if sorted_expiries:
            latest_expiry = sorted_expiries[-1]
            logger.info(f"[get_next_expiry_date] Fallback to latest expiry: {latest_expiry}")
            return latest_expiry

        return _calculate_next_thursday()

    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return _calculate_next_thursday()

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

def _construct_symbol_variants(base_symbol: str, expiry_str: str, strike: int, opt_type: str) -> List[str]:
    """
    Generate a list of possible symbol formats to try.
    Prioritizes the known correct format with enhanced variants.
    """
    variants = []
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        
        # 1. Primary Known Format: YYMONDD (e.g., NIFTY25AUG0724550CE)
        primary_format = _format_expiry_for_symbol_primary(expiry_str)
        if primary_format:
            variants.append(f"{base_symbol}{primary_format}{strike}{opt_type}")

        # 2. Alternative formats for different expiry types
        year_short = expiry_date.strftime("%y")
        month_short = expiry_date.strftime("%b").upper()
        day = expiry_date.strftime("%d")
        
        # Monthly format: YYMON (e.g., NIFTY25AUG24550CE)
        variants.append(f"{base_symbol}{year_short}{month_short}{strike}{opt_type}")
        
        # Weekly format with week number
        week_of_month = (expiry_date.day - 1) // 7 + 1
        variants.append(f"{base_symbol}{year_short}{month_short}{week_of_month}W{strike}{opt_type}")
        
        # Date format: YYMMDD
        date_format = expiry_date.strftime("%y%m%d")
        variants.append(f"{base_symbol}{date_format}{strike}{opt_type}")
        
        # Full month name format (less common)
        month_full = expiry_date.strftime("%B").upper()[:3]  # First 3 chars
        variants.append(f"{base_symbol}{year_short}{month_full}{day}{strike}{opt_type}")

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        logger.debug(f"[_construct_symbol_variants] Generated {len(unique_variants)} variants for {base_symbol} {expiry_str} {strike}{opt_type}")
        return unique_variants

    except Exception as e:
        logger.error(f"[_construct_symbol_variants] Error generating variants: {e}")
        # Fallback to primary format only
        primary_format = _format_expiry_for_symbol_primary(expiry_str)
        if primary_format:
            return [f"{base_symbol}{primary_format}{strike}{opt_type}"]
        return []

def _fuzzy_find_instrument(instruments_list: List[Dict], base_symbol: str, 
                          expiry_yyyy_mm_dd: str, strike: int, opt_type: str) -> Dict[str, Any]:
    """
    Enhanced fuzzy search for an instrument if direct symbol matching fails.
    Matches based on name, expiry, strike, and instrument_type with tolerance.
    """
    logger.debug(f"[_fuzzy_find_instrument] Fuzzy searching for {base_symbol} Exp:{expiry_yyyy_mm_dd} Strike:{strike} Type:{opt_type}")
    
    best_match = {}
    exact_matches = []
    
    try:
        for inst in instruments_list:
            # Check base symbol/name match
            if inst.get('name') != base_symbol:
                continue

            # Check expiry match
            inst_expiry = inst.get('expiry')
            if inst_expiry:
                if hasattr(inst_expiry, 'strftime'):
                    inst_expiry_str = inst_expiry.strftime("%Y-%m-%d")
                else:
                    inst_expiry_str = str(inst_expiry)
                    
                if inst_expiry_str != expiry_yyyy_mm_dd:
                    continue

            # Check instrument type (CE/PE)
            if inst.get('instrument_type') != opt_type:
                continue

            # Check strike match with tolerance
            inst_strike = inst.get('strike')
            if inst_strike is not None:
                try:
                    strike_diff = abs(int(float(inst_strike)) - int(strike))
                    if strike_diff == 0:  # Exact match
                        exact_matches.append({
                            "symbol": inst['tradingsymbol'],
                            "token": inst['instrument_token'],
                            "strike": int(float(inst_strike)),
                            "type": opt_type,
                            "expiry": expiry_yyyy_mm_dd,
                            "difference": strike_diff
                        })
                    elif strike_diff <= 50:  # Within one strike (50 points)
                        if not best_match or strike_diff < best_match.get('difference', float('inf')):
                            best_match = {
                                "symbol": inst['tradingsymbol'],
                                "token": inst['instrument_token'],
                                "strike": int(float(inst_strike)),
                                "type": opt_type,
                                "expiry": expiry_yyyy_mm_dd,
                                "difference": strike_diff
                            }
                except (ValueError, TypeError):
                    continue

        # Return exact match if available
        if exact_matches:
            result = exact_matches[0]
            logger.info(f"[_fuzzy_find_instrument] ✅ Exact match found: {result['symbol']} (token: {result['token']})")
            return result
        
        # Return best approximate match
        if best_match:
            logger.info(f"[_fuzzy_find_instrument] ✅ Approximate match found: {best_match['symbol']} (strike difference: {best_match['difference']})")
            return best_match

        logger.debug("[_fuzzy_find_instrument] ❌ No fuzzy match found.")
        return {}
        
    except Exception as e:
        logger.error(f"[_fuzzy_find_instrument] Error during fuzzy search: {e}")
        return {}

def get_instrument_tokens(
    symbol: str = "NIFTY",
    offset: int = 0,
    kite_instance: Optional[KiteConnect] = None,
    cached_nfo_instruments: Optional[List[Dict]] = None,
    cached_nse_instruments: Optional[List[Dict]] = None,
    strike_range: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Enhanced instrument token retrieval with comprehensive error handling and fallbacks.
    
    Args:
        symbol: Base symbol (default "NIFTY")
        offset: Strike offset in multiples of 50 (default 0 for ATM)
        kite_instance: KiteConnect instance
        cached_nfo_instruments: List of NFO instruments
        cached_nse_instruments: List of NSE instruments
        strike_range: Number of strikes to try on each side if exact not found
    
    Returns:
        Dictionary with instrument tokens and metadata, None if failed
    """
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None

    if not cached_nfo_instruments:
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
        return None

    try:
        spot_symbol_config = _get_spot_ltp_symbol()

        # Derive base symbol name
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
        base_symbol_for_search = potential_name_with_spaces.split()[0]
        
        logger.debug(f"[get_instrument_tokens] Base symbol for search: '{base_symbol_for_search}'")

        # 1. Get Spot Price with rate limiting and robust error handling
        spot_symbol_ltp = _get_spot_ltp_symbol()
        logger.debug(f"[get_instrument_tokens] Fetching spot LTP for: {spot_symbol_ltp}")
        
        spot_price = None
        try:
            ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol_ltp])
            
            # --- Start of Correction ---
            # Check if the response is a dictionary before processing
            if isinstance(ltp_data, dict):
                spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')
            else:
                # Log the unexpected response for better debugging
                logger.error(f"[get_instrument_tokens] Unexpected LTP data format. Expected dict, got {type(ltp_data)}. Data: {ltp_data}")
                spot_price = None
            # --- End of Correction ---

        except Exception as ltp_error:
            logger.error(f"[get_instrument_tokens] LTP fetch failed with exception: {ltp_error}")
            spot_price = None # Ensure spot_price is None on exception

        # Fallback logic if spot_price could not be fetched
        if not spot_price:
            logger.warning("[get_instrument_tokens] Primary LTP fetch failed. Trying alternatives...")
            alternative_symbols = [
                'NSE:NIFTY 50',
                'NSE:NIFTY50',
                'NSE:NIFTY',
                'NIFTY 50'
            ]
            
            for alt_symbol in alternative_symbols:
                try:
                    logger.debug(f"Trying alternative spot symbol: {alt_symbol}")
                    ltp_data = _rate_limited_api_call(kite_instance.ltp, [alt_symbol])
                    if isinstance(ltp_data, dict):
                        spot_price = ltp_data.get(alt_symbol, {}).get('last_price')
                        if spot_price:
                            logger.info(f"✅ Spot price fetched with alternative symbol {alt_symbol}: {spot_price}")
                            break
                except Exception:
                    continue
            
            if not spot_price:
                logger.error("[get_instrument_tokens] ❌ Failed to fetch spot price with any symbol format.")
                return None

        logger.info(f"[get_instrument_tokens] Spot price: {spot_price}")

        # 2. Calculate Strike and Expiry
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        
        # Get expiry with error handling
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)
        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine expiry date.")
            return None

        logger.info(f"[get_instrument_tokens] Target - Expiry: {expiry_yyyy_mm_dd}, Strike: {target_strike}, Offset: {offset}")

        # 3. Filter instruments efficiently
        nifty_instruments = [
            inst for inst in cached_nfo_instruments 
            if inst.get('name') == base_symbol_for_search
        ]
        
        if not nifty_instruments:
            logger.error(f"[get_instrument_tokens] No {base_symbol_for_search} instruments in cache")
            return None

        # Filter for target expiry
        expiry_instruments = []
        for inst in nifty_instruments:
            inst_expiry = inst.get('expiry')
            if inst_expiry:
                if hasattr(inst_expiry, 'strftime'):
                    inst_expiry_str = inst_expiry.strftime("%Y-%m-%d")
                else:
                    inst_expiry_str = str(inst_expiry)
                
                if inst_expiry_str == expiry_yyyy_mm_dd:
                    expiry_instruments.append(inst)

        if not expiry_instruments:
            logger.error(f"[get_instrument_tokens] No instruments found for expiry {expiry_yyyy_mm_dd}")
            # Show available expiries for debugging
            available_expiries = set()
            for inst in nifty_instruments:
                exp = inst.get('expiry')
                if exp:
                    exp_str = exp.strftime("%Y-%m-%d") if hasattr(exp, 'strftime') else str(exp)
                    available_expiries.add(exp_str)
            logger.info(f"Available expiries: {sorted(available_expiries)}")
            return None

        logger.info(f"[get_instrument_tokens] Found {len(expiry_instruments)} instruments for expiry {expiry_yyyy_mm_dd}")

        # 4. Initialize results
        results = {
            "spot_price": spot_price,
            "atm_strike": base_strike,  # Actual ATM
            "target_strike": target_strike,  # Target with offset
            "offset": offset,
            "actual_strikes": {},
            "expiry": expiry_yyyy_mm_dd,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None,
            "spot_token": None
        }

        # 5. Search for CE and PE tokens
        for opt_type in ['CE', 'PE']:
            found = False
            logger.debug(f"[get_instrument_tokens] Searching for {opt_type}...")
            
            # Define strike search order: target first, then expand
            strike_attempts = [target_strike]
            for i in range(1, strike_range + 1):
                strike_attempts.extend([target_strike + i * 50, target_strike - i * 50])
            
            for attempt_strike in strike_attempts:
                if found:
                    break
                
                logger.debug(f"[get_instrument_tokens] Trying {opt_type} strike: {attempt_strike}")

                # Try direct symbol matching first
                symbol_variants = _construct_symbol_variants(base_symbol_for_search, expiry_yyyy_mm_dd, attempt_strike, opt_type)
                
                for variant in symbol_variants:
                    matching_inst = next(
                        (inst for inst in expiry_instruments 
                         if inst.get('tradingsymbol') == variant and inst.get('instrument_type') == opt_type), 
                        None
                    )
                    
                    if matching_inst:
                        logger.info(f"[get_instrument_tokens] ✅ {opt_type} symbol match: {variant} (token: {matching_inst['instrument_token']})")
                        if opt_type == 'CE':
                            results['ce_symbol'] = variant
                            results['ce_token'] = matching_inst['instrument_token']
                            results['actual_strikes']['ce'] = attempt_strike
                        else: # PE
                            results['pe_symbol'] = variant
                            results['pe_token'] = matching_inst['instrument_token']
                            results['actual_strikes']['pe'] = attempt_strike
                        found = True
                        break # Move to next opt_type
                
                if found:
                    continue

                # If no direct symbol match, try fuzzy finding as a fallback
                logger.debug(f"[get_instrument_tokens] No direct symbol match for {opt_type} at strike {attempt_strike}, trying fuzzy find.")
                fuzzy_match = _fuzzy_find_instrument(expiry_instruments, base_symbol_for_search, expiry_yyyy_mm_dd, attempt_strike, opt_type)
                
                if fuzzy_match:
                    logger.info(f"[get_instrument_tokens] ✅ {opt_type} fuzzy match: {fuzzy_match['symbol']} (token: {fuzzy_match['token']})")
                    if opt_type == 'CE':
                        results['ce_symbol'] = fuzzy_match['symbol']
                        results['ce_token'] = fuzzy_match['token']
                        results['actual_strikes']['ce'] = fuzzy_match['strike']
                    else: # PE
                        results['pe_symbol'] = fuzzy_match['symbol']
                        results['pe_token'] = fuzzy_match['token']
                        results['actual_strikes']['pe'] = fuzzy_match['strike']
                    found = True
                    break # Move to next opt_type

            if not found:
                logger.error(f"[get_instrument_tokens] ❌ Could not find a valid instrument for {opt_type} near strike {target_strike}")

        # 6. Final validation and return
        if results.get('ce_token') and results.get('pe_token'):
            logger.info(f"✅ Successfully found tokens: CE={results['ce_token']}, PE={results['pe_token']}")
            return results
        else:
            logger.error("[get_instrument_tokens] ❌ Failed to retrieve one or both instrument tokens.")
            return None

    except Exception as e:
        logger.error(f"[get_instrument_tokens] 💥 Unhandled exception: {e}", exc_info=True)
        return None
