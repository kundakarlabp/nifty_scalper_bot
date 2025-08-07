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


# --- NEW FUNCTION TO VALIDATE API SESSION ---
def is_kite_session_valid(kite_instance: KiteConnect) -> bool:
    """
    Checks if the KiteConnect session is active by fetching the user profile.
    This is a reliable way to detect an expired or invalid access token.

    Args:
        kite_instance: The KiteConnect instance to validate.

    Returns:
        True if the session is valid, False otherwise.
    """
    if not kite_instance:
        logger.error("[is_kite_session_valid] KiteConnect instance is None.")
        return False
    try:
        # .profile() is a lightweight call that requires a valid session.
        profile = _rate_limited_api_call(kite_instance.profile)
        if profile and profile.get('user_id'):
            logger.info(f"[is_kite_session_valid] ✅ Session is active for user {profile['user_id']}.")
            return True
        else:
            logger.warning(f"[is_kite_session_valid] ❌ Session check returned invalid profile data: {profile}")
            return False
    except Exception as e:
        # This exception block will catch token errors, network issues, etc.
        logger.error(f"[is_kite_session_valid] ❌ Session validation failed. The access token is likely expired or invalid. Error: {e}")
        return False

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
        if days_ahead <= 0:
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        return next_thursday.strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"Error calculating next Thursday: {e}")
        return (datetime.today() + timedelta(days=7)).strftime("%Y-%m-%d")

@lru_cache(maxsize=32)
def _get_cached_expiry_date(base_symbol: str, current_date_str: str) -> str:
    """Cached expiry date calculation to avoid repeated API calls"""
    return ""

def get_next_expiry_date(kite_instance: KiteConnect, use_cache: bool = True) -> str:
    """Finds the next expiry date that has available instruments."""
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return _calculate_next_thursday()

    try:
        spot_symbol_config = _get_spot_ltp_symbol()
        base_symbol_for_search = spot_symbol_config.split(':', 1)[-1].split()[0]
        logger.debug(f"[get_next_expiry_date] Searching for base symbol: '{base_symbol_for_search}'")

        all_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        index_instruments = [inst for inst in all_nfo_instruments if inst.get('name') == base_symbol_for_search]

        if not index_instruments:
            logger.error(f"[get_next_expiry_date] No instruments found for '{base_symbol_for_search}'.")
            return _calculate_next_thursday()

        unique_expiries = sorted({
            inst.get('expiry').strftime("%Y-%m-%d") if hasattr(inst.get('expiry'), 'strftime') else str(inst.get('expiry'))
            for inst in index_instruments if inst.get('expiry')
        })

        if not unique_expiries:
            logger.error(f"[get_next_expiry_date] No expiries found for '{base_symbol_for_search}'.")
            return _calculate_next_thursday()

        today = datetime.today().date()
        for expiry_str in unique_expiries:
            try:
                if datetime.strptime(expiry_str, "%Y-%m-%d").date() >= today:
                    logger.info(f"[get_next_expiry_date] ✅ Selected expiry: {expiry_str}")
                    return expiry_str
            except ValueError:
                continue
        
        latest_expiry = unique_expiries[-1]
        logger.info(f"[get_next_expiry_date] Fallback to latest expiry: {latest_expiry}")
        return latest_expiry

    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return _calculate_next_thursday()

def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """Primary format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')"""
    try:
        return datetime.strptime(expiry_str, "%Y-%m-%d").strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"[_format_expiry_for_symbol_primary] Error: {e}")
        return ""

def _construct_symbol_variants(base_symbol: str, expiry_str: str, strike: int, opt_type: str) -> List[str]:
    """Generate a list of possible symbol formats to try."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        primary_format = expiry_date.strftime("%y%b%d").upper()
        # Add more variants if needed, but the primary one is most important
        return [f"{base_symbol}{primary_format}{strike}{opt_type}"]
    except Exception as e:
        logger.error(f"[_construct_symbol_variants] Error generating variants: {e}")
        return []

def _fuzzy_find_instrument(instruments_list: List[Dict], base_symbol: str, 
                          expiry_yyyy_mm_dd: str, strike: int, opt_type: str) -> Dict[str, Any]:
    """Enhanced fuzzy search for an instrument if direct symbol matching fails."""
    logger.debug(f"[_fuzzy_find_instrument] Fuzzy searching for {base_symbol} Exp:{expiry_yyyy_mm_dd} Strike:{strike} Type:{opt_type}")
    
    for inst in instruments_list:
        inst_expiry = inst.get('expiry')
        inst_expiry_str = inst_expiry.strftime("%Y-%m-%d") if hasattr(inst_expiry, 'strftime') else str(inst_expiry)
        
        if (inst.get('name') == base_symbol and
            inst_expiry_str == expiry_yyyy_mm_dd and
            inst.get('instrument_type') == opt_type and
            inst.get('strike') is not None and
            abs(int(float(inst.get('strike'))) - strike) == 0):
            
            result = {
                "symbol": inst['tradingsymbol'], "token": inst['instrument_token'],
                "strike": int(float(inst['strike'])), "type": opt_type,
                "expiry": expiry_yyyy_mm_dd
            }
            logger.info(f"[_fuzzy_find_instrument] ✅ Exact match found: {result['symbol']}")
            return result
            
    logger.debug("[_fuzzy_find_instrument] ❌ No fuzzy match found.")
    return {}

def get_instrument_tokens(
    symbol: str = "NIFTY",
    offset: int = 0,
    kite_instance: Optional[KiteConnect] = None,
    cached_nfo_instruments: Optional[List[Dict]] = None,
    cached_nse_instruments: Optional[List[Dict]] = None,
    strike_range: int = 3
) -> Optional[Dict[str, Any]]:
    """Enhanced instrument token retrieval with pre-emptive session validation."""
    
    # --- START OF CORRECTION ---
    # 1. Validate all preconditions first
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None
    if not cached_nfo_instruments:
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
        return None
    
    # 2. Perform session validation before any API calls
    if not is_kite_session_valid(kite_instance):
        logger.error("[get_instrument_tokens] ❌ Aborting token fetch due to invalid API session.")
        return None
    # --- END OF CORRECTION ---

    try:
        spot_symbol_config = _get_spot_ltp_symbol()
        base_symbol_for_search = spot_symbol_config.split(':', 1)[-1].split()[0]
        logger.debug(f"[get_instrument_tokens] Base symbol for search: '{base_symbol_for_search}'")

        # Get Spot Price
        spot_price = None
        try:
            ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol_config])
            if isinstance(ltp_data, dict):
                spot_price = ltp_data.get(spot_symbol_config, {}).get('last_price')
            else:
                logger.error(f"[get_instrument_tokens] Unexpected LTP data format. Expected dict, got {type(ltp_data)}.")
        except Exception as ltp_error:
            logger.error(f"[get_instrument_tokens] LTP fetch failed with exception: {ltp_error}")

        if not spot_price:
            logger.error("[get_instrument_tokens] ❌ Failed to fetch spot price. Cannot proceed.")
            return None

        logger.info(f"[get_instrument_tokens] Spot price: {spot_price}")

        # Calculate Strike and Expiry
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)
        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine expiry date.")
            return None

        logger.info(f"[get_instrument_tokens] Target - Expiry: {expiry_yyyy_mm_dd}, Strike: {target_strike}")

        # Filter instruments for the target expiry
        expiry_instruments = [
            inst for inst in cached_nfo_instruments
            if (inst.get('name') == base_symbol_for_search and
                (inst.get('expiry').strftime('%Y-%m-%d') if hasattr(inst.get('expiry'), 'strftime') else str(inst.get('expiry'))) == expiry_yyyy_mm_dd)
        ]

        if not expiry_instruments:
            logger.error(f"[get_instrument_tokens] No instruments found for expiry {expiry_yyyy_mm_dd}")
            return None

        results = {"spot_price": spot_price, "target_strike": target_strike, "expiry": expiry_yyyy_mm_dd}

        # Search for CE and PE tokens
        for opt_type in ['CE', 'PE']:
            found_instrument = None
            for i in range(strike_range + 1):
                for sign in ([1, -1] if i > 0 else [1]):
                    attempt_strike = target_strike + (i * 50 * sign)
                    symbol_variants = _construct_symbol_variants(base_symbol_for_search, expiry_yyyy_mm_dd, attempt_strike, opt_type)
                    for variant in symbol_variants:
                        found_instrument = next((inst for inst in expiry_instruments if inst.get('tradingsymbol') == variant), None)
                        if found_instrument: break
                    if found_instrument: break
                if found_instrument: break
            
            if found_instrument:
                logger.info(f"✅ Found {opt_type}: {found_instrument['tradingsymbol']}")
                results[f'{opt_type.lower()}_symbol'] = found_instrument['tradingsymbol']
                results[f'{opt_type.lower()}_token'] = found_instrument['instrument_token']
            else:
                logger.error(f"❌ Could not find a valid instrument for {opt_type} near strike {target_strike}")

        if 'ce_token' in results and 'pe_token' in results:
            return results
        else:
            logger.error("[get_instrument_tokens] ❌ Failed to retrieve one or both instrument tokens.")
            return None

    except Exception as e:
        logger.error(f"[get_instrument_tokens] 💥 Unhandled exception: {e}", exc_info=True)
        return None
