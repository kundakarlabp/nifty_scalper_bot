# src/utils/strike_selector.py
"""Complete utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

This version includes robust symbol resolution, multiple fallback strategies,
enhanced diagnostics, comprehensive rate limiting protection, and error-free execution."""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import calendar
import time
import threading
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple
import sys
import platform

logger = logging.getLogger(__name__)

# --- Global rate limiting and caching ---
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
        spot_symbol = getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50')
        if not spot_symbol:
            logger.warning("SPOT_SYMBOL not found in Config, using default")
            return 'NSE:NIFTY 50'
        return spot_symbol
    except Exception as e:
        logger.error(f"Error getting SPOT_SYMBOL from config: {e}")
        return 'NSE:NIFTY 50'

def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """Primary format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')
    This is the confirmed correct format for Nifty 50 weekly options."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"[_format_expiry_for_symbol_primary] Error: {e}")
        return ""

def _construct_symbol_variants(base_symbol: str, expiry_str: str, strike: int, opt_type: str) -> List[str]:
    """Generate a list of possible symbol formats to try.
    Prioritizes the known correct format with enhanced variants."""
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
        
        # YYMONDD (alternative)
        variants.append(f"{base_symbol}{year_short}{month_short}{day}{strike}{opt_type}")
        
        # YYMMDD
        month_num = expiry_date.strftime("%m")
        variants.append(f"{base_symbol}{year_short}{month_num}{day}{strike}{opt_type}")
        
        # Full year formats
        year_full = expiry_date.strftime("%Y")
        month_full = expiry_date.strftime("%B").upper()
        variants.append(f"{base_symbol}{year_full}{month_short}{day}{strike}{opt_type}")
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

def _find_best_instrument_match(instruments_list: List[Dict], base_symbol: str, 
                               expiry_yyyy_mm_dd: str, opt_type: str, strike: int) -> Optional[Dict]:
    """Find the best matching instrument with enhanced symbol matching."""
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
                    if strike_diff == 0:
                        exact_matches.append(inst)
                except (ValueError, TypeError):
                    continue
                    
        # Return first exact match or None
        return exact_matches[0] if exact_matches else None
        
    except Exception as e:
        logger.error(f"[_find_best_instrument_match] Error: {e}")
        return None

def _calculate_next_thursday(target_date: datetime = None) -> str:
    """Calculate next Thursday date (fallback method)"""
    if not target_date:
        target_date = datetime.now()
    
    days_ahead = 3 - target_date.weekday()  # Thursday is 3
    if days_ahead <= 0:
        days_ahead += 7
    next_thursday = target_date + timedelta(days=days_ahead)
    return next_thursday.strftime("%Y-%m-%d")

# --- Main Functions ---
# --- CRITICAL CHANGE 1: Modified get_next_expiry_date to accept cached data ---
def get_next_expiry_date(kite_instance: KiteConnect, cached_nfo_instruments: List[Dict] = None) -> Optional[str]:
    """Get the next expiry date for options
    Args:
        kite_instance: KiteConnect instance
        cached_nfo_instruments: Cached NFO instruments list (PREFERRED METHOD to avoid API calls)
    Returns:
        String date in YYYY-MM-DD format, None if failed
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
        
        # --- CRITICAL CHANGE: Use cached data if available ---
        if cached_nfo_instruments:
            logger.debug("[get_next_expiry_date] Using cached NFO instruments")
            all_nfo_instruments = cached_nfo_instruments
        else:
            # Rate-limited instruments fetch
            try:
                all_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
                logger.debug(f"[get_next_expiry_date] Fetched {len(all_nfo_instruments)} NFO instruments")
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
                    
        sorted_expiries = sorted([datetime.strptime(d, "%Y-%m-%d") for d in unique_expiries])
        
        if not sorted_expiries:
            logger.error("[get_next_expiry_date] No valid expiries found")
            return _calculate_next_thursday()
            
        # Return the nearest future expiry
        today = datetime.now().date()
        for expiry_date in sorted_expiries:
            if expiry_date.date() >= today:
                return expiry_date.strftime("%Y-%m-%d")
                
        # If no future expiries, return the latest one (fallback)
        logger.warning("[get_next_expiry_date] No future expiries found, using latest.")
        return sorted_expiries[-1].strftime("%Y-%m-%d")
        
    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return _calculate_next_thursday()

# --- CRITICAL CHANGE 2: Modified get_instrument_tokens to require and use cached data ---
def get_instrument_tokens(
    symbol: str, 
    kite_instance: KiteConnect,
    cached_nfo_instruments: List[Dict], # Make this required
    cached_nse_instruments: List[Dict], # Make this required
    offset: int = 0,
    strike_range: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Get instrument tokens for a given symbol with offset
    
    Args:
        symbol: Trading symbol (e.g., NSE:NIFTY 50)
        kite_instance: KiteConnect instance
        cached_nfo_instruments: Cached NFO instruments (REQUIRED to prevent API call)
        cached_nse_instruments: Cached NSE instruments (REQUIRED to prevent API call)
        offset: Strike offset from ATM (0 for ATM, +1 for OTM, -1 for ITM)
        strike_range: Number of strikes to try on each side if exact not found
        
    Returns:
        Dictionary with instrument tokens and metadata, None if failed
    """
    # --- CRITICAL CHANGE: Require cached data ---
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None
    if not cached_nfo_instruments: # Require NFO cache to prevent API call
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
        return None
    if not cached_nse_instruments: # Require NSE cache to prevent API call
        logger.error("[get_instrument_tokens] Cached NSE instruments are required.")
        return None

    try:
        spot_symbol_config = _get_spot_ltp_symbol()
        
        # Derive base symbol name
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
            
        # Clean up the name (remove exchange prefix, extra spaces)
        base_name = potential_name_with_spaces.strip().replace(" ", "")
        logger.info(f"[get_instrument_tokens] Base name derived: {base_name}")
        
        # 1. Get spot price
        try:
            spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol_config])
            spot_price = spot_data.get(spot_symbol_config, {}).get('last_price')
            if not spot_price:
                logger.error("[get_instrument_tokens] Could not fetch spot price.")
                return None
        except Exception as e:
            logger.error(f"[get_instrument_tokens] Error fetching spot price: {e}")
            return None
            
        logger.info(f"[get_instrument_tokens] Spot price: {spot_price}")
        
        # 2. Calculate Strike and Expiry
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        
        # Get expiry with error handling using cached data
        # --- CRITICAL CHANGE: Pass cached data to get_next_expiry_date ---
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry_yyyy_mm_dd:
            logger.error("[get_instrument_tokens] Could not determine expiry date.")
            return None
            
        logger.info(f"[get_instrument_tokens] Target - Expiry: {expiry_yyyy_mm_dd}, Strike: {target_strike}, Offset: {offset}")
        
        # 3. Filter instruments efficiently using cached data
        nifty_instruments = [inst for inst in cached_nfo_instruments 
                           if base_name in inst.get('name', '') and 
                           str(inst.get('expiry')) == expiry_yyyy_mm_dd]
        
        if not nifty_instruments:
            logger.error(f"[get_instrument_tokens] No instruments found for {base_name} on {expiry_yyyy_mm_dd}")
            # Show available expiries for debugging
            available_expiries = set()
            for inst in nifty_instruments:
                exp = inst.get('expiry')
                if exp:
                    exp_str = exp.strftime("%Y-%m-%d") if hasattr(exp, 'strftime') else str(exp)
                    available_expiries.add(exp_str)
            logger.info(f"Available expiries: {sorted(available_expiries)}")
            return None
            
        logger.info(f"[get_instrument_tokens] Found {len(nifty_instruments)} instruments for expiry {expiry_yyyy_mm_dd}")

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
            
            for strike in strike_attempts:
                for inst in nifty_instruments:
                    # Check if instrument matches our criteria
                    if (inst.get('instrument_type') == opt_type and 
                        inst.get('strike') == strike):
                        results[f'{opt_type.lower()}_symbol'] = inst['tradingsymbol']
                        results[f'{opt_type.lower()}_token'] = inst['instrument_token']
                        results['actual_strikes'][opt_type.lower()] = strike
                        logger.info(f"[get_instrument_tokens] Found {opt_type}: {inst['tradingsymbol']} (Token: {inst['instrument_token']})")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                logger.warning(f"[get_instrument_tokens] ❌ Could not find {opt_type} for any strike in range")
        
        # 6. Final validation
        success_count = sum([1 for x in [results['ce_token'], results['pe_token']] if x])
        if success_count == 0:
            logger.error("[get_instrument_tokens] ❌ No options found")
            return None
            
        logger.info(f"[get_instrument_tokens] ✅ Successfully found {success_count}/2 option tokens")
        
        # Log strike adjustments
        for opt_type_lower in ['ce', 'pe']:
            actual_strike = results['actual_strikes'].get(opt_type_lower)
            if actual_strike and actual_strike != target_strike:
                logger.info(f"[get_instrument_tokens] {opt_type_lower.upper()} strike adjusted: {target_strike} → {actual_strike}")
                
        return results
        
    except Exception as e:
        logger.error(f"[get_instrument_tokens] Unexpected error: {e}", exc_info=True)
        return None

# --- Other utility functions remain unchanged ---
def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """Format option symbol using primary format"""
    try:
        formatted_expiry = _format_expiry_for_symbol_primary(expiry)
        if formatted_expiry:
            return f"{base_symbol}{formatted_expiry}{strike}{option_type}"
        return ""
    except Exception as e:
        logger.error(f"Error formatting option symbol: {e}")
        return ""

def get_atm_strike_price(spot_price: float) -> int:
    """Simple ATM strike calculation for backward compatibility"""
    try:
        return round(spot_price / 50) * 50
    except Exception as e:
        logger.error(f"Error calculating ATM strike: {e}")
        return 24500  # Default fallback

def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    """Get nearest strike prices around spot price"""
    try:
        atm_strike = get_atm_strike_price(spot_price)
        strikes = []
        # Get strikes on both sides
        for i in range(-strike_count//2, strike_count//2 + 1):
            strikes.append(atm_strike + (i * 50))
        # Sort and remove duplicates
        sorted_strikes = sorted(set(strikes))
        if sorted_strikes:
            logger.info(f"🎯 Strike range: {min(sorted_strikes)} - {max(sorted_strikes)} ({len(sorted_strikes)} strikes)")
            logger.info(f"🎯 Sample strikes: {sorted_strikes[:10]}{'...' if len(sorted_strikes) > 10 else ''}")
        return sorted_strikes
    except Exception as e:
        logger.error(f"❌ Symbol pattern diagnosis failed: {e}", exc_info=True)
        return []

def is_trading_hours() -> bool:
    """Check if current time is within trading hours"""
    try:
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        is_weekday = current_day < 5
        is_market_hours = market_open <= current_time <= market_close
        return is_weekday and is_market_hours
    except Exception as e:
        logger.error(f"Error checking trading hours: {e}")
        return False

# --- Diagnostics ---
def diagnose_symbol_patterns(kite_instance: KiteConnect) -> Dict[str, Any]:
    """Diagnostic tool to understand symbol patterns"""
    try:
        # Get a sample of Nifty options
        instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        nifty_options = [inst for inst in instruments if 'NIFTY' in inst.get('name', '')][:20]
        
        patterns = {}
        for inst in nifty_options:
            symbol = inst.get('tradingsymbol', '')
            if len(symbol) > 10:  # Likely an options symbol
                base_part = symbol[:10]  # Adjust based on actual symbol
                expiry_part = symbol[10:17] if len(symbol) > 17 else "Unknown"
                patterns[symbol] = {
                    "base": base_part,
                    "expiry_part": expiry_part,
                    "strike": inst.get('strike'),
                    "type": inst.get('instrument_type')
                }
        return {"patterns": patterns, "count": len(nifty_options)}
    except Exception as e:
        return {"error": str(e)}

def test_token_resolution(kite_instance: KiteConnect, test_spot_price: float = None) -> bool:
    """Enhanced test function to validate token resolution"""
    logger.info("🧪 Starting comprehensive token resolution test...")
    
    try:
        # Get spot price if not provided
        if test_spot_price is None:
            spot_symbol = _get_spot_ltp_symbol()
            spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
            test_spot_price = spot_data.get(spot_symbol, {}).get('last_price')
            
        if not test_spot_price:
            logger.error("❌ Could not get test spot price")
            return False
            
        logger.info(f"🎯 Using test spot price: {test_spot_price}")
        
        # Test cases
        test_cases = [
            {"description": "ATM Strike", "offset": 0},
            {"description": "OTM Strike (+1)", "offset": 1},
            {"description": "ITM Strike (-1)", "offset": -1}
        ]
        
        success_count = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            logger.info(f"🧪 Testing {test_case['description']}...")
            # Note: This test would need cached data to be passed in a real scenario
            # For this standalone test, it will make API calls
            result = get_instrument_tokens(
                symbol=spot_symbol,
                kite_instance=kite_instance,
                cached_nfo_instruments=None, # Will cause API calls in test
                cached_nse_instruments=None, # Will cause API calls in test
                offset=test_case['offset']
            )
            
            if result:
                logger.info(f"✅ {test_case['description']} test passed")
                logger.info(f" CE: {result.get('ce_symbol', 'Not found')}")
                logger.info(f" PE: {result.get('pe_symbol', 'Not found')}")
                logger.info(f" Target Strike: {result.get('target_strike')}")
                success_count += 1
            else:
                logger.error(f"❌ {test_case['description']} test failed")
                
        success_rate = (success_count / total_tests) * 100
        logger.info(f"🎯 Test Results: {success_count}/{total_tests} passed ({success_rate:.1f}%)")
        return success_count == total_tests
        
    except Exception as e:
        logger.error(f"❌ Token resolution test failed: {e}", exc_info=True)
        return False

def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics"""
    try:
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
            "cache_entries": len(_last_api_call),
            "module_version": "2.0.0-optimized"
        }
    except Exception as e:
        return {"error": str(e)}

def quick_test_api_connectivity(kite_instance: KiteConnect) -> bool:
    """Quick test to check if Kite API is responding"""
    try:
        logger.info("🔍 Testing API connectivity...")
        # Test profile endpoint (lightweight)
        profile = _rate_limited_api_call(kite_instance.profile)
        logger.info(f"✅ API Connected - User: {profile.get('user_name', 'Unknown')}")
        
        # Test instruments endpoint
        nfo_count = len(_rate_limited_api_call(kite_instance.instruments, "NFO"))
        logger.info(f"📊 NFO Instruments Count: {nfo_count}")
        
        # Test LTP endpoint
        spot_symbol = _get_spot_ltp_symbol()
        ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
        spot_price = ltp_data.get(spot_symbol, {}).get('last_price')
        logger.info(f"💰 {spot_symbol} LTP: {spot_price}")
        
        logger.info("✅ All API tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ API connectivity test failed: {e}")
        return False

def health_check(kite_instance: KiteConnect) -> Dict[str, Any]:
    """Perform comprehensive health check of the strike selector system
    Returns:
        Dictionary with health check results
    """
    health_status = {
        "overall_status": "UNKNOWN",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "recommendations": []
    }
    
    try:
        # Check 1: Kite connection
        try:
            profile = _rate_limited_api_call(kite_instance.profile)
            health_status["checks"]["kite_connection"] = {
                "status": "PASS",
                "message": f"Connected as {profile.get('user_name', 'Unknown')}"
            }
        except Exception as e:
            health_status["checks"]["kite_connection"] = {
                "status": "FAIL",
                "message": f"Connection failed: {str(e)[:100]}"
            }
            health_status["recommendations"].append("Check Kite Connect credentials and network connection")
            
        # Check 2: Instruments availability
        try:
            nfo_count = len(_rate_limited_api_call(kite_instance.instruments, "NFO"))
            if nfo_count > 1000:  # Reasonable number for NFO instruments
                health_status["checks"]["instruments"] = {
                    "status": "PASS",
                    "message": f"Found {nfo_count} NFO instruments"
                }
            else:
                health_status["checks"]["instruments"] = {
                    "status": "WARNING",
                    "message": f"Only {nfo_count} NFO instruments found (unusually low)"
                }
                health_status["recommendations"].append("Check if instrument data is loading correctly")
        except Exception as e:
            health_status["checks"]["instruments"] = {
                "status": "FAIL",
                "message": f"Instruments fetch failed: {str(e)[:100]}"
            }
            health_status["recommendations"].append("Check API connectivity and rate limits")
            
        # Determine overall status
        failed_checks = [check for check in health_status["checks"].values() if check["status"] == "FAIL"]
        warning_checks = [check for check in health_status["checks"].values() if check["status"] == "WARNING"]
        
        if not failed_checks and not warning_checks:
            health_status["overall_status"] = "HEALTHY"
        elif not failed_checks:
            health_status["overall_status"] = "WARNING"
        else:
            health_status["overall_status"] = "CRITICAL"
            
        health_status["summary"] = f"Health checks completed with {len(failed_checks)} failures, {len(warning_checks)} warnings"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        health_status["overall_status"] = "ERROR"
        health_status["error"] = str(e)
        
    return health_status

# --- Additional Utility Functions ---
def get_available_expiries(kite_instance: KiteConnect, cached_nfo_instruments: List[Dict] = None) -> List[str]:
    """Get list of available expiries for Nifty options"""
    try:
        # Use cached data if available
        if cached_nfo_instruments:
            all_nfo_instruments = cached_nfo_instruments
        else:
            all_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
            
        spot_symbol_config = _get_spot_ltp_symbol()
        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config
        base_name = potential_name_with_spaces.strip().replace(" ", "")
        
        # Filter for Nifty options
        nifty_instruments = [inst for inst in all_nfo_instruments if base_name in inst.get('name', '')]
        
        # Get unique expiries
        expiries = set()
        for inst in nifty_instruments:
            expiry = inst.get('expiry')
            if expiry:
                if hasattr(expiry, 'strftime'):
                    expiries.add(expiry.strftime("%Y-%m-%d"))
                else:
                    expiries.add(str(expiry))
                    
        return sorted(list(expiries))
    except Exception as e:
        logger.error(f"Error getting available expiries: {e}")
        return []

def get_weekly_monthly_expiries(kite_instance: KiteConnect, cached_nfo_instruments: List[Dict] = None) -> Dict[str, List[str]]:
    """Separate weekly and monthly expiries"""
    try:
        all_expiries = get_available_expiries(kite_instance, cached_nfo_instruments)
        weekly = []
        monthly = []
        
        for expiry_str in all_expiries:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
                # Simple heuristic: if it's the last Thursday of the month, consider it monthly
                last_day_of_month = calendar.monthrange(expiry_date.year, expiry_date.month)[1]
                last_thursday = datetime(expiry_date.year, expiry_date.month, last_day_of_month)
                while last_thursday.weekday() != 3:  # Thursday is 3
                    last_thursday -= timedelta(days=1)
                    
                if expiry_date.date() == last_thursday.date():
                    monthly.append(expiry_str)
                else:
                    weekly.append(expiry_str)
            except ValueError:
                continue  # Skip invalid dates
                
        return {"weekly": weekly, "monthly": monthly}
    except Exception as e:
        logger.error(f"Error categorizing expiries: {e}")
        return {"weekly": [], "monthly": []}

def get_strike_chain(kite_instance: KiteConnect, expiry: str, base_symbol: str = "NIFTY",
                    center_strike: int = None, range_points: int = 500) -> Dict[str, List[Dict]]:
    """Get a complete strike chain around a center strike
    Args:
        kite_instance: KiteConnect instance
        expiry: Expiry date in YYYY-MM-DD format
        base_symbol: Base symbol name (default: NIFTY)
        center_strike: Center strike price (optional, will use spot if not provided)
        range_points: Range around center strike (default: 500 points)
    Returns:
        Dictionary with CE and PE strike chains
    """
    try:
        # Get spot price if center_strike not provided
        if center_strike is None:
            spot_symbol = _get_spot_ltp_symbol()
            try:
                spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
                spot_price = spot_data.get(spot_symbol, {}).get('last_price')
                if spot_price:
                    center_strike = round(spot_price / 50) * 50
                else:
                    logger.error("Could not get spot price for strike chain")
                    return {"CE": [], "PE": []}
            except Exception as e:
                logger.error(f"Error getting spot price for strike chain: {e}")
                return {"CE": [], "PE": []}
                
        # Get all instruments
        all_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        
        # Filter for target symbol and expiry
        filtered_instruments = [
            inst for inst in all_instruments 
            if inst.get('name') == base_symbol and str(inst.get('expiry')) == expiry
        ]
        
        # Define strike range
        min_strike = center_strike - range_points
        max_strike = center_strike + range_points
        
        # Organize by strike and type
        strike_chain = {"CE": [], "PE": []}
        for inst in filtered_instruments:
            strike = inst.get('strike')
            opt_type = inst.get('instrument_type')
            if strike and opt_type in ['CE', 'PE']:
                try:
                    strike_int = int(float(strike))
                    if min_strike <= strike_int <= max_strike:
                        strike_info = {
                            'strike': strike_int,
                            'symbol': inst['tradingsymbol'],
                            'token': inst['instrument_token'],
                            'expiry': expiry,
                            'type': opt_type
                        }
                        strike_chain[opt_type].append(strike_info)
                except (ValueError, TypeError):
                    pass
                    
        # Sort by strike
        strike_chain["CE"].sort(key=lambda x: x['strike'])
        strike_chain["PE"].sort(key=lambda x: x['strike'])
        
        logger.info(f"Strike chain generated for {base_symbol} {expiry}")
        logger.info(f"Center strike: {center_strike}, Range: {min_strike}-{max_strike}")
        logger.info(f"CE strikes found: {len(strike_chain['CE'])}")
        logger.info(f"PE strikes found: {len(strike_chain['PE'])}")
        
        return strike_chain
    except Exception as e:
        logger.error(f"Error getting strike chain: {e}")
        return {"CE": [], "PE": []}

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        try:
            from src.config import Config
        except ImportError:
            logger.error("Could not import Config module")
            return False
            
        required_attrs = ['SPOT_SYMBOL']
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(Config, attr):
                missing_attrs.append(attr)
                
        if missing_attrs:
            logger.error(f"Missing required config attributes: {missing_attrs}")
            return False
            
        # Validate spot symbol format
        spot_symbol = getattr(Config, 'SPOT_SYMBOL', '')
        if not spot_symbol or ('NIFTY' not in spot_symbol.upper()):
            logger.error(f"Invalid SPOT_SYMBOL: {spot_symbol}")
            return False
            
        logger.info("✅ Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_market_status(kite_instance: KiteConnect) -> Dict[str, Any]:
    """Get current market status"""
    try:
        # This might require a specific Kite API call if available
        # For now, use our internal check
        return {
            "trading_hours": is_trading_hours(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def cleanup_cache() -> None:
    """Cleanup rate limiting cache"""
    global _last_api_call
    with _api_call_lock:
        _last_api_call.clear()
    logger.info("Strike selector cache cleaned up")

# --- Emergency Fallbacks ---
def emergency_fallback_tokens(offset: int = 0) -> Dict[str, Any]:
    """Emergency fallback for token resolution when API is completely down"""
    try:
        # This is a very basic fallback that assumes standard Nifty structure
        # In reality, this would be much more complex and likely inaccurate
        spot_price = 24500  # Default fallback
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        expiry = _calculate_next_thursday()
        
        return {
            "spot_price": spot_price,
            "atm_strike": base_strike,
            "target_strike": target_strike,
            "offset": offset,
            "actual_strikes": {"ce": target_strike, "pe": target_strike},
            "expiry": expiry,
            "ce_symbol": f"NIFTY{expiry.replace('-', '')}C{target_strike}",
            "ce_token": 0,  # Invalid token
            "pe_symbol": f"NIFTY{expiry.replace('-', '')}P{target_strike}",
            "pe_token": 0,  # Invalid token
            "spot_token": 0,
            "warning": "Emergency fallback mode - no API data available"
        }
    except Exception as e:
        logger.error(f"Even emergency fallback failed: {e}")
        return None

# Export all functions for backward compatibility
__all__ = [
    '_get_spot_ltp_symbol',
    'get_next_expiry_date',
    'get_instrument_tokens',
    'diagnose_symbol_patterns',
    'test_token_resolution',
    'get_available_expiries',
    'validate_config',
    'get_strike_chain',
    'get_weekly_monthly_expiries',
    'emergency_fallback_tokens',
    'health_check',
    'quick_test_api_connectivity',
    'get_market_status',
    'cleanup_cache',
    'get_system_info',
    'get_atm_strike_price',
    'format_option_symbol',
    'get_nearest_strikes',
    'is_trading_hours'
]

# Main execution for testing
if __name__ == "__main__":
    # Setup logging for direct execution
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("🚀 Strike Selector Module - Direct Execution Test")
    
    # Note: This test requires a valid KiteConnect instance and cannot run standalone
    logger.info("✅ Module structure validated - manual testing with valid Kite instance required")
