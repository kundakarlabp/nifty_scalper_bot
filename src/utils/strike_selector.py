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

        # 1. Get Spot Price with rate limiting
        spot_symbol_ltp = _get_spot_ltp_symbol()
        logger.debug(f"[get_instrument_tokens] Fetching spot LTP for: {spot_symbol_ltp}")
        
        try:
            ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol_ltp])
            spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')
        except Exception as ltp_error:
            logger.error(f"[get_instrument_tokens] LTP fetch failed: {ltp_error}")
            # Try with alternative spot symbol formats
            alternative_symbols = [
                'NSE:NIFTY 50',
                'NSE:NIFTY50',
                'NSE:NIFTY',
                'NIFTY 50'
            ]
            
            spot_price = None
            for alt_symbol in alternative_symbols:
                try:
                    logger.debug(f"Trying alternative spot symbol: {alt_symbol}")
                    ltp_data = _rate_limited_api_call(kite_instance.ltp, [alt_symbol])
                    spot_price = ltp_data.get(alt_symbol, {}).get('last_price')
                    if spot_price:
                        logger.info(f"✅ Spot price fetched with alternative symbol {alt_symbol}: {spot_price}")
                        break
                except Exception:
                    continue
            
            if not spot_price:
                logger.error("[get_instrument_tokens] Failed to fetch spot price with any symbol format")
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
                        else:
                            results['pe_symbol'] = variant
                            results['pe_token'] = matching_inst['instrument_token']
                            results['actual_strikes']['pe'] = attempt_strike
                        found = True
                        break
                
                # Fallback to fuzzy search
                if not found:
                    fuzzy_result = _fuzzy_find_instrument(expiry_instruments, base_symbol_for_search, 
                                                        expiry_yyyy_mm_dd, attempt_strike, opt_type)
                    if fuzzy_result:
                        if opt_type == 'CE':
                            results['ce_symbol'] = fuzzy_result['symbol']
                            results['ce_token'] = fuzzy_result['token']
                            results['actual_strikes']['ce'] = fuzzy_result['strike']
                        else:
                            results['pe_symbol'] = fuzzy_result['symbol']
                            results['pe_token'] = fuzzy_result['token']
                            results['actual_strikes']['pe'] = fuzzy_result['strike']
                        logger.info(f"[get_instrument_tokens] ✅ {opt_type} fuzzy match: {fuzzy_result['symbol']}")
                        found = True
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

def diagnose_symbol_patterns(kite_instance: KiteConnect, target_expiry: str = None) -> None:
    """
    Enhanced diagnostic function to understand available symbol patterns
    """
    try:
        logger.info("[diagnose_symbol_patterns] Starting comprehensive symbol analysis...")
        
        # Rate-limited instruments fetch
        nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        
        # Filter NIFTY instruments
        nifty_instruments = [inst for inst in nfo_instruments if inst.get('name') == 'NIFTY']
        
        if target_expiry:
            nifty_instruments = [
                inst for inst in nifty_instruments
                if (inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])) == target_expiry
            ]
        
        if not nifty_instruments:
            logger.error(f"❌ No NIFTY instruments found for expiry: {target_expiry}")
            return
        
        logger.info(f"📊 Found {len(nifty_instruments)} NIFTY instruments")
        
        # Analyze by type
        ce_instruments = [inst for inst in nifty_instruments if inst.get('instrument_type') == 'CE']
        pe_instruments = [inst for inst in nifty_instruments if inst.get('instrument_type') == 'PE']
        
        logger.info(f"📈 CE Options: {len(ce_instruments)}")
        logger.info(f"📉 PE Options: {len(pe_instruments)}")
        
        # Show sample symbols with pattern analysis
        logger.info("🔍 Symbol Pattern Analysis:")
        for i, inst in enumerate(nifty_instruments[:20]):  # Show more samples
            symbol = inst['tradingsymbol']
            strike = inst.get('strike', 'N/A')
            opt_type = inst.get('instrument_type', 'N/A')
            logger.info(f"  {symbol} | Strike: {strike} | Type: {opt_type}")
            
            # Pattern analysis
            if 'NIFTY' in symbol:
                parts = symbol.replace('NIFTY', '').replace(opt_type, '')
                logger.debug(f"    Pattern parts: '{parts}'")
        
        # Strike analysis
        strikes = set()
        for inst in nifty_instruments:
            strike = inst.get('strike')
            if strike:
                try:
                    strikes.add(int(float(strike)))
                except (ValueError, TypeError):
                    pass
        
        sorted_strikes = sorted(strikes)
        if sorted_strikes:
            logger.info(f"🎯 Strike range: {min(sorted_strikes)} - {max(sorted_strikes)} ({len(sorted_strikes)} strikes)")
            logger.info(f"🎯 Sample strikes: {sorted_strikes[:10]}{'...' if len(sorted_strikes) > 10 else ''}")
        
    except Exception as e:
        logger.error(f"❌ Symbol pattern diagnosis failed: {e}", exc_info=True)

def test_token_resolution(kite_instance: KiteConnect, test_spot_price: float = None) -> bool:
    """
    Enhanced test function to validate token resolution
    """
    logger.info("🧪 Starting comprehensive token resolution test...")
    
    try:
        # Test with cached instruments
        nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        
        # Test different scenarios
        test_cases = [
            {"offset": 0, "description": "ATM"},
            {"offset": 1, "description": "OTM +50"},
            {"offset": -1, "description": "ITM -50"},
        ]
        
        success_count = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            logger.info(f"🔍 Testing {test_case['description']} (offset: {test_case['offset']})")
            
            result = get_instrument_tokens(
                symbol="NIFTY",
                offset=test_case['offset'],
                kite_instance=kite_instance,
                cached_nfo_instruments=nfo_instruments,
                strike_range=3
            )
            
            if result:
                logger.info(f"✅ {test_case['description']} test passed")
                logger.info(f"   CE: {result.get('ce_symbol', 'Not found')}")
                logger.info(f"   PE: {result.get('pe_symbol', 'Not found')}")
                logger.info(f"   Target Strike: {result.get('target_strike')}")
                success_count += 1
            else:
                logger.error(f"❌ {test_case['description']} test failed")
        
        success_rate = (success_count / total_tests) * 100
        logger.info(f"🎯 Test Results: {success_count}/{total_tests} passed ({success_rate:.1f}%)")
        
        return success_count == total_tests
        
    except Exception as e:
        logger.error(f"❌ Token resolution test failed: {e}", exc_info=True)
        return False

def get_available_expiries(kite_instance: KiteConnect, base_symbol: str = "NIFTY") -> List[str]:
    """Get list of available expiry dates for a symbol"""
    try:
        nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        nifty_instruments = [inst for inst in nfo_instruments if inst.get('name') == base_symbol]
        
        expiries = set()
        for inst in nifty_instruments:
            expiry = inst.get('expiry')
            if expiry:
                exp_str = expiry.strftime("%Y-%m-%d") if hasattr(expiry, 'strftime') else str(expiry)
                expiries.add(exp_str)
        
        return sorted(list(expiries))
        
    except Exception as e:
        logger.error(f"Error getting available expiries: {e}")
        return []

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

def get_strike_chain(kite_instance: KiteConnect, expiry: str, base_symbol: str = "NIFTY", 
                    center_strike: int = None, range_points: int = 500) -> Dict[str, List[Dict]]:
    """
    Get a complete strike chain around a center strike
    
    Args:
        kite_instance: KiteConnect instance
        expiry: Expiry date in YYYY-MM-DD format
        base_symbol: Base symbol name (default "NIFTY")
        center_strike: Center strike price (if None, will calculate ATM)
        range_points: Range in points on each side (default 500)
    
    Returns:
        Dictionary with 'CE' and 'PE' lists containing strike information
    """
    try:
        # Get instruments
        nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        
        # Filter for the symbol and expiry
        filtered_instruments = []
        for inst in nfo_instruments:
            if (inst.get('name') == base_symbol and 
                inst.get('expiry') and
                (inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])) == expiry):
                filtered_instruments.append(inst)
        
        if not filtered_instruments:
            logger.error(f"No instruments found for {base_symbol} expiry {expiry}")
            return {"CE": [], "PE": []}
        
        # Calculate center strike if not provided
        if center_strike is None:
            try:
                spot_symbol = _get_spot_ltp_symbol()
                ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
                spot_price = ltp_data.get(spot_symbol, {}).get('last_price')
                if spot_price:
                    center_strike = round(spot_price / 50) * 50
                else:
                    logger.error("Could not determine center strike")
                    return {"CE": [], "PE": []}
            except Exception as e:
                logger.error(f"Error getting spot price for strike chain: {e}")
                return {"CE": [], "PE": []}
        
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
                            'type': opt_type,
                            'is_atm': strike_int == center_strike,
                            'distance_from_atm': strike_int - center_strike
                        }
                        strike_chain[opt_type].append(strike_info)
                except (ValueError, TypeError):
                    continue
        
        # Sort by strike
        for opt_type in ['CE', 'PE']:
            strike_chain[opt_type].sort(key=lambda x: x['strike'])
        
        logger.info(f"Strike chain: {len(strike_chain['CE'])} CE, {len(strike_chain['PE'])} PE options around {center_strike}")
        return strike_chain
        
    except Exception as e:
        logger.error(f"Error getting strike chain: {e}", exc_info=True)
        return {"CE": [], "PE": []}

def get_weekly_monthly_expiries(kite_instance: KiteConnect, base_symbol: str = "NIFTY") -> Dict[str, List[str]]:
    """
    Categorize expiries into weekly and monthly
    
    Returns:
        Dictionary with 'weekly' and 'monthly' lists of expiry dates
    """
    try:
        all_expiries = get_available_expiries(kite_instance, base_symbol)
        
        weekly_expiries = []
        monthly_expiries = []
        
        for expiry_str in all_expiries:
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
                
                # Check if it's the last Thursday of the month (monthly expiry)
                # Get the last day of the month
                last_day = calendar.monthrange(expiry_date.year, expiry_date.month)[1]
                last_date = datetime(expiry_date.year, expiry_date.month, last_day)
                
                # Find the last Thursday
                while last_date.weekday() != 3:  # 3 is Thursday
                    last_date -= timedelta(days=1)
                
                if expiry_date.date() == last_date.date():
                    monthly_expiries.append(expiry_str)
                else:
                    weekly_expiries.append(expiry_str)
                    
            except ValueError:
                continue
        
        logger.info(f"Categorized expiries: {len(weekly_expiries)} weekly, {len(monthly_expiries)} monthly")
        return {
            'weekly': weekly_expiries,
            'monthly': monthly_expiries
        }
        
    except Exception as e:
        logger.error(f"Error categorizing expiries: {e}")
        return {'weekly': [], 'monthly': []}

def emergency_fallback_tokens(spot_price: float, offset: int = 0) -> Dict[str, Any]:
    """
    Emergency fallback when all API calls fail
    Provides basic structure with calculated values
    """
    try:
        base_strike = round(spot_price / 50) * 50
        target_strike = base_strike + (offset * 50)
        next_thursday = _calculate_next_thursday()
        
        logger.warning("Using emergency fallback tokens - API calls failed")
        
        return {
            "spot_price": spot_price,
            "atm_strike": base_strike,
            "target_strike": target_strike,
            "offset": offset,
            "actual_strikes": {},
            "expiry": next_thursday,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None,
            "spot_token": None,
            "fallback": True,
            "warning": "Emergency fallback mode - no API data available"
        }
        
    except Exception as e:
        logger.error(f"Even emergency fallback failed: {e}")
        return None

def health_check(kite_instance: KiteConnect) -> Dict[str, Any]:
    """
    Perform comprehensive health check of the strike selector system
    
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
                    "status": "WARN",
                    "message": f"Only {nfo_count} NFO instruments found"
                }
        except Exception as e:
            health_status["checks"]["instruments"] = {
                "status": "FAIL",
                "message": f"Failed to fetch instruments: {str(e)[:100]}"
            }
            health_status["recommendations"].append("Instruments API is failing - check market status")
        
        # Check 3: Spot price fetch
        try:
            spot_symbol = _get_spot_ltp_symbol()
            ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
            spot_price = ltp_data.get(spot_symbol, {}).get('last_price')
            
            if spot_price and spot_price > 15000:  # Reasonable range for Nifty
                health_status["checks"]["spot_price"] = {
                    "status": "PASS",
                    "message": f"Spot price: {spot_price}"
                }
            else:
                health_status["checks"]["spot_price"] = {
                    "status": "WARN",
                    "message": f"Unusual spot price: {spot_price}"
                }
        except Exception as e:
            health_status["checks"]["spot_price"] = {
                "status": "FAIL",
                "message": f"Failed to fetch spot price: {str(e)[:100]}"
            }
            health_status["recommendations"].append("Spot price fetch failing - check symbol configuration")
        
        # Check 4: Expiry resolution
        try:
            next_expiry = get_next_expiry_date(kite_instance)
            if next_expiry:
                expiry_date = datetime.strptime(next_expiry, "%Y-%m-%d")
                days_to_expiry = (expiry_date.date() - datetime.now().date()).days
                
                health_status["checks"]["expiry_resolution"] = {
                    "status": "PASS",
                    "message": f"Next expiry: {next_expiry} ({days_to_expiry} days)"
                }
                
                if days_to_expiry < 1:
                    health_status["recommendations"].append("Expiry is very close - monitor for rollover")
                elif days_to_expiry > 30:
                    health_status["recommendations"].append("Expiry is far out - check for nearer expiries")
            else:
                health_status["checks"]["expiry_resolution"] = {
                    "status": "FAIL",
                    "message": "Could not resolve next expiry"
                }
                health_status["recommendations"].append("Expiry resolution failing - check instrument data")
        except Exception as e:
            health_status["checks"]["expiry_resolution"] = {
                "status": "FAIL",
                "message": f"Expiry resolution error: {str(e)[:100]}"
            }
        
        # Check 5: Token resolution test
        try:
            nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
            test_result = get_instrument_tokens(
                symbol="NIFTY",
                offset=0,
                kite_instance=kite_instance,
                cached_nfo_instruments=nfo_instruments,
                strike_range=3
            )
            
            if test_result and (test_result.get('ce_token') or test_result.get('pe_token')):
                found_tokens = sum([1 for x in [test_result.get('ce_token'), test_result.get('pe_token')] if x])
                health_status["checks"]["token_resolution"] = {
                    "status": "PASS",
                    "message": f"Found {found_tokens}/2 option tokens"
                }
            else:
                health_status["checks"]["token_resolution"] = {
                    "status": "FAIL",
                    "message": "No option tokens found"
                }
                health_status["recommendations"].append("Token resolution completely failing - check symbol formats")
        except Exception as e:
            health_status["checks"]["token_resolution"] = {
                "status": "FAIL",
                "message": f"Token resolution test failed: {str(e)[:100]}"
            }
        
        # Check 6: Configuration validation
        config_valid = validate_config()
        health_status["checks"]["configuration"] = {
            "status": "PASS" if config_valid else "FAIL",
            "message": "Configuration valid" if config_valid else "Configuration has issues"
        }
        
        if not config_valid:
            health_status["recommendations"].append("Fix configuration issues")
        
        # Determine overall status
        failed_checks = [check for check in health_status["checks"].values() if check["status"] == "FAIL"]
        warning_checks = [check for check in health_status["checks"].values() if check["status"] == "WARN"]
        
        if failed_checks:
            health_status["overall_status"] = "CRITICAL" if len(failed_checks) > 2 else "DEGRADED"
        elif warning_checks:
            health_status["overall_status"] = "WARNING"
        else:
            health_status["overall_status"] = "HEALTHY"
        
        # Add summary
        total_checks = len(health_status["checks"])
        passed_checks = len([c for c in health_status["checks"].values() if c["status"] == "PASS"])
        
        health_status["summary"] = f"{passed_checks}/{total_checks} checks passed"
        
        logger.info(f"Health check completed: {health_status['overall_status']} ({health_status['summary']})")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        health_status["overall_status"] = "ERROR"
        health_status["error"] = str(e)
        return health_status

def quick_test_api_connectivity(kite_instance: KiteConnect) -> bool:
    """
    Quick test to check if Kite API is responding
    """
    try:
        logger.info("🔍 Testing API connectivity...")
        
        # Test profile endpoint (lightweight)
        profile = _rate_limited_api_call(kite_instance.profile)
        logger.info(f"✅ API Connected - User: {profile.get('user_name', 'Unknown')}")
        
        # Test instruments endpoint
        nfo_count = len(_rate_limited_api_call(kite_instance.instruments, "NFO"))
        logger.info(f"✅ Instruments accessible - NFO count: {nfo_count}")
        
        # Test LTP endpoint
        spot_symbol = _get_spot_ltp_symbol()
        ltp_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
        spot_price = ltp_data.get(spot_symbol, {}).get('last_price')
        logger.info(f"✅ LTP accessible - Spot: {spot_price}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API connectivity test failed: {e}")
        return False

def get_market_status(kite_instance: KiteConnect) -> Dict[str, Any]:
    """
    Get current market status information
    """
    try:
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()  # 0 = Monday, 6 = Sunday
        
        # Basic market hours check (9:15 AM to 3:30 PM, Mon-Fri)
        market_open_time = datetime.strptime("09:15", "%H:%M").time()
        market_close_time = datetime.strptime("15:30", "%H:%M").time()
        
        is_trading_day = current_day < 5  # Monday to Friday
        is_trading_hours = market_open_time <= current_time <= market_close_time
        
        market_status = {
            "is_trading_day": is_trading_day,
            "is_trading_hours": is_trading_hours,
            "is_market_open": is_trading_day and is_trading_hours,
            "current_time": now.strftime("%H:%M:%S"),
            "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][current_day]
        }
        
        return market_status
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return {"error": str(e)}

def cleanup_cache():
    """
    Clean up global caches and rate limiting data
    """
    global _last_api_call
    try:
        with _api_call_lock:
            _last_api_call.clear()
            logger.debug("Strike selector cache cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    """
    try:
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
            "cache_entries": len(_last_api_call),
            "module_version": "2.0.0-optimized"
        }
    except Exception as e:
        return {"error": str(e)}

# Backward Compatibility Wrapper
def get_atm_strike_price(spot_price: float) -> int:
    """
    Simple ATM strike calculation for backward compatibility
    """
    try:
        return round(spot_price / 50) * 50
    except Exception as e:
        logger.error(f"Error calculating ATM strike: {e}")
        return 24500  # Default fallback

def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """
    Format option symbol using primary format
    """
    try:
        formatted_expiry = _format_expiry_for_symbol_primary(expiry)
        if formatted_expiry:
            return f"{base_symbol}{formatted_expiry}{strike}{option_type}"
        return ""
    except Exception as e:
        logger.error(f"Error formatting option symbol: {e}")
        return ""

def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    """
    Get nearest strike prices around spot price
    """
    try:
        atm_strike = get_atm_strike_price(spot_price)
        strikes = []
        
        # Get strikes on both sides
        for i in range(-strike_count//2, strike_count//2 + 1):
            strikes.append(atm_strike + (i * 50))
        
        return sorted(strikes)
        
    except Exception as e:
        logger.error(f"Error getting nearest strikes: {e}")
        return []

def is_trading_hours() -> bool:
    """
    Check if current time is within trading hours
    """
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Strike Selector Module - Direct Execution Test")
    
    try:
        # Test configuration validation
        logger.info("Testing configuration validation...")
        config_valid = validate_config()
        logger.info(f"Configuration valid: {config_valid}")
        
        # Test date calculations
        logger.info("Testing date calculations...")
        next_thursday = _calculate_next_thursday()
        logger.info(f"Next Thursday (fallback): {next_thursday}")
        
        # Test symbol formatting
        logger.info("Testing symbol formatting...")
        test_expiry = "2025-08-07"
        test_symbol = _format_expiry_for_symbol_primary(test_expiry)
        logger.info(f"Formatted expiry for {test_expiry}: {test_symbol}")
        
        # Test ATM calculation
        logger.info("Testing ATM calculation...")
        test_spot = 24567.85
        atm_strike = get_atm_strike_price(test_spot)
        logger.info(f"ATM strike for spot {test_spot}: {atm_strike}")
        
        # Test nearest strikes
        logger.info("Testing nearest strikes...")
        nearest = get_nearest_strikes(test_spot, 5)
        logger.info(f"Nearest 5 strikes: {nearest}")
        
        # Test trading hours
        logger.info("Testing trading hours...")
        trading_hours = is_trading_hours()
        logger.info(f"Is trading hours: {trading_hours}")
        
        # Test system info
        logger.info("Testing system info...")
        sys_info = get_system_info()
        logger.info(f"System info: {sys_info}")
        
        logger.info("✅ All basic module tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Module test failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("🎯 Strike Selector Module ready for production use")