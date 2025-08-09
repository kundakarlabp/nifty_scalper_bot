# src/utils/strike_selector.py
"""
Complete utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

This version keeps your structure, adds robust symbol resolution,
fallbacks, cached instruments usage, and clearer logs.
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import time
import threading
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# --- Global rate limiting and caching ---
_last_api_call = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # Minimum 500ms between API calls

def _rate_limited_api_call(func, *args, **kwargs):
    """Rate-limited API call wrapper with single retry on rate-limit."""
    with _api_call_lock:
        call_key = getattr(func, "__name__", "api_call")
        now = time.time()
        if call_key in _last_api_call:
            elapsed = now - _last_api_call[call_key]
            if elapsed < _MIN_API_INTERVAL:
                time.sleep(_MIN_API_INTERVAL - elapsed)

        try:
            result = func(*args, **kwargs)
            _last_api_call[call_key] = time.time()
            return result
        except Exception as e:
            msg = str(e).lower()
            if "too many" in msg or "rate" in msg:
                logger.warning(f"Rate limit for {call_key}, retrying in 2s...")
                time.sleep(2)
                result = func(*args, **kwargs)
                _last_api_call[call_key] = time.time()
                return result
            raise

def _get_spot_ltp_symbol() -> str:
    """Read SPOT_SYMBOL from Config; fall back to 'NSE:NIFTY 50'."""
    try:
        from src.config import Config
        sym = getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50')
        return sym or 'NSE:NIFTY 50'
    except Exception as e:
        logger.debug(f"SPOT_SYMBOL fallback due to: {e}")
        return 'NSE:NIFTY 50'

def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """Format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')"""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except Exception as e:
        logger.error(f"[_format_expiry_for_symbol_primary] {e}")
        return ""

def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """Primary tradingsymbol format used by Zerodha for NIFTY weeklys."""
    try:
        exp = _format_expiry_for_symbol_primary(expiry)
        return f"{base_symbol}{exp}{int(strike)}{option_type}" if exp else ""
    except Exception as e:
        logger.error(f"[format_option_symbol] {e}")
        return ""

def get_atm_strike_price(spot_price: float) -> int:
    """Nearest 50-step strike."""
    try:
        return int(round(spot_price / 50.0) * 50)
    except Exception as e:
        logger.error(f"[get_atm_strike_price] {e}")
        return 24500

def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    """Centered range around ATM."""
    try:
        atm = get_atm_strike_price(spot_price)
        half = max(1, strike_count // 2)
        strikes = sorted(set(atm + i * 50 for i in range(-half, half + 1)))
        if strikes:
            logger.info(f"🎯 Strike range: {min(strikes)} - {max(strikes)} ({len(strikes)} strikes)")
        return strikes
    except Exception as e:
        logger.error(f"[get_nearest_strikes] {e}", exc_info=True)
        return []

def _calculate_next_thursday(target_date: Optional[datetime] = None) -> str:
    """Pure calendar fallback for 'next Thursday' in YYYY-MM-DD."""
    d = target_date or datetime.now()
    days_ahead = (3 - d.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    nxt = d + timedelta(days=days_ahead)
    return nxt.strftime("%Y-%m-%d")

# ---------------- Cached instruments helpers ----------------

def fetch_cached_instruments(kite: KiteConnect) -> Dict[str, List[Dict[str, Any]]]:
    """
    One-shot fetch for 'NFO' and 'NSE' instruments with rate limiting.
    Use this at app start and refresh occasionally to avoid repeated API calls.
    """
    try:
        nfo = _rate_limited_api_call(kite.instruments, "NFO")
    except Exception as e:
        logger.error(f"[fetch_cached_instruments] NFO fetch failed: {e}")
        nfo = []

    try:
        nse = _rate_limited_api_call(kite.instruments, "NSE")
    except Exception as e:
        logger.error(f"[fetch_cached_instruments] NSE fetch failed: {e}")
        nse = []

    return {"NFO": nfo or [], "NSE": nse or []}

# ---------------- Core selection functions ----------------

def get_next_expiry_date(kite_instance: KiteConnect, cached_nfo_instruments: Optional[List[Dict]] = None) -> Optional[str]:
    """
    Resolve the nearest upcoming expiry for NIFTY options based on cached NFO instruments.
    Falls back to 'next Thursday' if instruments unavailable.
    """
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return _calculate_next_thursday()

    try:
        spot_symbol_config = _get_spot_ltp_symbol()
        # Zerodha instruments 'name' for NIFTY options is 'NIFTY'
        base_name_for_search = "NIFTY"

        if cached_nfo_instruments is None:
            try:
                cached_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
            except Exception as e:
                logger.error(f"[get_next_expiry_date] instruments fetch failed: {e}")
                return _calculate_next_thursday()

        index_instruments = [i for i in cached_nfo_instruments if i.get('name') == base_name_for_search]
        if not index_instruments:
            logger.error(f"[get_next_expiry_date] No NFO instruments for '{base_name_for_search}'")
            return _calculate_next_thursday()

        expiries: List[datetime] = []
        for inst in index_instruments:
            exp = inst.get("expiry")
            if exp:
                if hasattr(exp, "strftime"):
                    expiries.append(exp)
                else:
                    try:
                        expiries.append(datetime.strptime(str(exp), "%Y-%m-%d"))
                    except Exception:
                        pass

        expiries = sorted(set(expiries))
        if not expiries:
            return _calculate_next_thursday()

        today = datetime.now().date()
        for exp in expiries:
            if exp.date() >= today:
                return exp.strftime("%Y-%m-%d")

        # All past? last one as fallback.
        return expiries[-1].strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"[get_next_expiry_date] Error: {e}", exc_info=True)
        return _calculate_next_thursday()

def get_instrument_tokens(
    symbol: str,
    kite_instance: KiteConnect,
    cached_nfo_instruments: List[Dict],
    cached_nse_instruments: List[Dict],
    offset: int = 0,
    strike_range: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Get CE/PE instrument tokens for a given NIFTY symbol with offset.
    Requires cached instruments to avoid API rate limits.
    """
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None
    if not cached_nfo_instruments:
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
        return None
    if not cached_nse_instruments:
        logger.error("[get_instrument_tokens] Cached NSE instruments are required.")
        return None

    try:
        spot_symbol_config = _get_spot_ltp_symbol()  # e.g., 'NSE:NIFTY 50'
        base_name = "NIFTY"  # Zerodha 'name' for NIFTY options

        # 1) Spot LTP
        try:
            spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol_config])
            spot_price = float(spot_data.get(spot_symbol_config, {}).get('last_price') or 0.0)
            if not spot_price:
                logger.error("[get_instrument_tokens] Could not fetch spot price.")
                return None
        except Exception as e:
            logger.error(f"[get_instrument_tokens] LTP error: {e}")
            return None

        # 2) ATM & target
        atm = get_atm_strike_price(spot_price)
        target = atm + (int(offset) * 50)

        # 3) Expiry
        expiry = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry:
            logger.error("[get_instrument_tokens] Could not resolve expiry.")
            return None

        logger.info(f"[get_instrument_tokens] Spot:{spot_price:.2f} ATM:{atm} Target:{target} Expiry:{expiry} Offset:{offset}")

        # 4) Filter relevant instruments: name=NIFTY, expiry matches
        # Note: cached instruments may store 'expiry' as datetime.date/datetime
        def _exp_str(x) -> str:
            if hasattr(x, "strftime"):
                return x.strftime("%Y-%m-%d")
            return str(x)

        candidates = [i for i in cached_nfo_instruments if i.get('name') == base_name and _exp_str(i.get('expiry')) == expiry]
        if not candidates:
            logger.error(f"[get_instrument_tokens] No instruments for {base_name} @ {expiry}")
            return None

        results: Dict[str, Any] = {
            "spot_price": spot_price,
            "atm_strike": atm,
            "target_strike": target,
            "offset": int(offset),
            "actual_strikes": {},
            "expiry": expiry,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None,
            "spot_token": None
        }

        # 5) Search strikes outward from target for CE/PE
        search_order: List[int] = [target]
        for i in range(1, int(strike_range) + 1):
            search_order.extend([target + i * 50, target - i * 50])

        for side in ("CE", "PE"):
            found = False
            for strike in search_order:
                for inst in candidates:
                    if inst.get("instrument_type") == side and int(float(inst.get("strike", 0))) == int(strike):
                        results[f"{side.lower()}_symbol"] = inst.get("tradingsymbol")
                        results[f"{side.lower()}_token"] = inst.get("instrument_token")
                        results["actual_strikes"][side.lower()] = int(strike)
                        logger.info(f"[get_instrument_tokens] Found {side}: {inst.get('tradingsymbol')} ({inst.get('instrument_token')})")
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.warning(f"[get_instrument_tokens] ❌ No {side} within ±{strike_range}*50 points")

        # 6) Validation
        ok = any([results["ce_token"], results["pe_token"]])
        if not ok:
            logger.error("[get_instrument_tokens] ❌ No options found in range")
            return None

        # Log any adjustment from target
        for side in ("ce", "pe"):
            a = results["actual_strikes"].get(side)
            if a and a != target:
                logger.info(f"[get_instrument_tokens] {side.upper()} strike adjusted: {target} → {a}")

        return results

    except Exception as e:
        logger.error(f"[get_instrument_tokens] Unexpected error: {e}", exc_info=True)
        return None

def is_trading_hours() -> bool:
    """Simple NSE hours gate: 09:15–15:30 IST, Mon–Fri."""
    try:
        now = datetime.now()
        wd = now.weekday()
        start = datetime.strptime("09:15", "%H:%M").time()
        end = datetime.strptime("15:30", "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)
    except Exception as e:
        logger.error(f"[is_trading_hours] {e}")
        return True