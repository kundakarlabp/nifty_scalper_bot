# src/utils/strike_selector.py
"""
Utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any

# Optional import to avoid import-time crash when kiteconnect is absent
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "_get_spot_ltp_symbol",
    "format_option_symbol",
    "get_atm_strike_price",
    "get_nearest_strikes",
    "fetch_cached_instruments",
    "get_next_expiry_date",
    "get_instrument_tokens",
    "is_trading_hours",
    "is_market_open",
    "health_check",
]

# --- Global rate limiting and basic call dedup ---
_last_api_call: Dict[str, float] = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # 500ms between calls per endpoint


def _rate_limited_api_call(func, *args, **kwargs):
    """Rate-limited API call wrapper with a single retry on rate-limit."""
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
                logger.warning("Rate limit for %s, retrying in 2s...", call_key)
                time.sleep(2)
                result = func(*args, **kwargs)
                _last_api_call[call_key] = time.time()
                return result
            raise


def _get_spot_ltp_symbol() -> str:
    """Read SPOT_SYMBOL from Config; fall back to 'NSE:NIFTY 50'."""
    try:
        from src.config import Config
        sym = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        return sym or "NSE:NIFTY 50"
    except Exception as e:
        logger.debug("SPOT_SYMBOL fallback due to: %s", e)
        return "NSE:NIFTY 50"


def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """Format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except Exception as e:
        logger.error("[_format_expiry_for_symbol_primary] %s", e)
        return ""


def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """Primary tradingsymbol format used by Zerodha for NIFTY weeklys."""
    try:
        exp = _format_expiry_for_symbol_primary(expiry)
        return f"{base_symbol}{exp}{int(strike)}{option_type}" if exp else ""
    except Exception as e:
        logger.error("[format_option_symbol] %s", e)
        return ""


def get_atm_strike_price(spot_price: float) -> int:
    """Nearest 50-step strike."""
    try:
        return int(round(float(spot_price) / 50.0) * 50)
    except Exception as e:
        logger.error("[get_atm_strike_price] %s", e)
        return 24500


def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    """Centered range around ATM."""
    try:
        atm = get_atm_strike_price(spot_price)
        half = max(1, strike_count // 2)
        strikes = sorted(set(atm + i * 50 for i in range(-half, half + 1)))
        if strikes:
            logger.info("🎯 Strike range: %s - %s (%d strikes)", min(strikes), max(strikes), len(strikes))
        return strikes
    except Exception as e:
        logger.error("[get_nearest_strikes] %s", e, exc_info=True)
        return []


def _calculate_next_thursday(target_date: Optional[date] = None) -> str:
    """Pure calendar fallback for 'next Thursday' in YYYY-MM-DD (always a future Thursday)."""
    d = target_date or date.today()
    days_ahead = (3 - d.weekday()) % 7  # Thu=3
    if days_ahead == 0:
        days_ahead = 7
    nxt = d + timedelta(days=days_ahead)
    return nxt.isoformat()


# ---------------- Cached instruments helpers ----------------
def fetch_cached_instruments(kite: Optional[KiteConnect]) -> Dict[str, List[Dict[str, Any]]]:
    """
    One-shot fetch for 'NFO' and 'NSE' instruments with rate limiting.
    Use this at app start and refresh occasionally to avoid repeated API calls.
    """
    if not kite:
        logger.error("[fetch_cached_instruments] No Kite instance.")
        return {"NFO": [], "NSE": []}

    try:
        nfo = _rate_limited_api_call(kite.instruments, "NFO")
    except Exception as e:
        logger.error("[fetch_cached_instruments] NFO fetch failed: %s", e)
        nfo = []

    try:
        nse = _rate_limited_api_call(kite.instruments, "NSE")
    except Exception as e:
        logger.error("[fetch_cached_instruments] NSE fetch failed: %s", e)
        nse = []

    return {"NFO": nfo or [], "NSE": nse or []}


# ---------------- Core selection functions ----------------
def get_next_expiry_date(
    kite_instance: Optional[KiteConnect],
    cached_nfo_instruments: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Resolve the nearest upcoming expiry for NIFTY options based on cached NFO instruments.
    Returns ISO 'YYYY-MM-DD'. Fallback: calendar 'next Thursday' if instruments unavailable.

    Fix: normalize instrument 'expiry' which may be datetime.date or datetime.datetime,
    and DO NOT call .date() on a date.
    """
    try:
        base_name_for_search = "NIFTY"

        if cached_nfo_instruments is None:
            if not kite_instance:
                return _calculate_next_thursday()
            try:
                cached_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
            except Exception as e:
                logger.warning("[get_next_expiry_date] instruments fetch failed, fallback: %s", e)
                return _calculate_next_thursday()

        index_instruments = [i for i in (cached_nfo_instruments or []) if i.get("name") == base_name_for_search]
        if not index_instruments:
            logger.warning("[get_next_expiry_date] No NFO instruments for '%s', fallback.", base_name_for_search)
            return _calculate_next_thursday()

        candidates: set[date] = set()
        for inst in index_instruments:
            exp = inst.get("expiry")
            if not exp:
                continue
            # Normalize to date
            if isinstance(exp, datetime):
                exp_d = exp.date()
            elif isinstance(exp, date):
                exp_d = exp
            else:
                # Rare string case: try parse 'YYYY-MM-DD'
                try:
                    y, m, d = map(int, str(exp)[:10].split("-"))
                    exp_d = date(y, m, d)
                except Exception:
                    continue
            if exp_d >= date.today():
                candidates.add(exp_d)

        if candidates:
            nearest = min(candidates)
            return nearest.isoformat()

        # Everything in the past → safer to return next Thursday than stale dates
        return _calculate_next_thursday()

    except Exception as e:
        logger.warning("[get_next_expiry_date] Error, using fallback: %s", e, exc_info=True)
        return _calculate_next_thursday()


def _resolve_spot_token_from_cache(
    cached_nse_instruments: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Try to find the NSE instrument token for NIFTY 50 index from cached instruments.
    Fallback to Config.INSTRUMENT_TOKEN if not found.
    """
    try:
        from src.config import Config
        # Common listing in instruments dump for the index:
        # tradingsymbol == "NIFTY 50", exchange == "NSE", segment contains "INDICES"
        for inst in cached_nse_instruments or []:
            tsym = (inst.get("tradingsymbol") or "").strip().upper()
            seg = (inst.get("segment") or "").upper()
            if tsym == "NIFTY 50" and "INDICE" in seg:
                tok = inst.get("instrument_token")
                if tok:
                    return int(tok)
        # fallback to configured token (256265 is widely used for NIFTY index)
        return int(getattr(Config, "INSTRUMENT_TOKEN", 256265))
    except Exception as e:
        logger.debug("[_resolve_spot_token_from_cache] %s", e)
        return None


def get_instrument_tokens(
    symbol: str,  # kept for backward compatibility (unused)
    kite_instance: KiteConnect,
    cached_nfo_instruments: List[Dict[str, Any]],
    cached_nse_instruments: List[Dict[str, Any]],
    offset: int = 0,
    strike_range: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Get CE/PE instrument tokens for a given NIFTY symbol with offset.
    Requires cached instruments to avoid API rate limits.

    Returns dict with:
      {
        "spot_price": float,
        "atm_strike": int,
        "target_strike": int,
        "offset": int,
        "actual_strikes": {"ce": int?, "pe": int?},
        "expiry": "YYYY-MM-DD",
        "ce_symbol": str?,
        "ce_token": int?,
        "pe_symbol": str?,
        "pe_token": int?,
        "spot_token": int?,
      }
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
            spot_price = float(spot_data.get(spot_symbol_config, {}).get("last_price") or 0.0)
            if not spot_price:
                logger.error("[get_instrument_tokens] Could not fetch spot price.")
                return None
        except Exception as e:
            logger.error("[get_instrument_tokens] LTP error: %s", e)
            return None

        # 2) ATM & target
        atm = get_atm_strike_price(spot_price)
        target = atm + (int(offset) * 50)

        # 3) Expiry
        expiry = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry:
            logger.error("[get_instrument_tokens] Could not resolve expiry.")
            return None

        logger.info(
            "[get_instrument_tokens] Spot:%.2f ATM:%s Target:%s Expiry:%s Offset:%s",
            spot_price, atm, target, expiry, offset
        )

        def _exp_str(x) -> str:
            if hasattr(x, "strftime"):
                return x.strftime("%Y-%m-%d")
            return str(x)

        # 4) Filter relevant instruments: name=NIFTY, expiry matches
        candidates = [
            i for i in cached_nfo_instruments
            if i.get("name") == base_name and _exp_str(i.get("expiry")) == expiry
        ]
        if not candidates:
            logger.error("[get_instrument_tokens] No instruments for %s @ %s", base_name, expiry)
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
            "spot_token": _resolve_spot_token_from_cache(cached_nse_instruments),
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
                        logger.info(
                            "[get_instrument_tokens] Found %s: %s (%s)",
                            side, inst.get("tradingsymbol"), inst.get("instrument_token")
                        )
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.warning("[get_instrument_tokens] No %s within ±%s*50 points", side, strike_range)

        # 6) Validation
        ok = any([results["ce_token"], results["pe_token"]])
        if not ok:
            logger.error("[get_instrument_tokens] No options found in range")
            return None

        # Log any adjustment from target
        for side in ("ce", "pe"):
            a = results["actual_strikes"].get(side)
            if a and a != target:
                logger.info("[get_instrument_tokens] %s strike adjusted: %s → %s", side.upper(), target, a)

        return results

    except Exception as e:
        logger.error("[get_instrument_tokens] Unexpected error: %s", e, exc_info=True)
        return None


def is_trading_hours() -> bool:
    """Simple NSE hours gate: 09:15–15:30 local time, Mon–Fri."""
    try:
        now = datetime.now()
        wd = now.weekday()
        start = datetime.strptime("09:15", "%H:%M").time()
        end = datetime.strptime("15:30", "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)
    except Exception as e:
        logger.error("[is_trading_hours] %s", e)
        return True  # fail-open to avoid unexpected blocking


def is_market_open(start_hhmm: str, end_hhmm: str) -> bool:
    """Configurable market-hours gate (HH:MM strings, local time)."""
    try:
        now = datetime.now()
        wd = now.weekday()
        start = datetime.strptime(start_hhmm, "%H:%M").time()
        end = datetime.strptime(end_hhmm, "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)
    except Exception as e:
        logger.error("[is_market_open] %s", e)
        return True


# ---------------- Diagnostics ----------------
def health_check(kite: Optional[KiteConnect]) -> Dict[str, Any]:
    """
    Lightweight readiness probe used by StrategyRunner/Application.
    Checks LTP reachability and validates instruments cache fetch.
    """
    status: Dict[str, Any] = {"overall_status": "OK", "message": "", "checks": {}}
    try:
        if not kite:
            status.update(overall_status="ERROR", message="No Kite instance")
            return status

        # LTP check
        spot_sym = _get_spot_ltp_symbol()
        try:
            ltp = _rate_limited_api_call(kite.ltp, [spot_sym])
            ok = bool(ltp.get(spot_sym, {}).get("last_price"))
            status["checks"]["ltp"] = "OK" if ok else "FAIL"
            if not ok:
                status["overall_status"] = "ERROR"
        except Exception as e:
            status["checks"]["ltp"] = f"FAIL: {e}"
            status["overall_status"] = "ERROR"

        # Instruments check
        try:
            nfo = _rate_limited_api_call(kite.instruments, "NFO")
            ok = isinstance(nfo, list) and len(nfo) > 0
            status["checks"]["instruments"] = "OK" if ok else "FAIL"
            if not ok:
                status["overall_status"] = "ERROR"
        except Exception as e:
            status["checks"]["instruments"] = f"FAIL: {e}"
            status["overall_status"] = "ERROR"

        status["message"] = " | ".join(f"{k}:{v}" for k, v in status["checks"].items())
        return status
    except Exception as e:
        logger.error("[health_check] %s", e, exc_info=True)
        return {"overall_status": "ERROR", "message": str(e), "checks": {}}
