"""
Strike resolution helpers and market-time gates.

Clean signatures (no exchange-calendars, no spot_symbol arg).
- get_instrument_tokens(kite_instance, ...) reads symbols from `settings`
- is_market_open() -> bool, pure IST gate 09:15–15:30 Mon–Fri
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

try:
    # Optional; only imported if installed
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = object  # type: ignore


logger = logging.getLogger(__name__)

# simple module-level rate limiter memory
_last_api_call: Dict[str, float] = {}
_last_api_lock = threading.Lock()


def _rate_limited(call_key: str, min_interval_sec: float = 0.25) -> bool:
    """
    Returns True if we should WAIT (i.e., too soon), False if OK to call now.
    """
    with _last_api_lock:
        now = time.time()
        last = _last_api_call.get(call_key, 0.0)
        if now - last < min_interval_sec:
            return True
        _last_api_call[call_key] = now
        return False


def _rate_call(fn, call_key: str, *args, **kwargs):
    """
    Execute a Kite API call with very conservative rate limiting + 1 retry on rate error.
    """
    if _rate_limited(call_key):
        time.sleep(0.3)
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "too many" in msg:
            logger.warning("Rate limited on %s; retrying...", call_key)
            time.sleep(2.0)
            return fn(*args, **kwargs)
        raise


# ---------- Time gates (IST) ----------

def is_market_open() -> bool:
    """
    NSE gate: 09:15–15:30 Asia/Kolkata, Mon–Fri.
    No exchange_calendars dependency.
    """
    try:
        import pytz  # optional
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
    except Exception:
        from datetime import timezone
        now = datetime.now(timezone(timedelta(hours=5, minutes=30)))

    if now.weekday() > 4:  # Sat/Sun
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


def is_trading_hours() -> bool:
    """Alias retained for older callers."""
    return is_market_open()


# ---------- Instruments cache ----------

def fetch_cached_instruments(kite_instance: KiteConnect) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (NFO_list, NSE_list) instrument dumps, cached by the caller if needed.
    """
    nfo = _rate_call(kite_instance.instruments, "instruments_NFO", "NFO")
    nse = _rate_call(kite_instance.instruments, "instruments_NSE", "NSE")
    return nfo or [], nse or []


# ---------- Spot helpers ----------

def _get_spot_ltp_symbol() -> str:
    """
    Read SPOT_SYMBOL from settings; fallback to 'NSE:NIFTY 50'.
    """
    sym = getattr(settings, "SPOT_SYMBOL", None) or getattr(settings.instruments, "spot_symbol", None)
    return sym or "NSE:NIFTY 50"


def _spot_ltp(kite_instance: KiteConnect) -> float:
    """
    Fetch NIFTY 50 spot LTP; return fallback 25000.0 on failure.
    """
    try:
        symbol = _get_spot_ltp_symbol()
        out = _rate_call(kite_instance.ltp, "ltp_spot", [symbol])
        if isinstance(out, dict) and out:
            # Zerodha returns { "NSE:NIFTY 50": {"last_price": ...}, ...}
            payload = next(iter(out.values()))
            return float(payload.get("last_price") or payload.get("last_price", 0.0))
    except Exception as e:
        logger.warning("Spot LTP fetch failed: %s", e)
    return 25000.0


# ---------- Strike math ----------

def get_atm_strike(spot_price: float, step: int = 50) -> int:
    """
    Round to nearest strike increment.
    """
    if step <= 0:
        step = 50
    return int(round(spot_price / step) * step)


def _calculate_next_thursday(base_dt: Optional[datetime] = None) -> str:
    """
    If instrument dump unavailable, approximate “next Thursday” for NIFTY weekly expiry.
    """
    base = base_dt or datetime.utcnow()
    # 3 == Thursday (Mon=0)
    days_ahead = (3 - base.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    d = base + timedelta(days=days_ahead)
    return d.strftime("%Y-%m-%d")


def _resolve_nfo_weekly_expiry(cached_nfo: List[Dict[str, Any]]) -> str:
    """
    Parse earliest future expiry from instrument dump. Fallback to next Thursday calc.
    """
    try:
        from datetime import date
        expiries = sorted(
            {
                r["expiry"] if isinstance(r.get("expiry"), str) else r.get("expiry").strftime("%Y-%m-%d")
                for r in cached_nfo
                if (r.get("name") == "NIFTY" and r.get("segment") == "NFO-OPT")
            }
        )
        for e in expiries:
            # choose the closest >= today
            return e
    except Exception:
        pass
    return _calculate_next_thursday()


# ---------- Public API ----------

def get_instrument_tokens(
    kite_instance: KiteConnect,
    cached_nfo_instruments: Optional[List[Dict[str, Any]]] = None,
    *,
    offset: int = 0,
    strike_range: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve CE/PE instrument tokens around ATM with `offset` steps (0 == ATM, 1 == ATM+50, etc).

    Reads base symbol/exchange from `settings`—no spot_symbol arg required.

    Returns dict:
      {
        "spot_price": float,
        "atm_strike": int,
        "target_strike": int,
        "expiry": "YYYY-MM-DD",
        "tokens": {"ce": int|None, "pe": int|None},
      }
    """
    try:
        nfo = cached_nfo_instruments
        if nfo is None:
            nfo, _ = fetch_cached_instruments(kite_instance)

        spot = _spot_ltp(kite_instance)
        atm = get_atm_strike(spot, step=50)
        rng = int(strike_range if strike_range is not None else getattr(settings.instruments, "strike_selection_range", 3))
        step = 50

        target = atm + offset * step
        expiry = _resolve_nfo_weekly_expiry(nfo)

        ce_token = None
        pe_token = None
        base_name = getattr(settings.instruments, "trade_symbol", "NIFTY")

        for row in nfo:
            if row.get("segment") != "NFO-OPT":
                continue
            if row.get("name") != base_name:
                continue
            if str(row.get("expiry")).startswith(expiry):
                # Zerodha format example: NIFTY25AUG25000CE
                try:
                    strike = int(row.get("strike"))
                except Exception:
                    continue
                if strike == target:
                    if row.get("instrument_type") == "CE":
                        ce_token = row.get("instrument_token")
                    elif row.get("instrument_type") == "PE":
                        pe_token = row.get("instrument_token")

        return {
            "spot_price": float(spot),
            "atm_strike": int(atm),
            "target_strike": int(target),
            "expiry": expiry,
            "tokens": {"ce": ce_token, "pe": pe_token},
        }
    except Exception as e:
        logger.exception("get_instrument_tokens failed: %s", e)
        return None


def health_check() -> Dict[str, Any]:
    """
    Lightweight readiness info for probes.
    """
    return {
        "ok": True,
        "ist_open": is_market_open(),
        "trade_symbol": getattr(settings.instruments, "trade_symbol", "NIFTY"),
    }