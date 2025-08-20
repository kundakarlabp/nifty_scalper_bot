# src/utils/strike_selector.py
"""
Strike resolution helpers and market-time gates.

Clean signatures (no external spot args; no exchange-calendars).
- get_instrument_tokens(kite_instance=None) reads symbols from `settings`
- is_market_open() -> bool, pure IST gate 09:15–15:30 Mon–Fri
- All functions are safe when `kite_instance` is None (shadow mode).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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


def _rate_call(fn, call_key: str, *args, **kwargs) -> Any:
    """Helper to rate-limit a function call."""
    while _rate_limited(call_key):
        time.sleep(0.05)  # cooperative wait
    return fn(*args, **kwargs)


def is_market_open() -> bool:
    """
    Checks if the market is currently open in IST (Indian Standard Time).

    Trading hours are from 09:15 to 15:30, Monday to Friday.
    """
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    weekday = now.weekday()  # Monday is 0, Sunday is 6
    if weekday > 4:  # Saturday or Sunday
        return False
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start_time <= now < end_time


def _fetch_cached_instruments(kite: Optional[KiteConnect] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the full list of NFO instruments (options).
    Uses a simple rate-limiting wrapper. Safe when kite is None.
    """
    if not kite or kite is object:
        logger.warning("Kite instance missing; cannot fetch instruments.")
        return None

    try:
        call_key = "kite-instruments-nfo"
        instruments = _rate_call(kite.instruments, call_key, segment="NFO")
        return instruments
    except Exception as e:
        logger.exception("Failed to fetch instruments from Kite: %s", e)
        return None


def _resolve_nfo_weekly_expiry(nfo_instruments: List[Dict[str, Any]]) -> str:
    """
    Finds the nearest weekly expiry date for Nifty options.
    Excludes the current week only if today's Thursday has *passed* market hours.
    Returns date as 'YYYY-MM-DD' string; falls back to today if nothing found.
    """
    now = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()

    expiries = set()
    for row in nfo_instruments or []:
        # Look at CE rows to avoid duplicates (CE/PE share expiry)
        if row.get("instrument_type") == "CE" and row.get("expiry"):
            expiries.add(str(row["expiry"])[:10])

    if not expiries:
        return now.isoformat()

    sorted_expiries = sorted(expiries)
    for exp_str in sorted_expiries:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if exp_date >= now:
            return exp_str

    # If all are in the past (unexpected), return the last known
    return sorted_expiries[-1]


def _get_spot_ltp(kite: Optional[KiteConnect]) -> Optional[float]:
    """
    Fetch the current spot LTP via Kite. Safe if kite is None.
    """
    if not kite or kite is object:
        return None
    symbol = getattr(getattr(settings, "instruments", object()), "spot_symbol", "NSE:NIFTY 50")
    try:
        data = _rate_call(kite.ltp, "kite-ltp-spot", [symbol])
        if symbol in data:
            return float(data[symbol]["last_price"])
        # fallback: first value
        for _, v in data.items():
            return float(v["last_price"])
        return None
    except Exception as e:
        logger.warning("Failed to fetch spot LTP for %s: %s", symbol, e)
        return None


def get_instrument_tokens(kite_instance: Optional[KiteConnect] = None) -> Optional[Dict[str, Any]]:
    """
    Resolve spot token, ATM and target option strikes and tokens for the configured trade symbol.

    Returns:
        {
            "spot_token": int,
            "spot_price": float,
            "atm_strike": int,
            "target_strike": int,
            "expiry": "YYYY-MM-DD",
            "tokens": {"ce": Optional[int], "pe": Optional[int]},
        }
    """
    try:
        # --- read config with safe fallbacks ---
        inst = getattr(settings, "instruments", object())
        trade_symbol = str(getattr(inst, "trade_symbol", "NIFTY"))
        spot_token = int(getattr(inst, "instrument_token", 256265))  # NIFTY index token default
        step = 50  # NIFTY option strike step

        # Prefer new name; fallback to older ones
        strike_range = getattr(inst, "strike_range", None)
        if strike_range is None:
            strike_range = getattr(inst, "strike_selection_range", None)
        if strike_range is None:
            # legacy env or typo resilience
            strike_range = getattr(settings, "OPTION_RANGE", 3)
        try:
            strike_range = int(strike_range)
        except Exception:
            strike_range = 3

        # --- resolve spot price ---
        spot_price = _get_spot_ltp(kite_instance)
        if spot_price is None or spot_price <= 0:
            logger.warning("Spot LTP unavailable; cannot resolve ATM/target strikes.")
            return {
                "spot_token": spot_token,
                "spot_price": None,
                "atm_strike": None,
                "target_strike": None,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
            }

        # ATM rounding to nearest 50
        atm = int(round(spot_price / step) * step)
        target = int(atm + strike_range * step)

        # --- fetch instruments and resolve tokens ---
        nfo = _fetch_cached_instruments(kite_instance) or []
        if not nfo:
            logger.warning("NFO instruments unavailable; returning strike math without tokens.")
            return {
                "spot_token": spot_token,
                "spot_price": float(spot_price),
                "atm_strike": atm,
                "target_strike": target,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
            }

        expiry = _resolve_nfo_weekly_expiry(nfo)

        ce_token = None
        pe_token = None
        for row in nfo:
            if row.get("segment") != "NFO-OPT":
                continue
            if row.get("name") != trade_symbol:
                continue
            if not str(row.get("expiry", "")).startswith(expiry):
                continue
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
            "spot_token": int(spot_token),
            "spot_price": float(spot_price),
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
        "trade_symbol": getattr(getattr(settings, "instruments", object()), "trade_symbol", "NIFTY"),
    }
