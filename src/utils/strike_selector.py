# src/utils/strike_selector.py
"""
Strike resolution helpers and market-time gates.

Clean signatures (no exchange-calendars):
- is_market_open() -> bool, pure IST gate 09:15–15:30 Mon–Fri
- get_instrument_tokens(kite_instance=None, spot_price: float|None=None) -> dict|None
    Reads symbols from `settings.instruments`
    Returns a dict with ATM math and CE/PE tokens for the chosen target strike.

Safe in shadow mode (when kite_instance is None).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.config import settings

try:
    # Optional; only imported if installed
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = object  # type: ignore


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Static NSE holiday calendar (limited to years needed in tests/backtests)
# -----------------------------------------------------------------------------
# Source: https://www.nseindia.com/ (trading holidays)
_NSE_HOLIDAYS: dict[int, set[date]] = {
    2024: {
        date(2024, 1, 26),  # Republic Day
        date(2024, 3, 8),  # Mahashivratri
        date(2024, 3, 25),  # Holi
        date(2024, 3, 29),  # Good Friday
        date(2024, 4, 8),  # Gudi Padwa
        date(2024, 4, 11),  # Id-Ul-Fitr
        date(2024, 5, 1),  # Maharashtra Day
        date(2024, 6, 17),  # Bakri Id
        date(2024, 7, 17),  # Muharram
        date(2024, 8, 15),  # Independence Day
        date(2024, 10, 2),  # Gandhi Jayanti
        date(2024, 11, 1),  # Diwali (Laxmi Puja)
        date(2024, 11, 15),  # Gurunanak Jayanti
        date(2024, 12, 25),  # Christmas
    }
}


def _is_nse_holiday(dt: datetime) -> bool:
    """Return True if the given datetime falls on an NSE trading holiday."""
    return dt.date() in _NSE_HOLIDAYS.get(dt.year, set())

# -----------------------------------------------------------------------------
# Module-level rate limiting and small caches
# -----------------------------------------------------------------------------
_last_api_call: Dict[str, float] = {}
_last_api_lock = threading.Lock()

_instruments_cache: Optional[List[Dict[str, Any]]] = None
_instruments_cache_ts: float = 0.0
_INSTRUMENTS_TTL = 60.0  # seconds

_ltp_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, ts)
_LTP_TTL = 2.0  # seconds


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


# -----------------------------------------------------------------------------
# Pure time gate (IST)
# -----------------------------------------------------------------------------
def is_market_open() -> bool:
    """
    Checks if the market is currently open in IST (Indian Standard Time).
    Trading hours are from 09:15 to 15:30, Monday to Friday, excluding NSE holidays.
    """
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    if _is_nse_holiday(now):  # NSE holiday
        return False
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start_time <= now < end_time


# -----------------------------------------------------------------------------
# Data fetchers (safe when kite is None)
# -----------------------------------------------------------------------------
def _fetch_instruments_nfo(kite: Optional[KiteConnect]) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the full list of NFO instruments (options) with a short cache & rate limit.
    """
    global _instruments_cache, _instruments_cache_ts
    now = time.time()

    # Return cached copy if fresh enough
    if _instruments_cache is not None and (now - _instruments_cache_ts) < _INSTRUMENTS_TTL:
        return _instruments_cache

    if not kite or kite is object:
        logger.debug("Kite instance missing; cannot fetch instruments (shadow mode).")
        return None

    try:
        call_key = "kite-instruments-nfo"
        # Correct signature for Kite: instruments(exchange)
        instruments = _rate_call(kite.instruments, call_key, "NFO")
        if isinstance(instruments, list) and instruments:
            _instruments_cache = instruments
            _instruments_cache_ts = now
            return instruments
        return None
    except Exception as e:
        logger.warning("Failed to fetch instruments from Kite: %s", e)
        return None


def _get_spot_ltp(kite: Optional[KiteConnect], symbol: str) -> Optional[float]:
    """
    Fetch the current spot LTP via Kite with a tiny cache. Safe if kite is None.
    """
    now = time.time()

    # cache hit?
    cached = _ltp_cache.get(symbol)
    if cached and (now - cached[1]) < _LTP_TTL:
        return float(cached[0])

    if not kite or kite is object:
        return None

    try:
        data = _rate_call(kite.ltp, f"kite-ltp-{symbol}", [symbol])
        if symbol in data:
            px = float(data[symbol]["last_price"])
        else:
            # fallback: first value
            (px,) = [float(v["last_price"]) for v in data.values()][0:1] or (None,)
        if px is not None:
            _ltp_cache[symbol] = (px, now)
            return px
        return None
    except Exception as e:
        logger.warning("Failed to fetch spot LTP for %s: %s", symbol, e)
        return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _infer_step(trade_symbol: str) -> int:
    """
    Infer option strike step. Override via settings.instruments.strike_step if present.
    Defaults:
      - NIFTY / FINNIFTY: 50
      - BANKNIFTY: 100
      - else: 50
    """
    inst = getattr(settings, "instruments", object())
    step = getattr(inst, "strike_step", None)
    if step:
        try:
            return int(step)
        except Exception:
            pass

    s = (trade_symbol or "").upper()
    if "BANKNIFTY" in s:
        return 100
    return 50  # NIFTY / FINNIFTY / default


def _resolve_weekly_expiry_from_dump(nfo_instruments: List[Dict[str, Any]], trade_symbol: str) -> Optional[str]:
    """
    From the live instrument dump, collect expiries for the given name/symbol and
    pick the nearest one >= today (IST). Return 'YYYY-MM-DD' or None.
    """
    if not nfo_instruments:
        return None

    ist_today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    expiries: set[str] = set()

    for row in nfo_instruments:
        try:
            if row.get("segment") != "NFO-OPT":
                continue
            if row.get("name") != trade_symbol:
                continue
            expiry = row.get("expiry")
            if not expiry:
                continue
            expiries.add(str(expiry)[:10])
        except Exception:
            continue

    if not expiries:
        return None

    sorted_exp = sorted(expiries)
    for exp in sorted_exp:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= ist_today:
            return exp
    # else, all past—return last known (unexpected but better than None)
    return sorted_exp[-1]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_instrument_tokens(
    kite_instance: Optional[KiteConnect] = None,
    spot_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve spot token, ATM and target option strikes and tokens for the configured trade symbol.

    Args:
        kite_instance: optional KiteConnect instance; if None, runs in shadow mode.
        spot_price: optionally pass in a spot LTP (otherwise we'll fetch from Kite).

    Returns:
        {
            "spot_token": int,
            "spot_price": float | None,
            "atm_strike": int | None,
            "target_strike": int | None,
            "expiry": str | None,  # 'YYYY-MM-DD'
            "tokens": {"ce": Optional[int], "pe": Optional[int]},
        }
        or None on unrecoverable failure.
    """
    try:
        # --- read config with safe fallbacks ---
        inst = getattr(settings, "instruments", object())
        trade_symbol = str(getattr(inst, "trade_symbol", "NIFTY"))
        spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
        spot_token = int(getattr(inst, "instrument_token", 256265))  # NIFTY index token default

        # strike step and target offset
        step = _infer_step(trade_symbol)
        strike_range = getattr(inst, "strike_range", None)
        if strike_range is None:
            strike_range = getattr(inst, "strike_selection_range", None)
        if strike_range is None:
            strike_range = getattr(settings, "OPTION_RANGE", 0)  # default ATM-only if not set
        try:
            strike_range = int(strike_range)
        except Exception:
            strike_range = 0

        # --- resolve spot price ---
        px = None
        if spot_price is not None:
            try:
                px = float(spot_price)
            except Exception:
                px = None
        if (px is None or px <= 0.0):
            px = _get_spot_ltp(kite_instance, spot_symbol)

        # Without a spot price, we can still return static info
        if px is None or px <= 0.0:
            logger.debug("Spot LTP unavailable; returning minimal structure without strikes/tokens.")
            return {
                "spot_token": spot_token,
                "spot_price": None,
                "atm_strike": None,
                "target_strike": None,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
            }

        # --- ATM rounding and target selection ---
        atm = int(round(px / step) * step)
        target = int(atm + strike_range * step)

        # --- fetch instruments and resolve CE/PE tokens (shadow-safe) ---
        nfo = _fetch_instruments_nfo(kite_instance) or []
        if not nfo:
            logger.debug("NFO instruments unavailable; returning strike math without tokens.")
            return {
                "spot_token": spot_token,
                "spot_price": float(px),
                "atm_strike": atm,
                "target_strike": target,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
            }

        expiry = _resolve_weekly_expiry_from_dump(nfo, trade_symbol)
        ce_token = None
        pe_token = None

        for row in nfo:
            try:
                if row.get("segment") != "NFO-OPT":
                    continue
                if row.get("name") != trade_symbol:
                    continue
                if expiry and not str(row.get("expiry", "")).startswith(expiry):
                    continue
                strike = int(row.get("strike"))
                if strike != target:
                    continue
                itype = row.get("instrument_type")
                if itype == "CE":
                    ce_token = row.get("instrument_token")
                elif itype == "PE":
                    pe_token = row.get("instrument_token")
            except Exception:
                continue

        return {
            "spot_token": int(spot_token),
            "spot_price": float(px),
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