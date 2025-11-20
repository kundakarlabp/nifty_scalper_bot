# src/nifty_scalper_bot/utils/smart_symbol.py
"""
Utilities to generate and validate candidate NIFTY option tradingsymbols.

Provides:
- get_next_valid_symbols(strikes, instrument_map, opt_types=('CE','PE')) -> List[dict]
- generate_candidate_symbols(expiry_date, strike, opt_type, month_map) -> List[str]

Notes:
- Weekly format: SYMBOL + YY + M + DD + STRIKE + TYPE  (e.g. NIFTY25N2524000CE)
  where M is single-char month code: 1-9 / O / N / D
- Monthly format: SYMBOL + YY + MMM + STRIKE + TYPE  (e.g. NIFTY25NOV24000CE)
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo  # Python 3.9+
from typing import Dict, List, Optional, Sequence, Tuple, Iterable
import logging

logger = logging.getLogger(__name__)

# Constants / defaults
EXCHANGE_PREFIX = "NFO"
UNDERLYING = "NIFTY"
WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (0=Mon)
IST = ZoneInfo("Asia/Kolkata")

# Default month code map for weekly single-char codes
_DEFAULT_MONTH_MAP: Dict[int, str] = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "O",
    11: "N",
    12: "D",
}


def now_ist() -> datetime:
    """Return current time in IST timezone-aware datetime."""
    return datetime.now(IST)


def next_weekday(start_date: date, target_weekday: int) -> date:
    """
    Return the next target_weekday after start_date.
    If start_date is target_weekday, returns the next week's day (never returns today).
    """
    days_ahead = (target_weekday - start_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return start_date + timedelta(days=days_ahead)


def infer_month_code_from_master(instrument_map: Dict[str, dict]) -> Dict[int, str]:
    """
    Attempt to infer month-code mapping from an instrument master by scanning
    tradingsymbol strings. If inference is not possible or fails, return default map.
    This is conservative: we prefer default mapping if heuristics are uncertain.
    """
    if not instrument_map:
        return dict(_DEFAULT_MONTH_MAP)

    try:
        # Very lightweight heuristic: if master contains weekly-style (single-char) symbols,
        # prefer default mapping because robust inference requires many examples.
        for key, inst in instrument_map.items():
            ts = (inst.get("tradingsymbol") or inst.get("tradingsymbol", "") or "")
            if not ts or not ts.startswith(UNDERLYING):
                continue
            tail = ts[len(UNDERLYING):]
            # Expect at least YY + code + DD ...
            if len(tail) >= 5:
                yy = tail[:2]
                maybe = tail[2:]
                # if the third char is a non-digit (likely single-char month code), prefer default
                if maybe and not maybe[0].isdigit():
                    # We don't attempt map resolution here; default is safe and standard.
                    return dict(_DEFAULT_MONTH_MAP)
        return dict(_DEFAULT_MONTH_MAP)
    except Exception:
        return dict(_DEFAULT_MONTH_MAP)


def generate_candidate_symbols(
    expiry_date: date,
    strike: int,
    opt_type: str,
    month_map: Optional[Dict[int, str]] = None,
    underlying: str = UNDERLYING,
) -> List[str]:
    """
    Generate a list of candidate tradingsymbol strings for a given expiry, strike and option type.
    Order is: weekly (single-char + dd), monthly with day (MMM + DD), monthly without day (MMM).
    """
    if month_map is None:
        month_map = dict(_DEFAULT_MONTH_MAP)

    yy = str(expiry_date.year)[-2:]
    dd = f"{expiry_date.day:02d}"
    m_code = month_map.get(expiry_date.month, str(expiry_date.month))
    opt = str(opt_type).upper()

    candidates: List[str] = []
    # weekly: single-char month + day
    try:
        candidates.append(f"{underlying}{yy}{m_code}{dd}{strike}{opt}")
    except Exception:
        pass
    # monthly: MMM + day (sometimes used)
    try:
        candidates.append(f"{underlying}{yy}{expiry_date.strftime('%b').upper()}{dd}{strike}{opt}")
    except Exception:
        pass
    # monthly: MMM without day (common monthly representation)
    try:
        candidates.append(f"{underlying}{yy}{expiry_date.strftime('%b').upper()}{strike}{opt}")
    except Exception:
        pass

    # Ensure unique order-preserving
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve_symbol_from_master(candidates: Iterable[str], instrument_map: Dict[str, dict]) -> Optional[dict]:
    """
    Given candidate tradingsymbols, attempt to find a matching instrument entry in instrument_map.
    Tries both "NFO:SYMBOL" and "SYMBOL" keys (case-sensitive and uppercase fallback).
    Returns the first matching instrument dict or None.
    """
    if not instrument_map:
        return None
    for c in candidates:
        for prefix in (f"{EXCHANGE_PREFIX}:", ""):
            key = prefix + c
            inst = instrument_map.get(key)
            if inst:
                return inst
            # try uppercase fallback keys some resolvers use
            inst = instrument_map.get(key.upper())
            if inst:
                return inst
    return None


def get_next_valid_symbols(
    strikes: Sequence[int],
    instrument_map: Dict[str, dict],
    opt_types: Iterable[str] = ("CE", "PE"),
    underlying: str = UNDERLYING,
) -> List[dict]:
    """
    For the next weekly expiry (following the Tuesday rule), generate candidate symbols for each strike
    and option type, validate them against instrument_map, and return a list of resolved instrument dicts.

    Args:
        strikes: sequence of strike integers to generate (e.g., [24000, 24100])
        instrument_map: mapping keyed by either 'NFO:TRADINGSYMBOL' or 'TRADINGSYMBOL'
                        values are instrument metadata dicts (token, lot_size, tradingsymbol, etc.)
        opt_types: iterable of option types to consider (default ('CE','PE'))
        underlying: underlying symbol string (default 'NIFTY')

    Returns:
        List of resolved instrument dicts (entries pulled from instrument_map). Order follows strikes/opt_types.
    """

    try:
        today = now_ist().date()
    except Exception:
        today = datetime.utcnow().date()

    expiry = next_weekday(today, WEEKLY_EXPIRY_WEEKDAY)
    month_map = infer_month_code_from_master(instrument_map)

    resolved_list: List[dict] = []

    # If expiry is last Tuesday of the month, monthly format may be primary — but we still try both.
    # We simply generate candidates and attempt lookups in priority order.
    for strike in strikes:
        for opt in opt_types:
            cands = generate_candidate_symbols(expiry, int(strike), str(opt), month_map, underlying)
            inst = resolve_symbol_from_master(cands, instrument_map)
            if inst:
                resolved_list.append(inst)
            else:
                logger.debug(
                    "No instrument match for expiry=%s strike=%s opt=%s candidates=%s",
                    expiry,
                    strike,
                    opt,
                    cands,
                )

    return resolved_list
