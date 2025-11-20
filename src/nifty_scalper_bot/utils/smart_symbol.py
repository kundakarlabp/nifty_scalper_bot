# src/nifty_scalper_bot/utils/smart_symbol.py
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo  # py3.9+; alternative: pytz
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)
EXCHANGE_PREFIX = "NFO"
UNDERLYING = "NIFTY"
WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (0=Mon)

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)


def next_weekday(start_date: date, target_weekday: int) -> date:
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return start_date + timedelta(days=days_ahead)


def infer_month_code_from_master(instrument_map: Dict[str, dict]) -> Dict[int, str]:
    """
    Attempt to infer month-code mapping from an instrument master by scanning
    weekly tradingsymbols. Fallback to default mapping if not possible.
    Returns mapping: month(int) -> code(str)
    """
    default = {i: str(i) for i in range(1, 10)}
    default.update({10: "O", 11: "N", 12: "D"})

    # Try a very cheap heuristic scan; if anything odd, return default (safe).
    try:
        for key, inst in (instrument_map or {}).items():
            ts = (inst.get("tradingsymbol") or inst.get("symbol") or "").upper()
            if not ts.startswith(UNDERLYING):
                continue
            tail = ts[len(UNDERLYING) :]
            # e.g. tail might be "25N25..." or "25NOV25..." etc.
            if len(tail) >= 3 and not tail[0].isdigit():
                # we see a non-digit char at month position; probably single-char code present
                # but mapping code->month isn't reliably recoverable here; return default
                return default
    except Exception:
        # if anything goes wrong, return default mapping
        return default

    return default


def generate_candidate_symbols(expiry_date: date, strike: int, opt_type: str, month_map: Dict[int, str]) -> List[str]:
    """
    Given an expiry date, strike and option type ("CE"/"PE"), produce candidate tradingsymbols
    that cover the common weekly and monthly naming variants:
      - Weekly style: NIFTY{yy}{M}{DD}{strike}{CE/PE}    (M = single-char month code)
      - Monthly style (with day): NIFTY{yy}{MON}{DD}{strike}{CE/PE}
      - Monthly style (without day): NIFTY{yy}{MON}{strike}{CE/PE}
    Returns list of candidate strings (no "NFO:" prefix).
    """
    yy = str(expiry_date.year)[-2:]
    dd = f"{expiry_date.day:02d}"
    m_code = month_map.get(expiry_date.month, str(expiry_date.month))
    candidates = []
    # weekly single-char + day (most compact weekly style)
    candidates.append(f"{UNDERLYING}{yy}{m_code}{dd}{strike}{opt_type}")
    # monthly 3-letter + day (explicit monthly style)
    candidates.append(f"{UNDERLYING}{yy}{expiry_date.strftime('%b').upper()}{dd}{strike}{opt_type}")
    # monthly without day (alternative monthly style)
    candidates.append(f"{UNDERLYING}{yy}{expiry_date.strftime('%b').upper()}{strike}{opt_type}")
    # also include bare numeric-day variant if strikes include trailing zeros etc
    return candidates


def resolve_symbol_from_master(candidates: List[str], instrument_map: Dict[str, dict]) -> Optional[dict]:
    for c in candidates:
        for prefix in (f"{EXCHANGE_PREFIX}:", ""):
            key = prefix + c
            inst = instrument_map.get(key)
            if inst:
                return inst
    return None


def get_next_valid_symbols(strikes: List[int], opt_types=("CE", "PE"), instrument_map: Dict[str, dict] = None) -> List[dict]:
    """
    Primary convenience function used by runner: return list of instrument metadata
    for the NEXT weekly expiry matching the requested strikes & option types.

    - strikes: list of strike integers (e.g., [24000, 24100])
    - opt_types: tuple/list of "CE"/"PE"
    - instrument_map: the instrument master mapping loaded by the resolver/datahub; it's okay if None
    """
    today = now_ist().date()
    expiry = next_weekday(today, WEEKLY_EXPIRY_WEEKDAY)
    month_map = infer_month_code_from_master(instrument_map or {})
    valid = []
    for strike in strikes:
        for ot in (opt_types or ("CE", "PE")):
            cands = generate_candidate_symbols(expiry, strike, ot, month_map)
            inst = resolve_symbol_from_master(cands, instrument_map or {})
            if inst:
                valid.append(inst)
            else:
                logger.debug(
                    "No instrument match for expiry=%s strike=%s opt=%s candidates=%s",
                    expiry,
                    strike,
                    ot,
                    cands,
                )
    return valid


# Backwards compatible alias required by older imports
def generate_candidate_symbols_for_expiry(expiry_date: date, strike: int, opt_type: str, month_map: Dict[int, str]) -> List[str]:
    return generate_candidate_symbols(expiry_date, strike, opt_type, month_map)
