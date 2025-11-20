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
    default = {i: str(i) for i in range(1,10)}
    default.update({10: 'O', 11: 'N', 12: 'D'})

    # attempt inference
    try:
        mapping = {}
        for key, inst in instrument_map.items():
            ts = inst.get("tradingsymbol") or ""
            # Heuristic: find patterns like 'NIFTY25N25...' -> extract single-letter month code
            if ts.startswith(UNDERLYING) and len(ts) > len(UNDERLYING) + 4:
                # skip if contains month abbrev like 'NOV'
                tail = ts[len(UNDERLYING):]
                yy = tail[:2]
                maybe = tail[2:]  # e.g., 'N25...'
                # if maybe begins with a single non-digit -> candidate
                if maybe and not maybe[0].isdigit():
                    code = maybe[0]
                    # We can't reliably map code->month without more heuristics; skip heavy inference
                    # Return default for safety
                    return default
        return default
    except Exception:
        return default

def generate_candidate_symbols(expiry_date: date, strike: int, opt_type: str, month_map: Dict[int,str]) -> List[str]:
    yy = str(expiry_date.year)[-2:]
    dd = f"{expiry_date.day:02d}"
    m_code = month_map.get(expiry_date.month, str(expiry_date.month))
    candidates = []
    # weekly single-char + day
    candidates.append(f"{UNDERLYING}{yy}{m_code}{dd}{strike}{opt_type}")
    # monthly 3-letter + day
    candidates.append(f"{UNDERLYING}{yy}{expiry_date.strftime('%b').upper()}{dd}{strike}{opt_type}")
    # monthly without day (common monthly style)
    candidates.append(f"{UNDERLYING}{yy}{expiry_date.strftime('%b').upper()}{strike}{opt_type}")
    return candidates

def resolve_symbol_from_master(candidates: List[str], instrument_map: Dict[str, dict]) -> Optional[dict]:
    for c in candidates:
        for prefix in (f"{EXCHANGE_PREFIX}:", ""):
            key = prefix + c
            inst = instrument_map.get(key)
            if inst:
                return inst
    return None

def get_next_valid_symbols(strikes: List[int], opt_types=('CE','PE'), instrument_map: Dict[str, dict]) -> List[dict]:
    today = now_ist().date()
    expiry = next_weekday(today, WEEKLY_EXPIRY_WEEKDAY)
    month_map = infer_month_code_from_master(instrument_map)
    valid = []
    for strike in strikes:
        for ot in opt_types:
            cands = generate_candidate_symbols(expiry, strike, ot, month_map)
            inst = resolve_symbol_from_master(cands, instrument_map)
            if inst:
                valid.append(inst)
            else:
                logger.debug("No instrument match for expiry=%s strike=%s opt=%s candidates=%s", expiry, strike, ot, cands)
    return valid
