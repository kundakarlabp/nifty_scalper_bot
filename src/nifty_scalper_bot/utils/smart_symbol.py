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

# Add NSE market holidays (Update yearly)
NSE_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 20),   # Id-ul-Fitr 
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti (TUESDAY HOLIDAY)
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
}

def now_ist() -> datetime:
    return datetime.now(IST)

def get_actual_expiry_date(start_date: date, target_weekday: int) -> date:
    """Calculates expiry date, shifting backwards if it falls on a holiday or weekend."""
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    expiry = start_date + timedelta(days=days_ahead)
    
    # Shift to previous trading day if the planned expiry is a holiday or weekend
    while expiry in NSE_HOLIDAYS or expiry.weekday() >= 5:
        expiry -= timedelta(days=1)
        
    return expiry

def next_weekday(start_date: date, target_weekday: int) -> date:
    # Overriding the original function to use the holiday-adjusted date
    # so all downstream symbol generation functions automatically get the right day.
    return get_actual_expiry_date(start_date, target_weekday)



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
    for the NEXT available expiry matching the requested strikes & option types.
    (Dynamically scans available options instead of guessing dates).
    """
    if not instrument_map:
        return []
        
    today = now_ist().date()
    
    # 1. Gather all valid future NIFTY options matching our strikes & types
    available_options = []
    for key, inst in instrument_map.items():
        ts = str(inst.get("tradingsymbol", "")).upper()
        if not ts.startswith(UNDERLYING): continue
        if not any(ts.endswith(ot) for ot in opt_types): continue
        
        # Safely parse and match strike
        strike_val = inst.get("strike")
        if strike_val is None: continue
        try:
            if int(float(strike_val)) not in strikes: continue
        except (ValueError, TypeError):
            continue
            
        # Safely parse expiry date
        exp_str = inst.get("expiry")
        if not exp_str: continue
        try:
            # Handle standard YYYY-MM-DD formats from broker dumps
            raw_date = str(exp_str).split("T")[0][:10]
            exp_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except Exception:
            continue
            
        # Keep if the expiry is today or in the future
        if exp_date >= today:
            available_options.append((exp_date, inst))
            
    if not available_options:
        logger.debug(f"No available {UNDERLYING} options found for strikes {strikes} >= {today}")
        return []
        
    # 2. Sort by date and find the absolute closest expiry
    available_options.sort(key=lambda x: x[0])
    nearest_expiry_date = available_options[0][0]
    
    # 3. Return only the unique options that match this nearest date
    valid = []
    seen_tokens = set() # Prevents duplicates if map has NFO:SYMBOL and SYMBOL keys
    
    for exp_date, inst in available_options:
        if exp_date == nearest_expiry_date:
            token = inst.get("instrument_token")
            if token not in seen_tokens:
                valid.append(inst)
                seen_tokens.add(token)
                
    return valid


# Backwards compatible alias required by older imports
def generate_candidate_symbols_for_expiry(expiry_date: date, strike: int, opt_type: str, month_map: Dict[int, str]) -> List[str]:
    return generate_candidate_symbols(expiry_date, strike, opt_type, month_map)
