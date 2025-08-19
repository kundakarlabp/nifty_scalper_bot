# src/utils/strike_selector.py
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "_get_spot_ltp_symbol",
    "format_option_symbol",
    "get_atm_strike_price",
    "get_nearest_strikes",
    "get_next_expiry_date",
    "fetch_cached_instruments",
    "get_instrument_tokens",
    "is_trading_hours",
    "health_check",
]

_last_api_call: Dict[str, float] = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # seconds


def _rate_limited_api_call(func, *args, **kwargs):
    with _api_call_lock:
        key = getattr(func, "__name__", "api_call")
        now = time.time()
        prev = _last_api_call.get(key, 0.0)
        if now - prev < _MIN_API_INTERVAL:
            time.sleep(_MIN_API_INTERVAL - (now - prev))
        try:
            res = func(*args, **kwargs)
            _last_api_call[key] = time.time()
            return res
        except Exception as e:
            if any(s in str(e).lower() for s in ("rate", "too many")):
                logger.warning("Rate-limited on %s, retrying…", key)
                time.sleep(2)
                res = func(*args, **kwargs)
                _last_api_call[key] = time.time()
                return res
            raise


def _get_spot_ltp_symbol() -> str:
    try:
        from src.config import Config
        sym = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        return sym or "NSE:NIFTY 50"
    except Exception:
        return "NSE:NIFTY 50"


def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    try:
        dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        return dt.strftime("%y%b%d").upper()
    except Exception as e:
        logger.error("format expiry failed: %s", e)
        return ""


def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    exp = _format_expiry_for_symbol_primary(expiry)
    return f"{base_symbol}{exp}{int(strike)}{option_type}" if exp else ""


def get_atm_strike_price(spot_price: float) -> int:
    try:
        return int(round(float(spot_price) / 50.0) * 50)
    except Exception:
        return 24500


def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    atm = get_atm_strike_price(spot_price)
    half = max(1, strike_count // 2)
    return sorted(set(atm + i * 50 for i in range(-half, half + 1)))


def _calculate_next_thursday(d: Optional[date] = None) -> str:
    d = d or date.today()
    days = (3 - d.weekday()) % 7
    if days == 0:
        days = 7
    return (d + timedelta(days=days)).isoformat()


def fetch_cached_instruments(kite: KiteConnect) -> Dict[str, List[Dict[str, Any]]]:
    try:
        nfo = _rate_limited_api_call(kite.instruments, "NFO")
    except Exception as e:
        logger.error("NFO instruments fetch failed: %s", e)
        nfo = []
    try:
        nse = _rate_limited_api_call(kite.instruments, "NSE")
    except Exception as e:
        logger.error("NSE instruments fetch failed: %s", e)
        nse = []
    return {"NFO": nfo or [], "NSE": nse or []}


def get_next_expiry_date(kite_instance: KiteConnect, cached_nfo_instruments: Optional[List[Dict]] = None) -> Optional[str]:
    if not kite_instance:
        return _calculate_next_thursday()
    try:
        if cached_nfo_instruments is None:
            cached_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
        base = [i for i in (cached_nfo_instruments or []) if i.get("name") == "NIFTY"]
        if not base:
            return _calculate_next_thursday()

        cands: set[date] = set()
        for inst in base:
            exp = inst.get("expiry")
            if isinstance(exp, datetime):
                exp_d = exp.date()
            elif isinstance(exp, date):
                exp_d = exp
            else:
                try:
                    y, m, d = map(int, str(exp)[:10].split("-"))
                    exp_d = date(y, m, d)
                except Exception:
                    continue
            if exp_d >= date.today():
                cands.add(exp_d)
        return (min(cands).isoformat() if cands else _calculate_next_thursday())
    except Exception as e:
        logger.warning("get_next_expiry_date fallback: %s", e)
        return _calculate_next_thursday()


def _resolve_spot_token_from_cache(cached_nse_instruments: List[Dict]) -> Optional[int]:
    try:
        from src.config import Config
        for inst in cached_nse_instruments or []:
            ts = (inst.get("tradingsymbol") or "").strip().upper()
            seg = (inst.get("segment") or "").upper()
            if ts == "NIFTY 50" and "INDICE" in seg:
                tok = inst.get("instrument_token")
                if tok:
                    return int(tok)
        return int(getattr(Config, "INSTRUMENT_TOKEN", 256265))
    except Exception:
        return None


def get_instrument_tokens(
    symbol: str,
    kite_instance: KiteConnect,
    cached_nfo_instruments: List[Dict],
    cached_nse_instruments: List[Dict],
    offset: int = 0,
    strike_range: int = 3,
) -> Optional[Dict[str, Any]]:
    if not (kite_instance and cached_nfo_instruments and cached_nse_instruments):
        return None
    try:
        spot_key = _get_spot_ltp_symbol()
        spot = _rate_limited_api_call(kite_instance.ltp, [spot_key])
        spot_price = float(spot.get(spot_key, {}).get("last_price") or 0.0)
        if not spot_price:
            return None

        atm = get_atm_strike_price(spot_price)
        target = atm + int(offset) * 50
        expiry = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry:
            return None

        def _exp_str(x) -> str:
            return x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)

        cands = [i for i in cached_nfo_instruments if i.get("name") == "NIFTY" and _exp_str(i.get("expiry")) == expiry]
        if not cands:
            return None

        res: Dict[str, Any] = {
            "spot_price": spot_price,
            "atm_strike": atm,
            "target_strike": target,
            "offset": int(offset),
            "actual_strikes": {},
            "expiry": expiry,
            "ce_symbol": None, "ce_token": None,
            "pe_symbol": None, "pe_token": None,
            "spot_token": _resolve_spot_token_from_cache(cached_nse_instruments),
        }

        order: List[int] = [target]
        for i in range(1, int(strike_range) + 1):
            order.extend([target + i * 50, target - i * 50])

        for side in ("CE", "PE"):
            found = False
            for strike in order:
                for inst in cands:
                    if inst.get("instrument_type") == side and int(float(inst.get("strike", 0))) == int(strike):
                        res[f"{side.lower()}_symbol"] = inst.get("tradingsymbol")
                        res[f"{side.lower()}_token"] = inst.get("instrument_token")
                        res["actual_strikes"][side.lower()] = int(strike)
                        found = True
                        break
                if found:
                    break

        if not (res["ce_token"] or res["pe_token"]):
            return None
        return res
    except Exception as e:
        logger.error("get_instrument_tokens: %s", e, exc_info=True)
        return None


def is_trading_hours(start_hhmm: str = "09:15", end_hhmm: str = "15:30", tz_name: Optional[str] = None) -> bool:
    try:
        tzid = tz_name or os.getenv("TZ") or "Asia/Kolkata"
        now = datetime.now(ZoneInfo(tzid)) if ZoneInfo else datetime.now()
        wd = now.weekday()
        start = datetime.strptime(start_hhmm, "%H:%M").time()
        end = datetime.strptime(end_hhmm, "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)
    except Exception as e:
        logger.error("is_trading_hours: %s", e)
        return True


def health_check(kite: Optional[KiteConnect]) -> Dict[str, Any]:
    status = {"overall_status": "OK", "message": "", "checks": {}}
    try:
        if not kite:
            status.update(overall_status="ERROR", message="No Kite instance")
            return status
        spot_sym = _get_spot_ltp_symbol()
        try:
            q = _rate_limited_api_call(kite.ltp, [spot_sym])
            ok = bool(q.get(spot_sym, {}).get("last_price"))
            status["checks"]["ltp"] = "OK" if ok else "FAIL"
            if not ok:
                status["overall_status"] = "ERROR"
        except Exception as e:
            status["checks"]["ltp"] = f"FAIL: {e}"; status["overall_status"] = "ERROR"
        try:
            nfo = _rate_limited_api_call(kite.instruments, "NFO")
            status["checks"]["instruments"] = "OK" if (isinstance(nfo, list) and nfo) else "FAIL"
            if status["checks"]["instruments"] == "FAIL":
                status["overall_status"] = "ERROR"
        except Exception as e:
            status["checks"]["instruments"] = f"FAIL: {e}"; status["overall_status"] = "ERROR"
        status["message"] = " | ".join(f"{k}:{v}" for k, v in status["checks"].items())
        return status
    except Exception as e:
        logger.error("health_check: %s", e, exc_info=True)
        return {"overall_status": "ERROR", "message": str(e), "checks": {}}
