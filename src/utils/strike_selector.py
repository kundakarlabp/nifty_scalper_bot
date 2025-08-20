# src/utils/strike_selector.py
"""
Utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

Public API consumed elsewhere:
- get_instrument_tokens(...)
- get_next_expiry_date(...)
- get_nearest_strikes(...)
- fetch_cached_instruments(...)
- is_market_open(...)
- health_check(...)

Design highlights:
- Uses cached instruments (pass them in) to avoid rate limits.
- Robust expiry normalization (date/datetime/string).
- Safe spot token resolution with fallback to known token (256265).
- Optional Greeks-based selection via src.utils.greeks (if present).
"""

from __future__ import annotations

import os
import time
import logging
import threading
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple

import pytz
import exchange_calendars as tcals
from kiteconnect import KiteConnect

# Optional Greeks helpers
try:
    from src.utils.greeks import implied_vol_bisection, estimate_delta
except Exception:
    implied_vol_bisection = None
    estimate_delta = None

logger = logging.getLogger(__name__)

__all__ = [
    "get_instrument_tokens",
    "get_next_expiry_date",
    "get_nearest_strikes",
    "fetch_cached_instruments",
    "is_market_open",
    "health_check",
]

# ----------------- rate limiting -----------------
_last_api_call: Dict[str, float] = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # seconds between calls per endpoint


def _rate_limited_api_call(func, *args, **kwargs):
    """Rate-limited API call wrapper with single retry on rate-limit."""
    with _api_call_lock:
        call_key = getattr(func, "__name__", "api_call")
        now = time.time()
        last = _last_api_call.get(call_key, 0.0)
        if now - last < _MIN_API_INTERVAL:
            time.sleep(_MIN_API_INTERVAL - (now - last))
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


# ----------------- small helpers -----------------
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
            logger.info("🎯 Strike range: %d - %d (%d strikes)", min(strikes), max(strikes), len(strikes))
        return strikes
    except Exception as e:
        logger.error("[get_nearest_strikes] %s", e, exc_info=True)
        return []


def _calculate_next_thursday(target_date: Optional[date] = None) -> str:
    """Pure calendar fallback for 'next Thursday' in YYYY-MM-DD (always future)."""
    d = target_date or date.today()
    days_ahead = (3 - d.weekday()) % 7  # Thu=3
    if days_ahead == 0:
        days_ahead = 7
    nxt = d + timedelta(days=days_ahead)
    return nxt.isoformat()


def _exp_to_date(exp: Any) -> Optional[date]:
    """Normalize an instrument 'expiry' field to date."""
    try:
        if isinstance(exp, datetime):
            return exp.date()
        if isinstance(exp, date):
            return exp
        # try ISO string "YYYY-MM-DD"
        y, m, d = map(int, str(exp)[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


# ----------------- cached instruments -----------------
def fetch_cached_instruments(kite: KiteConnect) -> Dict[str, List[Dict[str, Any]]]:
    """
    One-shot fetch for 'NFO' and 'NSE' instruments with rate limiting.
    Use at app start and refresh occasionally to avoid repeated API calls.
    """
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


# ----------------- expiry resolution -----------------
def get_next_expiry_date(
    kite_instance: KiteConnect,
    cached_nfo_instruments: Optional[List[Dict]] = None
) -> Optional[str]:
    """
    Resolve nearest upcoming expiry for NIFTY options based on cached NFO instruments.
    Returns ISO 'YYYY-MM-DD'. Fallback: calendar 'next Thursday' if instruments unavailable.
    """
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return _calculate_next_thursday()

    try:
        base_name_for_search = "NIFTY"  # Zerodha instruments 'name' for NIFTY options

        if cached_nfo_instruments is None:
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
        today = date.today()
        for inst in index_instruments:
            exp_d = _exp_to_date(inst.get("expiry"))
            if exp_d and exp_d >= today:
                candidates.add(exp_d)

        if candidates:
            return min(candidates).isoformat()

        return _calculate_next_thursday()
    except Exception as e:
        logger.warning("[get_next_expiry_date] Error, using fallback: %s", e, exc_info=True)
        return _calculate_next_thursday()


# ----------------- spot token resolution -----------------
def _resolve_spot_token_from_cache(cached_nse_instruments: List[Dict]) -> Optional[int]:
    """
    Try to find the NSE instrument token for NIFTY 50 index from cached instruments.
    Fallback to a known NIFTY 50 token if not found.
    """
    try:
        for inst in cached_nse_instruments or []:
            tsym = (inst.get("tradingsymbol") or "").strip().upper()
            inst_type = str(inst.get("instrument_type") or "").upper()
            # Zerodha uses instrument_type == "INDICES" for indices
            if tsym in ("NIFTY 50", "NIFTY") and inst_type == "INDICES":
                tok = inst.get("instrument_token")
                if tok:
                    return int(tok)
        return 256265  # hard fallback
    except Exception as e:
        logger.debug("[_resolve_spot_token_from_cache] %s", e)
        return 256265


def _find_instrument(
    nfo_list: List[Dict[str, Any]],
    *,
    strike: int,
    opt_type: str,
    expiry: date
) -> Optional[Dict[str, Any]]:
    """Find a single NIFTY option instrument row for given strike/type/expiry."""
    try:
        for inst in nfo_list or []:
            if inst.get("name") != "NIFTY":
                continue
            if str(inst.get("instrument_type")).upper() != opt_type:
                continue
            try:
                if int(float(inst.get("strike", 0))) != int(strike):
                    continue
            except Exception:
                continue
            exp_d = _exp_to_date(inst.get("expiry"))
            if not exp_d or exp_d != expiry:
                continue
            return inst
        return None
    except Exception:
        return None


# ----------------- Greeks-driven selection (optional) -----------------
def _greeks_pick_strikes(
    *,
    kite_instance: KiteConnect,
    spot_price: float,
    atm_strike: int,
    expiry_dt: date,
    nfo_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Greeks-driven CE/PE selection around ATM:
    - Target deltas (±0.35 by default) with tolerance
    - Require minimum OI (if enabled)
    - IV estimated from LTP via inverse BS if available; fallback 20%
    """
    if implied_vol_bisection is None or estimate_delta is None:
        logger.debug("[Greeks] utils.greeks not available; skipping Greeks mode.")
        return None

    # config knobs (ENV)
    rf = float(os.environ.get("RISK_FREE_RATE", "0.06"))
    tgt_call = float(os.environ.get("TARGET_DELTA_CALL", "0.35"))
    tgt_put = float(os.environ.get("TARGET_DELTA_PUT", "-0.35"))
    tol = float(os.environ.get("DELTA_TOL", "0.05"))
    min_oi = int(os.environ.get("MIN_OI", "50000"))
    require_oi = str(os.environ.get("REQUIRE_OI", "true")).lower() in ("1", "true", "yes", "y", "on")
    iv_mode = (os.environ.get("IV_SOURCE", "LTP_IMPLIED") or "LTP_IMPLIED").upper()

    # strike window around ATM
    step = 50
    window = 6
    strikes = [atm_strike + i * step for i in range(-window, window + 1)]

    # days to expiry
    now = date.today()
    dte_days = max(0.5, (expiry_dt - now).days or 0.5)

    def _pick(opt_type: str) -> Optional[Dict[str, Any]]:
        best = None
        best_err = 1e9
        for k in strikes:
            row = _find_instrument(nfo_list, strike=k, opt_type=opt_type, expiry=expiry_dt)
            if not row:
                continue

            oi = int(row.get("oi") or 0)
            if require_oi and oi < min_oi:
                continue

            ts = row.get("tradingsymbol")
            exch = row.get("exchange") or "NFO"
            try:
                ltp = _rate_limited_api_call(kite_instance.ltp, [f"{exch}:{ts}"])[f"{exch}:{ts}"]["last_price"]
            except Exception:
                continue

            # estimate IV from LTP
            if iv_mode == "LTP_IMPLIED":
                try:
                    iv = implied_vol_bisection(
                        target_price=float(ltp),
                        spot=float(spot_price),
                        strike=float(k),
                        days_to_expiry=float(dte_days),
                        opt_type=("CE" if opt_type == "CE" else "PE"),
                        r=rf,
                        q=0.0,
                    )
                except Exception:
                    iv = 0.20
            else:
                iv = 0.20

            dlt = estimate_delta(
                spot=float(spot_price),
                strike=float(k),
                days_to_expiry=float(dte_days),
                iv=float(iv),
                opt_type=("CE" if opt_type == "CE" else "PE"),
            )

            target = tgt_call if opt_type == "CE" else tgt_put
            err = abs(dlt - target)
            if err <= tol and err < best_err:
                best_err = err
                best = {
                    "tradingsymbol": ts,
                    "token": int(row.get("instrument_token")),
                    "strike": int(k),
                    "delta": float(dlt),
                    "iv": float(iv),
                    "ltp": float(ltp),
                }
        return best

    ce_best = _pick("CE")
    pe_best = _pick("PE")
    if ce_best and pe_best:
        logger.info("[Greeks] Selected CE %s(Δ=%.2f), PE %s(Δ=%.2f)",
                    ce_best["tradingsymbol"], ce_best["delta"],
                    pe_best["tradingsymbol"], pe_best["delta"])
        return {"ce": ce_best, "pe": pe_best}

    logger.info("[Greeks] No strike within delta tolerance → fallback to ATM selector")
    return None


# ----------------- main API -----------------
def get_instrument_tokens(
    kite_instance: KiteConnect,
    spot_symbol: str,
    cached_nfo_instruments: List[Dict],
    cached_nse_instruments: List[Dict],
    offset: int = 0,
    strike_range: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Get CE/PE instrument tokens for NIFTY weeklys.
    Returns dict with keys:
      - ce_symbol, ce_token, pe_symbol, pe_token
      - spot_price, spot_token, atm_strike, target_strike, actual_strikes{ce,pe}
      - expiry (ISO), offset, ce_delta/pe_delta (when Greeks enabled)
    """
    if not kite_instance:
        logger.error("[get_instrument_tokens] KiteConnect instance is required.")
        return None
    if not cached_nfo_instruments:
        logger.error("[get_instrument_tokens] Cached NFO instruments are required.")
        return None    # do not spam APIs; runner should refresh cache
    if not cached_nse_instruments:
        logger.error("[get_instrument_tokens] Cached NSE instruments are required.")
        return None

    try:
        base_name = "NIFTY"  # Zerodha 'name' for NIFTY options

        # 1) Spot LTP
        try:
            spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol]) or {}
            spot_price = float(spot_data.get(spot_symbol, {}).get("last_price") or 0.0)
            if not spot_price:
                logger.error("[get_instrument_tokens] Could not fetch spot price for %s.", spot_symbol)
                return None
        except Exception as e:
            logger.error("[get_instrument_tokens] LTP error: %s", e)
            return None

        # 2) ATM & target (legacy/offset path)
        atm = get_atm_strike_price(spot_price)
        target = atm + (int(offset) * 50)

        # 3) Expiry (ISO) and as date
        expiry_iso = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry_iso:
            logger.error("[get_instrument_tokens] Could not resolve expiry.")
            return None
        y, m, d = map(int, expiry_iso.split("-"))
        expiry_dt = date(y, m, d)

        logger.info("[get_instrument_tokens] Spot:%.2f ATM:%d Target:%d Expiry:%s Offset:%d",
                    spot_price, atm, target, expiry_iso, int(offset))

        # 4) Filter relevant NFO instruments: name=NIFTY, expiry matches
        candidates = [
            i for i in (cached_nfo_instruments or [])
            if i.get("name") == base_name and _exp_to_date(i.get("expiry")) == expiry_dt
        ]
        if not candidates:
            logger.error("[get_instrument_tokens] No instruments for %s @ %s", base_name, expiry_iso)
            return None

        results: Dict[str, Any] = {
            "spot_price": spot_price,
            "atm_strike": atm,
            "target_strike": target,
            "offset": int(offset),
            "actual_strikes": {},
            "expiry": expiry_iso,
            "ce_symbol": None,
            "ce_token": None,
            "pe_symbol": None,
            "pe_token": None,
            "spot_token": _resolve_spot_token_from_cache(cached_nse_instruments),
        }

        # 5) Optional Greeks selection
        use_greeks = str(os.environ.get("USE_GREEKS_STRIKE_RANKING", "false")).lower() in ("1", "true", "yes", "y", "on")
        if use_greeks:
            picked = _greeks_pick_strikes(
                kite_instance=kite_instance,
                spot_price=spot_price,
                atm_strike=atm,
                expiry_dt=expiry_dt,
                nfo_list=candidates,
            )
            if picked:
                ce_best, pe_best = picked["ce"], picked["pe"]
                results.update({
                    "ce_symbol": ce_best["tradingsymbol"],
                    "ce_token": ce_best["token"],
                    "pe_symbol": pe_best["tradingsymbol"],
                    "pe_token": pe_best["token"],
                    "actual_strikes": {"ce": ce_best["strike"], "pe": pe_best["strike"]},
                    "ce_delta": round(ce_best["delta"], 3),
                    "pe_delta": round(pe_best["delta"], 3),
                })
                return results

        # 6) Legacy ATM/offset search (fallback or when Greeks disabled)
        search_order: List[int] = [target]
        for i in range(1, int(strike_range) + 1):
            search_order.extend([target + i * 50, target - i * 50])

        for side in ("CE", "PE"):
            found = False
            for strike in search_order:
                inst = _find_instrument(candidates, strike=int(strike), opt_type=side, expiry=expiry_dt)
                if not inst:
                    continue
                results[f"{side.lower()}_symbol"] = inst.get("tradingsymbol")
                results[f"{side.lower()}_token"] = inst.get("instrument_token")
                results["actual_strikes"][side.lower()] = int(strike)
                logger.info("[get_instrument_tokens] Found %s: %s (%s)",
                            side, inst.get("tradingsymbol"), inst.get("instrument_token"))
                found = True
                break
            if not found:
                logger.warning("[get_instrument_tokens] ❌ No %s within ±%d*50 points", side, int(strike_range))

        if not any([results["ce_token"], results["pe_token"]]):
            logger.error("[get_instrument_tokens] ❌ No options found in range")
            return None

        for side in ("ce", "pe"):
            a = results["actual_strikes"].get(side)
            if a and a != target:
                logger.info("[get_instrument_tokens] %s strike adjusted: %d → %d", side.upper(), target, a)

        return results
    except Exception as e:
        logger.error("[get_instrument_tokens] Unexpected error: %s", e, exc_info=True)
        return None


# ----------------- market hours -----------------
def is_market_open(
    time_filter_start: str = "09:15",
    time_filter_end: str = "15:30",
    exchange: str = "XNSE",  # NSE calendar
) -> bool:
    """
    Checks if the specified exchange is open, considering market holidays
    and standard trading hours PLUS the provided tighter time filter (IST).
    """
    try:
        cal = tcals.get_calendar(exchange)
        now_utc = datetime.now(pytz.utc)

        # Is today a trading session?
        if not cal.is_session(now_utc.date()):
            return False

        # Exchange session bounds (tz-aware, UTC)
        session_open, session_close = cal.session_open_close(now_utc.date())

        if not (session_open <= now_utc <= session_close):
            return False

        # User-defined tighter window in IST
        tz_ist = pytz.timezone("Asia/Kolkata")
        now_ist = now_utc.astimezone(tz_ist)

        s_h, s_m = map(int, time_filter_start.split(":"))
        e_h, e_m = map(int, time_filter_end.split(":"))

        start_time = now_ist.replace(hour=s_h, minute=s_m, second=0, microsecond=0).time()
        end_time = now_ist.replace(hour=e_h, minute=e_m, second=0, microsecond=0).time()

        return start_time <= now_ist.time() <= end_time
    except Exception as e:
        logger.error("[is_market_open] Error checking market hours: %s", e, exc_info=True)
        # Fail-open to avoid unexpected blocking if calendars break
        return True


# ----------------- diagnostics -----------------
def health_check(kite: Optional[KiteConnect], spot_symbol: str) -> Dict[str, Any]:
    """
    Lightweight readiness probe.
    Checks LTP reachability and validates instruments cache fetch.
    """
    status = {"overall_status": "OK", "message": "", "checks": {}}
    try:
        if not kite:
            status.update(overall_status="ERROR", message="No Kite instance")
            return status

        # LTP check
        try:
            ltp = _rate_limited_api_call(kite.ltp, [spot_symbol]) or {}
            ok = bool(ltp.get(spot_symbol, {}).get("last_price"))
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
