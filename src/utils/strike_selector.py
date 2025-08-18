# src/utils/strike_selector.py
"""
Utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

- Robust symbol resolution and conservative fallbacks
- Cached instruments usage (passed in), with rate-limited API wrappers
- Optional Greeks-driven selection with OI/IV filters (delta targeting)
- Spread-agnostic (selection only), used by the trader which applies its own guards
- Lightweight health_check() for readiness probes

Public functions used elsewhere:
- _get_spot_ltp_symbol()
- get_instrument_tokens(...)
- get_next_expiry_date(...)
- get_nearest_strikes(...)
- fetch_cached_instruments(...)
- is_trading_hours()
- health_check(...)
"""

from __future__ import annotations

import os
from kiteconnect import KiteConnect
from datetime import datetime, date, timedelta
import logging
import time
import threading
from typing import Optional, Dict, List, Any, Tuple

# Greeks helpers (new)
try:
    from src.utils.greeks import implied_vol_bisection, estimate_delta
except Exception:
    # keep import-safe in environments where file isn't present yet
    implied_vol_bisection = None
    estimate_delta = None

logger = logging.getLogger(__name__)

__all__ = [
    "_get_spot_ltp_symbol",
    "get_instrument_tokens",
    "get_next_expiry_date",
    "get_nearest_strikes",
    "fetch_cached_instruments",
    "is_trading_hours",
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
                logger.warning(f"Rate limit for {call_key}, retrying in 2s...")
                time.sleep(2)
                result = func(*args, **kwargs)
                _last_api_call[call_key] = time.time()
                return result
            raise


def _format_expiry_for_symbol_primary(expiry_str: str) -> str:
    """Format: YYMONDD (e.g., '2025-08-07' -> '25AUG07')."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except Exception as e:
        logger.error(f"[_format_expiry_for_symbol_primary] {e}")
        return ""


def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    """Primary tradingsymbol format used by Zerodha for NIFTY weeklys."""
    try:
        exp = _format_expiry_for_symbol_primary(expiry)
        return f"{base_symbol}{exp}{int(strike)}{option_type}" if exp else ""
    except Exception as e:
        logger.error(f"[format_option_symbol] {e}")
        return ""


def get_atm_strike_price(spot_price: float) -> int:
    """Nearest 50-step strike."""
    try:
        return int(round(float(spot_price) / 50.0) * 50)
    except Exception as e:
        logger.error(f"[get_atm_strike_price] {e}")
        return 24500


def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    """Centered range around ATM."""
    try:
        atm = get_atm_strike_price(spot_price)
        half = max(1, strike_count // 2)
        strikes = sorted(set(atm + i * 50 for i in range(-half, half + 1)))
        if strikes:
            logger.info(f"🎯 Strike range: {min(strikes)} - {max(strikes)} ({len(strikes)} strikes)")
        return strikes
    except Exception as e:
        logger.error(f"[get_nearest_strikes] {e}", exc_info=True)
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
def fetch_cached_instruments(kite: KiteConnect) -> Dict[str, List[Dict[str, Any]]]:
    """
    One-shot fetch for 'NFO' and 'NSE' instruments with rate limiting.
    Use this at app start and refresh occasionally to avoid repeated API calls.
    """
    try:
        nfo = _rate_limited_api_call(kite.instruments, "NFO")
    except Exception as e:
        logger.error(f"[fetch_cached_instruments] NFO fetch failed: {e}")
        nfo = []

    try:
        nse = _rate_limited_api_call(kite.instruments, "NSE")
    except Exception as e:
        logger.error(f"[fetch_cached_instruments] NSE fetch failed: {e}")
        nse = []

    return {"NFO": nfo or [], "NSE": nse or []}


# ---------------- Core selection functions ----------------
def get_next_expiry_date(
    kite_instance: KiteConnect,
    cached_nfo_instruments: Optional[List[Dict]] = None
) -> Optional[str]:
    """
    Resolve the nearest upcoming expiry for NIFTY options based on cached NFO instruments.
    Returns ISO 'YYYY-MM-DD'. Fallback: calendar 'next Thursday' if instruments unavailable.

    Fix: normalize instrument 'expiry' which may be datetime.date or datetime.datetime,
    and DO NOT call .date() on a date.
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
                logger.warning(f"[get_next_expiry_date] instruments fetch failed, fallback: {e}")
                return _calculate_next_thursday()

        index_instruments = [i for i in (cached_nfo_instruments or []) if i.get("name") == base_name_for_search]
        if not index_instruments:
            logger.warning(f"[get_next_expiry_date] No NFO instruments for '{base_name_for_search}', fallback.")
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

        # If everything is in the past, fallback to next Thursday (safer than returning stale)
        return _calculate_next_thursday()

    except Exception as e:
        logger.warning(f"[get_next_expiry_date] Error, using fallback: {e}", exc_info=True)
        return _calculate_next_thursday()


def _resolve_spot_token_from_cache(
    cached_nse_instruments: List[Dict],
) -> Optional[int]:
    """
    Try to find the NSE instrument token for NIFTY 50 index from cached instruments.
    Fallback to a hardcoded default if not found.
    """
    try:
        for inst in cached_nse_instruments or []:
            tsym = (inst.get("tradingsymbol") or "").strip().upper()
            seg = (inst.get("segment") or "").upper()
            if tsym == "NIFTY 50" and "INDICE" in seg:
                tok = inst.get("instrument_token")
                if tok:
                    return int(tok)
        # Fallback to a known NIFTY 50 token if not found in cache
        return 256265
    except Exception as e:
        logger.debug(f"[resolve_spot_token_from_cache] {e}")
        return None


def _exp_to_date(exp: Any) -> Optional[date]:
    """Normalize an instrument 'expiry' field to date."""
    try:
        if isinstance(exp, datetime):
            return exp.date()
        if isinstance(exp, date):
            return exp
        y, m, d = map(int, str(exp)[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


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
            if inst.get("instrument_type") != opt_type:
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

    # config knobs
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
    now = datetime.now().date()
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
        logger.info(
            "[Greeks] Selected CE %s(Δ=%.2f), PE %s(Δ=%.2f)",
            ce_best["tradingsymbol"], ce_best["delta"],
            pe_best["tradingsymbol"], pe_best["delta"],
        )
        return {
            "ce": ce_best,
            "pe": pe_best,
        }

    logger.info("[Greeks] No strike within delta tolerance → fallback to ATM selector")
    return None


import pytz
import exchange_calendars as tcals


def get_instrument_tokens(
    kite_instance: KiteConnect,
    spot_symbol: str,
    cached_nfo_instruments: List[Dict],
    cached_nse_instruments: List[Dict],
    offset: int = 0,
    strike_range: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Get CE/PE instrument tokens for a given NIFTY symbol with offset.
    Requires cached instruments to avoid API rate limits.
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
        base_name = "NIFTY"  # Zerodha 'name' for NIFTY options

        # 1) Spot LTP
        try:
            spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_symbol])
            spot_price = float(spot_data.get(spot_symbol, {}).get("last_price") or 0.0)
            if not spot_price:
                logger.error("[get_instrument_tokens] Could not fetch spot price.")
                return None
        except Exception as e:
            logger.error(f"[get_instrument_tokens] LTP error: {e}")
            return None

        # 2) ATM & target (for legacy/offset path)
        atm = get_atm_strike_price(spot_price)
        target = atm + (int(offset) * 50)

        # 3) Expiry (ISO) and as date
        expiry_iso = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry_iso:
            logger.error("[get_instrument_tokens] Could not resolve expiry.")
            return None
        y, m, d = map(int, expiry_iso.split("-"))
        expiry_dt = date(y, m, d)

        logger.info(
            f"[get_instrument_tokens] Spot:{spot_price:.2f} ATM:{atm} Target:{target} "
            f"Expiry:{expiry_iso} Offset:{offset}"
        )

        def _exp_str(x) -> str:
            if hasattr(x, "strftime"):
                return x.strftime("%Y-%m-%d")
            return str(x)

        # 4) Filter relevant instruments: name=NIFTY, expiry matches
        candidates = [
            i
            for i in cached_nfo_instruments
            if i.get("name") == base_name and _exp_str(i.get("expiry")) == expiry_iso
        ]
        if not candidates:
            logger.error(f"[get_instrument_tokens] No instruments for {base_name} @ {expiry_iso}")
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

        # 5) Greeks-driven selection (opt-in)
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
                return results  # success; done

        # 6) Legacy ATM/offset search (fallback or when Greeks disabled)
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
                            f"[get_instrument_tokens] Found {side}: "
                            f"{inst.get('tradingsymbol')} ({inst.get('instrument_token')})"
                        )
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.warning(f"[get_instrument_tokens] ❌ No {side} within ±{strike_range}*50 points")

        ok = any([results["ce_token"], results["pe_token"]])
        if not ok:
            logger.error("[get_instrument_tokens] ❌ No options found in range")
            return None

        for side in ("ce", "pe"):
            a = results["actual_strikes"].get(side)
            if a and a != target:
                logger.info(f"[get_instrument_tokens] {side.upper()} strike adjusted: {target} → {a}")

        return results

    except Exception as e:
        logger.error(f"[get_instrument_tokens] Unexpected error: {e}", exc_info=True)
        return None


def is_market_open(
    time_filter_start: str = "09:15",
    time_filter_end: str = "15:30",
    exchange: str = "XNSE"
) -> bool:
    """
    Checks if the specified exchange is open, considering market holidays
    and standard trading hours.
    """
    try:
        cal = tcals.get_calendar(exchange)
        now_utc = datetime.now(pytz.utc)

        # Check if today is a trading day
        if not cal.is_session(now_utc.date()):
            return False

        # Check if current time is within trading hours
        market_open, market_close = cal.open_and_close_for_session(now_utc.date())

        # Check against the broader exchange hours
        if not (market_open <= now_utc <= market_close):
            return False

        # Check against the user-defined tighter time filter
        tz_ist = pytz.timezone("Asia/Kolkata")
        now_ist = now_utc.astimezone(tz_ist)

        start_h, start_m = map(int, time_filter_start.split(":"))
        end_h, end_m = map(int, time_filter_end.split(":"))

        start_time = now_ist.replace(hour=start_h, minute=start_m, second=0, microsecond=0).time()
        end_time = now_ist.replace(hour=end_h, minute=end_m, second=0, microsecond=0).time()

        return start_time <= now_ist.time() <= end_time

    except Exception as e:
        logger.error(f"[is_market_open] Error checking market hours: {e}", exc_info=True)
        return True  # Fail-open to avoid unexpected blocking


# ---------------- Diagnostics ----------------
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
            ltp = _rate_limited_api_call(kite.ltp, [spot_symbol])
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
        logger.error(f"[health_check] {e}", exc_info=True)
        return {"overall_status": "ERROR", "message": str(e), "checks": {}}