# src/utils/strike_selector.py
"""
Utility functions for selecting strike prices and fetching instrument tokens
for Nifty 50 options trading.

- Robust symbol resolution + conservative fallbacks
- Uses cached instruments (passed in) with rate-limited Kite API wrappers
- Optional Greeks-driven selection with OI/IV/premium filters and looser fallback
- ATM/offset legacy fallback path for reliability
- Lightweight health_check() for readiness probes
"""

from __future__ import annotations

import os
import logging
import threading
import time
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any

from kiteconnect import KiteConnect

# Greeks helpers (optional)
try:
    from src.utils.greeks import implied_vol_bisection, estimate_delta
except Exception:  # keep import-safe if module not present
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

# ---------------- Rate limit wrapper ----------------
_last_api_call: Dict[str, float] = {}
_api_call_lock = threading.RLock()
_MIN_API_INTERVAL = 0.5  # per-endpoint cadence


def _rate_limited_api_call(func, *args, **kwargs):
    with _api_call_lock:
        key = getattr(func, "__name__", "api_call")
        now = time.time()
        if key in _last_api_call:
            elapsed = now - _last_api_call[key]
            if elapsed < _MIN_API_INTERVAL:
                time.sleep(_MIN_API_INTERVAL - elapsed)
        try:
            out = func(*args, **kwargs)
            _last_api_call[key] = time.time()
            return out
        except Exception as e:
            msg = str(e).lower()
            if "too many" in msg or "rate" in msg:
                logger.warning("Rate limit on %s; retrying in 2s…", key)
                time.sleep(2)
                out = func(*args, **kwargs)
                _last_api_call[key] = time.time()
                return out
            raise


# ---------------- Basics ----------------
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
        logger.error("[_format_expiry_for_symbol_primary] %s", e)
        return ""


def format_option_symbol(base_symbol: str, expiry: str, strike: int, option_type: str) -> str:
    try:
        exp = _format_expiry_for_symbol_primary(expiry)
        return f"{base_symbol}{exp}{int(strike)}{option_type}" if exp else ""
    except Exception as e:
        logger.error("[format_option_symbol] %s", e)
        return ""


def get_atm_strike_price(spot_price: float) -> int:
    try:
        return int(round(float(spot_price) / 50.0) * 50)
    except Exception as e:
        logger.error("[get_atm_strike_price] %s", e)
        return 24500


def get_nearest_strikes(spot_price: float, strike_count: int = 5) -> List[int]:
    try:
        atm = get_atm_strike_price(spot_price)
        half = max(1, strike_count // 2)
        return sorted(set(atm + i * 50 for i in range(-half, half + 1)))
    except Exception as e:
        logger.error("[get_nearest_strikes] %s", e, exc_info=True)
        return []


def _calculate_next_thursday(target_date: Optional[date] = None) -> str:
    d = target_date or date.today()
    days_ahead = (3 - d.weekday()) % 7  # Thu=3
    if days_ahead == 0:
        days_ahead = 7
    return (d + timedelta(days=days_ahead)).isoformat()


# ---------------- Cached instruments helpers ----------------
def fetch_cached_instruments(kite: KiteConnect) -> Dict[str, List[Dict[str, Any]]]:
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


# ---------------- Core selection helpers ----------------
def get_next_expiry_date(
    kite_instance: KiteConnect,
    cached_nfo_instruments: Optional[List[Dict]] = None,
) -> Optional[str]:
    if not kite_instance:
        logger.error("[get_next_expiry_date] KiteConnect instance is required.")
        return _calculate_next_thursday()

    try:
        if cached_nfo_instruments is None:
            try:
                cached_nfo_instruments = _rate_limited_api_call(kite_instance.instruments, "NFO")
            except Exception as e:
                logger.warning("[get_next_expiry_date] instruments fetch failed; fallback: %s", e)
                return _calculate_next_thursday()

        index_rows = [i for i in (cached_nfo_instruments or []) if i.get("name") == "NIFTY"]
        if not index_rows:
            logger.warning("[get_next_expiry_date] No NFO 'NIFTY' instruments; fallback to calendar.")
            return _calculate_next_thursday()

        cands: set[date] = set()
        for inst in index_rows:
            exp = inst.get("expiry")
            if not exp:
                continue
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
        logger.warning("[get_next_expiry_date] Error; using fallback: %s", e, exc_info=True)
        return _calculate_next_thursday()


def _resolve_spot_token_from_cache(cached_nse_instruments: List[Dict]) -> Optional[int]:
    try:
        from src.config import Config
        for inst in cached_nse_instruments or []:
            tsym = (inst.get("tradingsymbol") or "").strip().upper()
            seg = (inst.get("segment") or "").upper()
            if tsym == "NIFTY 50" and "INDICE" in seg:
                tok = inst.get("instrument_token")
                if tok:
                    return int(tok)
        return int(getattr(Config, "INSTRUMENT_TOKEN", 256265))
    except Exception:
        return None


def _exp_to_date(x: Any) -> Optional[date]:
    try:
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        y, m, d = map(int, str(x)[:10].split("-"))
        return date(y, m, d)
    except Exception:
        return None


def _find_instrument(nfo_list: List[Dict[str, Any]], *, strike: int, opt_type: str, expiry: date) -> Optional[Dict]:
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
        if _exp_to_date(inst.get("expiry")) != expiry:
            continue
        return inst
    return None


# ---------------- Greeks-driven picker ----------------
_last_no_pick_log_ts = 0.0  # throttle "no strike within tol" logs


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _greeks_pick_strikes(
    *,
    kite_instance: KiteConnect,
    spot_price: float,
    atm_strike: int,
    expiry_dt: date,
    nfo_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick CE/PE near target deltas with optional looser fallback and hygiene filters."""
    global _last_no_pick_log_ts

    if implied_vol_bisection is None or estimate_delta is None:
        logger.debug("[Greeks] greeks utils unavailable; skipping Greeks mode.")
        return None

    # Config knobs
    rf = float(os.environ.get("RISK_FREE_RATE", "0.06"))
    tgt_call = float(os.environ.get("TARGET_DELTA_CALL", "0.35"))
    tgt_put = float(os.environ.get("TARGET_DELTA_PUT", "-0.35"))
    tol = float(os.environ.get("DELTA_TOL", "0.05"))
    min_oi = int(os.environ.get("MIN_OI", "50000"))
    require_oi = _env_bool("REQUIRE_OI", True)
    iv_mode = (os.environ.get("IV_SOURCE", "LTP_IMPLIED") or "LTP_IMPLIED").upper()
    iv_floor = float(os.environ.get("IV_FLOOR", "0.0"))
    iv_ceil = float(os.environ.get("IV_CEIL", "10.0"))
    min_prem = float(os.environ.get("MIN_PREMIUM", "0.0"))
    loose_fallback = _env_bool("GREEKS_LOOSE_FALLBACK", True)
    log_throttle = int(os.environ.get("GREEKS_LOG_THROTTLE_SEC", "60"))

    # Strike search window
    step = 50
    win = max(2, int(os.environ.get("MAX_GREEKS_STRIKE_WINDOW", "6")))
    strikes = [atm_strike + i * step for i in range(-win, win + 1)]

    # DTE in days
    dte_days = max(0.5, (expiry_dt - datetime.now().date()).days or 0.5)

    # Build symbol list once → bulk LTP to minimize rate calls
    sym_rows = []
    for k in strikes:
        for side in ("CE", "PE"):
            row = _find_instrument(nfo_list, strike=k, opt_type=side, expiry=expiry_dt)
            if row:
                ts = row.get("tradingsymbol")
                exch = row.get("exchange") or "NFO"
                sym_rows.append((f"{exch}:{ts}", row, k, side))
    if not sym_rows:
        return None

    try:
        ltps = _rate_limited_api_call(kite_instance.ltp, [s for s, *_ in sym_rows]) or {}
    except Exception as e:
        logger.debug("[Greeks] bulk LTP failed: %s", e)
        ltps = {}

    def _eval(side: str):
        target = tgt_call if side == "CE" else tgt_put
        best_strict = None
        best_strict_err = 1e9
        best_loose = None
        best_loose_err = 1e9

        for s, row, strike, s_side in sym_rows:
            if s_side != side:
                continue

            oi = int(row.get("oi") or 0)
            if require_oi and oi < min_oi:
                continue

            ltp = float((ltps.get(s) or {}).get("last_price") or 0.0)
            if ltp <= 0.0 or ltp < min_prem:
                continue

            # IV estimate
            if iv_mode == "LTP_IMPLIED":
                try:
                    iv = float(
                        implied_vol_bisection(
                            target_price=ltp,
                            spot=float(spot_price),
                            strike=float(strike),
                            days_to_expiry=float(dte_days),
                            opt_type=side,
                            r=rf,
                            q=0.0,
                        )
                    )
                except Exception:
                    iv = 0.20
            else:
                iv = 0.20

            iv = max(iv_floor, min(iv, iv_ceil))

            try:
                delta = float(
                    estimate_delta(
                        spot=float(spot_price),
                        strike=float(strike),
                        days_to_expiry=float(dte_days),
                        iv=float(iv),
                        opt_type=side,
                    )
                )
            except Exception:
                continue

            err = abs(delta - target)
            cand = {
                "tradingsymbol": row.get("tradingsymbol"),
                "token": int(row.get("instrument_token")),
                "strike": int(strike),
                "delta": float(delta),
                "iv": float(iv),
                "ltp": float(ltp),
                "oi": oi,
            }

            if err <= tol and err < best_strict_err:
                best_strict = cand
                best_strict_err = err
            elif loose_fallback and err <= (2 * tol) and err < best_loose_err:
                best_loose = cand
                best_loose_err = err

        return best_strict or best_loose

    ce = _eval("CE")
    pe = _eval("PE")

    if ce and pe:
        logger.info(
            "[Greeks] Selected CE %s(Δ=%.2f, iv=%.2f, ltp=%.2f) | PE %s(Δ=%.2f, iv=%.2f, ltp=%.2f)",
            ce["tradingsymbol"], ce["delta"], ce["iv"], ce["ltp"],
            pe["tradingsymbol"], pe["delta"], pe["iv"], pe["ltp"],
        )
        return {"ce": ce, "pe": pe}

    # Throttled info log to avoid spam
    now = time.time()
    if now - _last_no_pick_log_ts > log_throttle:
        _last_no_pick_log_ts = now
        logger.info("[Greeks] No strike within tolerance — using ATM/offset fallback")
    return None


# ---------------- Public: instrument resolution ----------------
def get_instrument_tokens(
    symbol: str,
    kite_instance: KiteConnect,
    cached_nfo_instruments: List[Dict],
    cached_nse_instruments: List[Dict],
    offset: int = 0,
    strike_range: int = 3,
) -> Optional[Dict[str, Any]]:
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
        # Spot
        spot_sym = _get_spot_ltp_symbol()
        spot_data = _rate_limited_api_call(kite_instance.ltp, [spot_sym]) or {}
        spot_price = float((spot_data.get(spot_sym) or {}).get("last_price") or 0.0)
        if spot_price <= 0:
            logger.error("[get_instrument_tokens] Could not fetch spot price.")
            return None

        # ATM / target
        atm = get_atm_strike_price(spot_price)
        target = atm + (int(offset) * 50)

        # Expiry
        expiry_iso = get_next_expiry_date(kite_instance, cached_nfo_instruments)
        if not expiry_iso:
            logger.error("[get_instrument_tokens] Could not resolve expiry.")
            return None
        y, m, d = map(int, expiry_iso.split("-"))
        expiry_dt = date(y, m, d)

        logger.info(
            "[get_instrument_tokens] Spot:%.2f ATM:%d Target:%d Expiry:%s Offset:%d",
            spot_price, atm, target, expiry_iso, int(offset)
        )

        # Filter instruments for that expiry
        def _exp_str(x) -> str:
            return x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)

        candidates = [i for i in cached_nfo_instruments if i.get("name") == "NIFTY" and _exp_str(i.get("expiry")) == expiry_iso]
        if not candidates:
            logger.error("[get_instrument_tokens] No NIFTY instruments @ %s", expiry_iso)
            return None

        # Base result
        res: Dict[str, Any] = {
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
            "ce_delta": None,
            "pe_delta": None,
        }

        # Greeks mode (opt-in)
        use_greeks = _env_bool("USE_GREEKS_STRIKE_RANKING", False)
        if use_greeks:
            picked = _greeks_pick_strikes(
                kite_instance=kite_instance,
                spot_price=spot_price,
                atm_strike=atm,
                expiry_dt=expiry_dt,
                nfo_list=candidates,
            )
            if picked:
                ce, pe = picked["ce"], picked["pe"]
                res.update(
                    ce_symbol=ce["tradingsymbol"],
                    ce_token=ce["token"],
                    pe_symbol=pe["tradingsymbol"],
                    pe_token=pe["token"],
                    actual_strikes={"ce": ce["strike"], "pe": pe["strike"]},
                    ce_delta=round(ce["delta"], 3),
                    pe_delta=round(pe["delta"], 3),
                )
                return res

        # Legacy ATM/offset fallback search (or Greeks disabled)
        search_order: List[int] = [target]
        for i in range(1, int(strike_range) + 1):
            search_order.extend([target + i * 50, target - i * 50])

        for side in ("CE", "PE"):
            found = False
            for k in search_order:
                for inst in candidates:
                    if inst.get("instrument_type") != side:
                        continue
                    try:
                        if int(float(inst.get("strike", 0))) != int(k):
                            continue
                    except Exception:
                        continue

                    res[f"{side.lower()}_symbol"] = inst.get("tradingsymbol")
                    res[f"{side.lower()}_token"] = inst.get("instrument_token")
                    res["actual_strikes"][side.lower()] = int(k)
                    logger.info(
                        "[get_instrument_tokens] Found %s: %s (%s)",
                        side, inst.get("tradingsymbol"), inst.get("instrument_token")
                    )
                    found = True
                    break
                if found:
                    break
            if not found:
                logger.warning("[get_instrument_tokens] No %s within ±%d*50", side, int(strike_range))

        if not (res["ce_token"] or res["pe_token"]):
            logger.error("[get_instrument_tokens] No options found in range.")
            return None

        # Inform if adjusted from target
        for side in ("ce", "pe"):
            a = res["actual_strikes"].get(side)
            if a and a != target:
                logger.info("[get_instrument_tokens] %s strike adjusted: %d → %d", side.upper(), target, a)

        return res

    except Exception as e:
        logger.error("[get_instrument_tokens] Unexpected error: %s", e, exc_info=True)
        return None


# ---------------- Trading-hours + health ----------------
def is_trading_hours() -> bool:
    try:
        now = datetime.now()
        wd = now.weekday()
        start = datetime.strptime("09:15", "%H:%M").time()
        end = datetime.strptime("15:30", "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)
    except Exception:
        return True  # fail-open


def health_check(kite: Optional[KiteConnect]) -> Dict[str, Any]:
    status = {"overall_status": "OK", "message": "", "checks": {}}
    try:
        if not kite:
            status.update(overall_status="ERROR", message="No Kite instance")
            return status

        # LTP
        spot = _get_spot_ltp_symbol()
        try:
            l = _rate_limited_api_call(kite.ltp, [spot]) or {}
            ok = bool((l.get(spot) or {}).get("last_price"))
            status["checks"]["ltp"] = "OK" if ok else "FAIL"
            if not ok:
                status["overall_status"] = "ERROR"
        except Exception as e:
            status["checks"]["ltp"] = f"FAIL: {e}"
            status["overall_status"] = "ERROR"

        # Instruments
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