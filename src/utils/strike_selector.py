from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = Any  # fallback for type checking

# ------------ lightweight in-memory cache with TTL ------------

_INSTR_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_TTL = 300.0  # 5 minutes


def _now() -> float:
    return time.time()


def _get_instruments(kite: KiteConnect, exchange: str = "NFO") -> List[Dict[str, Any]]:
    key = f"instruments::{exchange}"
    ts_list = _INSTR_CACHE.get(key)
    if ts_list and (_now() - ts_list[0]) < _CACHE_TTL:
        return ts_list[1]
    # rate-limited fetch
    instruments = kite.instruments(exchange)
    _INSTR_CACHE[key] = (_now(), instruments)
    return instruments


def _nearest_expiry(instruments: List[Dict[str, Any]], symbol: str) -> date:
    exps = sorted(
        {
            r["expiry"].date()
            for r in instruments
            if r.get("name") == symbol and r.get("instrument_type") in ("CE", "PE") and r.get("expiry")
        }
    )
    today = date.today()
    for d in exps:
        if d >= today:
            return d
    return exps[-1] if exps else today


def _round_to_strike(px: float, step: int = 50) -> int:
    return int(round(px / step) * step)


def get_instrument_tokens(
    kite: KiteConnect,
    *,
    symbol: str = "NIFTY",
    exchange: str = "NFO",
    expiry: Optional[str] = None,         # "YYYY-MM-DD"; if None -> nearest
    strike: Optional[int] = None,
    spot_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Robust token selector. Accepts either `strike` or `spot_price`.
    Returns:
      {
        "spot_token": 256265,         # if available in your settings
        "atm_strike": 25100,
        "target_strike": 25100,
        "expiry": "YYYY-MM-DD",
        "tokens": {"ce": 123, "pe": 456}
      }
    """
    instruments = _get_instruments(kite, exchange=exchange)

    # expiry resolve
    if expiry:
        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        except Exception:
            exp_dt = _nearest_expiry(instruments, symbol)
    else:
        exp_dt = _nearest_expiry(instruments, symbol)

    # strike resolve
    resolved_strike: int
    if strike is not None:
        resolved_strike = int(strike)
    elif spot_price is not None:
        resolved_strike = _round_to_strike(float(spot_price), step=50)
    else:
        raise ValueError("get_instrument_tokens: provide either strike or spot_price")

    # filter rows
    rows = [r for r in instruments if r.get("name") == symbol and r.get("expiry") and r["expiry"].date() == exp_dt]
    ce_rows = [r for r in rows if r.get("instrument_type") == "CE" and int(r.get("strike", 0)) == resolved_strike]
    pe_rows = [r for r in rows if r.get("instrument_type") == "PE" and int(r.get("strike", 0)) == resolved_strike]

    if not ce_rows or not pe_rows:
        # try a single refresh in case cache is stale
        instruments = kite.instruments(exchange)
        _INSTR_CACHE[f"instruments::{exchange}"] = (_now(), instruments)
        rows = [r for r in instruments if r.get("name") == symbol and r.get("expiry") and r["expiry"].date() == exp_dt]
        ce_rows = [r for r in rows if r.get("instrument_type") == "CE" and int(r.get("strike", 0)) == resolved_strike]
        pe_rows = [r for r in rows if r.get("instrument_type") == "PE" and int(r.get("strike", 0)) == resolved_strike]

    if not ce_rows or not pe_rows:
        raise RuntimeError(f"Could not resolve CE/PE tokens for {symbol} {exp_dt} {resolved_strike}")

    return {
        "atm_strike": resolved_strike,
        "target_strike": resolved_strike,
        "expiry": exp_dt.isoformat(),
        "tokens": {"ce": int(ce_rows[0]["instrument_token"]), "pe": int(pe_rows[0]["instrument_token"])},
    }