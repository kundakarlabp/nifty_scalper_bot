# src/utils/account_info.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from kiteconnect import KiteConnect
from src.config import Config

logger = logging.getLogger(__name__)


# ----------------------------- Kite bootstrap ----------------------------- #

def _build_kite() -> Optional[KiteConnect]:
    """Create a KiteConnect client using Config; return None if creds missing."""
    api_key = getattr(Config, "ZERODHA_API_KEY", None)
    access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)

    if not api_key or not access_token:
        logger.error("Kite credentials missing (ZERODHA_API_KEY / KITE_ACCESS_TOKEN).")
        return None

    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return kite
    except Exception as e:
        logger.error("Failed to initialise KiteConnect: %s", e, exc_info=True)
        return None


# ----------------------------- Margins helpers ---------------------------- #

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        return f if f == f else default  # NaN guard
    except Exception:
        return default


def get_margins(segment: str = "equity") -> Dict[str, float]:
    """
    Fetch margin snapshot for a given segment ('equity' or 'commodity').

    Returns:
        {
          "available_cash": float,
          "available_net": float,
          "utilised_total": float,
          "utilisation_pct": float,   # utilised_total / (available_cash + utilised_total)
        }
    """
    kite = _build_kite()
    if not kite:
        return {"available_cash": 0.0, "available_net": 0.0, "utilised_total": 0.0, "utilisation_pct": 0.0}

    try:
        m = kite.margins(segment=segment)
    except Exception as e:
        logger.error("margins(%s) failed: %s", segment, e, exc_info=True)
        return {"available_cash": 0.0, "available_net": 0.0, "utilised_total": 0.0, "utilisation_pct": 0.0}

    available = m.get("available", {}) if isinstance(m, dict) else {}
    utilised = m.get("utilised", {}) if isinstance(m, dict) else {}

    cash = _safe_float(available.get("cash", 0.0))
    net = _safe_float(available.get("net", cash))
    # Zerodha provides detailed utilised fields; take a conservative total
    utilised_total = 0.0
    try:
        if isinstance(utilised, dict):
            for v in utilised.values():
                utilised_total += max(0.0, _safe_float(v, 0.0))
    except Exception:
        utilised_total = _safe_float(m.get("utilised", 0.0), 0.0)

    denom = cash + utilised_total
    utilisation_pct = (utilised_total / denom) * 100.0 if denom > 0 else 0.0

    return {
        "available_cash": cash,
        "available_net": net,
        "utilised_total": utilised_total,
        "utilisation_pct": round(utilisation_pct, 2),
    }


def get_all_margins() -> Dict[str, Dict[str, float]]:
    """Convenience: margins for both segments."""
    eq = get_margins("equity")
    com = get_margins("commodity")
    return {"equity": eq, "commodity": com}


def get_dynamic_account_balance(use_net: bool = False, segment: str = "equity") -> float:
    """
    Return current balance for a segment.
    Args:
        use_net: if True, return 'available_net'; else 'available_cash'
        segment: 'equity' or 'commodity'
    """
    m = get_margins(segment=segment)
    return float(m["available_net"] if use_net else m["available_cash"])


def get_margin_utilisation(segment: str = "equity") -> float:
    """Return current utilisation percentage for a segment."""
    return float(get_margins(segment).get("utilisation_pct", 0.0))


# ----------------------------- P&L (positions) ---------------------------- #

def _sum_pnl_fields(pos: Dict[str, Any]) -> Tuple[float, float]:
    """
    Try to read realised/unrealised P&L from Zerodha position row.
    Falls back to 'pnl' if present.
    """
    realised = _safe_float(pos.get("realised", pos.get("realised_pnl", 0.0)), 0.0)
    unrealised = _safe_float(pos.get("unrealised", pos.get("unrealised_pnl", 0.0)), 0.0)
    if realised == 0.0 and unrealised == 0.0:
        # Some APIs provide only 'pnl'
        pnl = _safe_float(pos.get("pnl", 0.0), 0.0)
        # assume it's unrealised for live net positions
        unrealised = pnl
    return realised, unrealised


def _approx_unrealised_from_ltp(pos: Dict[str, Any], kite: KiteConnect) -> float:
    """
    As a last resort, approximate unrealised P&L using LTP if price fields are present.
    """
    try:
        qty = _safe_float(pos.get("quantity", pos.get("net_quantity", 0)))
        if qty == 0:
            return 0.0
        avg = _safe_float(pos.get("average_price", 0.0))
        tradingsymbol = pos.get("tradingsymbol") or ""
        exchange = pos.get("exchange") or pos.get("product") or "NSE"
        key = f"{exchange}:{tradingsymbol}"
        ltp_map = kite.ltp([key]) or {}
        ltp = _safe_float((ltp_map.get(key) or {}).get("last_price"))
        return (ltp - avg) * qty
    except Exception:
        return 0.0


def get_positions_pnl() -> Dict[str, float]:
    """
    Aggregate realised and unrealised P&L across all positions.
    Returns:
        { "realised": float, "unrealised": float, "total": float }
    """
    kite = _build_kite()
    if not kite:
        return {"realised": 0.0, "unrealised": 0.0, "total": 0.0}

    realised_sum = 0.0
    unrealised_sum = 0.0

    try:
        pos = kite.positions() or {}
        # Zerodha returns { "day": [...], "net": [...] }
        buckets = []
        if isinstance(pos, dict):
            buckets.extend(pos.get("net") or [])
            buckets.extend(pos.get("day") or [])
        elif isinstance(pos, list):
            buckets = pos

        for p in buckets:
            r, u = _sum_pnl_fields(p)
            # If both were zero and we have price fields, try to approximate
            if r == 0.0 and u == 0.0 and p.get("average_price") and p.get("tradingsymbol"):
                u = _approx_unrealised_from_ltp(p, kite)
            realised_sum += r
            unrealised_sum += u

    except Exception as e:
        logger.error("positions() fetch failed: %s", e, exc_info=True)

    total = realised_sum + unrealised_sum
    return {
        "realised": round(realised_sum, 2),
        "unrealised": round(unrealised_sum, 2),
        "total": round(total, 2),
    }


# ----------------------------- Snapshot helper ---------------------------- #

def get_account_snapshot(use_net: bool = False) -> Dict[str, Any]:
    """
    One-shot view of account health for dashboards/circuit-breakers.
    """
    eq = get_margins("equity")
    com = get_margins("commodity")
    pnl = get_positions_pnl()

    balance = eq["available_net"] if use_net else eq["available_cash"]
    snapshot = {
        "balance": round(float(balance), 2),
        "equity": eq,
        "commodity": com,
        "pnl": pnl,
    }
    return snapshot