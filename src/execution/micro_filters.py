from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def micro_from_l1(l1: Optional[Dict[str, Any]], *, lot_size: int, depth_min_lots: int) -> Tuple[Optional[float], Optional[bool], Optional[Dict[str, Any]]]:
    """Compute microstructure metrics from level-1 data."""
    if not l1 or "depth" not in l1:
        return None, None, None
    try:
        b = l1["depth"]["buy"][0]
        s = l1["depth"]["sell"][0]
        bid = float(b.get("price", 0.0))
        ask = float(s.get("price", 0.0))
        bq = int(b.get("quantity", 0))
        sq = int(s.get("quantity", 0))
    except Exception:
        return None, None, None
    if bid <= 0 or ask <= 0:
        return None, None, None
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0
    depth_ok = min(bq, sq) >= depth_min_lots * lot_size
    return spread_pct, depth_ok, {"bid": bid, "ask": ask, "bid5": bq, "ask5": sq}


def micro_from_quote(q: Optional[Dict[str, Any]], *, lot_size: int, depth_min_lots: int) -> Tuple[Optional[float], Optional[bool]]:
    """Compute spread% and depth flag from a quote payload.

    Parameters
    ----------
    q: Quote dictionary with ``depth`` field as returned by ``kite.quote``.
    lot_size: Contract lot size.
    depth_min_lots: Minimum lot depth required on both sides.
    """
    if not q or "depth" not in q:
        return None, None
    try:
        b = q["depth"]["buy"][0]
        s = q["depth"]["sell"][0]
        bid = float(b["price"])
        ask = float(s["price"])
        bq = int(b["quantity"])
        sq = int(s["quantity"])
    except Exception:
        return None, None
    if bid <= 0 or ask <= 0:
        return None, None
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid * 100.0
    depth_ok = min(bq, sq) >= depth_min_lots * lot_size
    return spread, depth_ok
