from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Configurable microstructure guards
# ---------------------------------------------------------------------------
# These caps are intentionally lightweight and can be overridden via
# environment variables at runtime. Values are expressed in percentage points
# (0.35 => 0.35%).
MICRO_SPREAD_CAP: float = float(os.getenv("MICRO_SPREAD_CAP", "0.35"))
# Maximum time to wait for acceptable microstructure before aborting entry.
ENTRY_WAIT_S: float = float(os.getenv("ENTRY_WAIT_S", "8"))


def micro_from_l1(
    l1: Optional[Dict[str, Any]], *, lot_size: int, depth_min_lots: int
) -> Tuple[Optional[float], Optional[bool], Optional[Dict[str, Any]]]:
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


def micro_from_quote(
    q: Optional[Dict[str, Any]], *, lot_size: int, depth_min_lots: int
) -> Tuple[Optional[float], Optional[bool]]:
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


def depth_to_lots(depth: Any, *, lot_size: int) -> Optional[float]:
    """Normalize depth information to lots.

    ``depth`` may be a single quantity (in units) or a tuple/list containing
    bid and ask quantities. The function returns the smaller side expressed in
    lot units. Invalid inputs yield ``None``.
    """

    if depth is None:
        return None
    try:
        if isinstance(depth, (tuple, list)):
            qty = min(float(depth[0]), float(depth[1]))
        else:
            qty = float(depth)
        if lot_size <= 0:
            return None
        return qty / float(lot_size)
    except (TypeError, ValueError):
        return None


def _micro_cfg(cfg: Any) -> Dict[str, Any]:
    """Extract the ``micro`` configuration section from ``cfg``.

    Accepts either a mapping containing ``micro`` or an object with a
    ``micro`` attribute. Missing values yield an empty dict.
    """

    if hasattr(cfg, "micro"):
        return getattr(cfg, "micro") or {}
    if isinstance(cfg, dict):
        return cfg.get("micro", {})
    return {}


def cap_for_mid(mid: float, cfg: Any) -> float:
    """Return the spread cap (%) for a given mid price using ``cfg``."""

    micro = _micro_cfg(cfg)
    if not micro.get("dynamic", True):
        return float(micro.get("max_spread_pct", 1.0))
    cap = float(micro.get("max_spread_pct", 1.0))
    for row in micro.get("table", []):
        try:
            if mid >= float(row.get("min_mid")):
                cap = float(row.get("cap_pct"))
        except Exception:
            continue
    return cap


def depth_required_lots(atr_pct: float) -> int:
    """Determine minimum depth (in lots) required based on ATR%."""

    return 1 if atr_pct >= 0.02 else 2


def evaluate_micro(
    q: Optional[Dict[str, Any]],
    *,
    lot_size: int,
    atr_pct: Optional[float],
    cfg: Any,
) -> Dict[str, Any]:
    """Evaluate microstructure metrics and gating conditions.

    Returns a dictionary with spread %, depth flag and gating decision.
    """

    micro_cfg = _micro_cfg(cfg)
    mode = str(micro_cfg.get("mode", "HARD")).upper()
    hard = mode == "HARD"

    if not q or "bid" not in q or "ask" not in q:
        return {
            "spread_pct": None,
            "depth_ok": None,
            "mode": mode,
            "reason": "no_quote",
            "would_block": hard,
        }

    try:
        bid = float(q.get("bid", 0.0))
        ask = float(q.get("ask", 0.0))
    except Exception:
        bid = ask = 0.0
    if bid <= 0 or ask <= 0:
        return {
            "spread_pct": None,
            "depth_ok": None,
            "mode": mode,
            "reason": "bad_quote",
            "would_block": hard,
        }

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0
    cap = cap_for_mid(mid, cfg)
    bq = int(q.get("bid_qty") or q.get("bid5_qty") or 0)
    sq = int(q.get("ask_qty") or q.get("ask5_qty") or 0)
    need_lots = depth_required_lots(float(atr_pct or 0.0))
    depth_ok = min(bq, sq) >= need_lots * lot_size
    would_block = (spread_pct > cap) or (not depth_ok)
    return {
        "mid": mid,
        "spread_pct": spread_pct,
        "cap_pct": cap,
        "depth_ok": depth_ok,
        "need_lots": need_lots,
        "lot_size": lot_size,
        "bid_qty": bq,
        "ask_qty": sq,
        "bid": bid,
        "ask": ask,
        "mode": mode,
        "would_block": would_block if hard else False,
    }
