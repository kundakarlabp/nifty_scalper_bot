from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

from src.config import (
    MICRO__DEPTH_MULTIPLIER,
    MICRO__REQUIRE_DEPTH,
    MICRO__STALE_MS,
    RISK__EXPOSURE_CAP_PCT,
    settings,
)
from src.logs import structured_log
from src.signals.micro_filters import (
    cap_for_mid,
    depth_required_lots,
    micro_check,
    _quote_reference_price,
    _top5_quantities,
)


# ---------------------------------------------------------------------------
# Configurable microstructure guards
# ---------------------------------------------------------------------------
# These caps are intentionally lightweight and sourced from ``settings.micro``.
# Values are expressed in percentage points (0.35 => 0.35%).
_MICRO_CFG = getattr(settings, "micro", None)
MICRO_SPREAD_CAP: float = float(getattr(_MICRO_CFG, "spread_cap_pct", 0.35))
# Maximum time (seconds) to wait for acceptable microstructure before aborting entry.
ENTRY_WAIT_S: float = float(getattr(_MICRO_CFG, "entry_wait_seconds", 8.0))

__all__ = [
    "MICRO_SPREAD_CAP",
    "ENTRY_WAIT_S",
    "micro_from_l1",
    "micro_from_quote",
    "depth_to_lots",
    "cap_for_mid",
    "depth_required_lots",
    "evaluate_micro",
]


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
    if not q:
        return None, None

    depth = q.get("depth")
    if not isinstance(depth, Mapping):
        return None, None

    buy_levels = depth.get("buy") if isinstance(depth, Mapping) else None
    sell_levels = depth.get("sell") if isinstance(depth, Mapping) else None
    buy_levels = buy_levels if isinstance(buy_levels, Sequence) else []
    sell_levels = sell_levels if isinstance(sell_levels, Sequence) else []
    buy_qty = _top5_quantities(q.get("bid5_qty"), buy_levels)
    sell_qty = _top5_quantities(q.get("ask5_qty"), sell_levels)

    if not buy_qty or not sell_qty:
        return None, None

    try:
        bid = float(buy_levels[0].get("price", 0.0)) if buy_levels else 0.0
        ask = float(sell_levels[0].get("price", 0.0)) if sell_levels else 0.0
    except Exception:
        bid = ask = 0.0
    ltp = _quote_reference_price(q)
    if bid <= 0 or ask <= 0:
        spread = 999.0
    else:
        ref_price = ltp if ltp > 0 else (bid + ask) / 2.0
        spread = abs(ask - bid) / max(ref_price, 1e-6) * 100.0
    need_units = max(int(depth_min_lots), 0) * max(int(lot_size), 0)
    depth_ok = True
    if need_units > 0:
        depth_ok = min(sum(buy_qty), sum(sell_qty)) >= need_units
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


def evaluate_micro(
    q: Optional[Dict[str, Any]],
    *,
    lot_size: int,
    atr_pct: Optional[float],
    cfg: Any,
    side: Optional[str] = None,
    lots: Optional[int] = None,
    depth_multiplier: Optional[float] = None,
    require_depth: Optional[bool] = None,
    trace_id: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate microstructure metrics and gating conditions."""

    micro = micro_check(
        q,
        lot_size=lot_size,
        atr_pct=atr_pct,
        cfg=cfg,
        side=side,
        lots=lots,
        depth_multiplier=depth_multiplier,
        require_depth=require_depth,
        mode_override=mode_override,
    )

    need_units = (micro.get("need_lots") or 0) * (micro.get("lot_size") or 0)
    payload = {
        "trace_id": trace_id,
        "mode": micro.get("mode"),
        "side": micro.get("side"),
        "need_lots": micro.get("need_lots"),
        "lot_size": micro.get("lot_size"),
        "need_units": need_units,
        "depth_multiplier": micro.get("depth_multiplier"),
        "require_depth": micro.get("require_depth"),
        "atr_pct": float(atr_pct) if atr_pct is not None else None,
        "spread_pct": micro.get("spread_pct"),
        "cap_pct": micro.get("cap_pct"),
        "exposure_cap_pct": micro.get("exposure_cap_pct"),
        "depth_ok": micro.get("depth_ok"),
        "depth_available": micro.get("depth_available"),
        "available_bid_qty": micro.get("available_bid_qty"),
        "available_ask_qty": micro.get("available_ask_qty"),
        "spread_block": micro.get("spread_block"),
        "depth_block": micro.get("depth_block"),
        "raw_block": micro.get("raw_block"),
        "would_block": micro.get("would_block"),
        "reason": micro.get("reason"),
        "bid": micro.get("bid"),
        "ask": micro.get("ask"),
        "mid": micro.get("mid"),
        "last_tick_age_ms": micro.get("last_tick_age_ms"),
        "required_qty": micro.get("required_qty"),
    }

    structured_log.event("micro_eval", **payload)
    reason = str(micro.get("reason") or "")
    if reason in {"no_quote", "stale_quote", "no_depth"}:
        structured_log.event("micro_wait", **payload)
    elif micro.get("would_block"):
        structured_log.event("micro_block", **payload)
    else:
        structured_log.event("micro_pass", **payload)

    return micro


