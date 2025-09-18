from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

from src.config import settings
from src.logs import structured_log


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
    if bid <= 0 or ask <= 0:
        return None, None
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / ((bid + ask) / 2.0) * 100.0
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
    side: Optional[str] = None,
    lots: Optional[int] = None,
    depth_multiplier: Optional[float] = None,
    require_depth: Optional[bool] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate microstructure metrics and gating conditions.

    Returns a dictionary with spread %, depth flag and gating decision.
    """

    micro_cfg = _micro_cfg(cfg)
    mode = str(micro_cfg.get("mode", "HARD")).upper()
    hard = mode == "HARD"
    side_norm = str(side).upper() if side else None

    lots_val: int
    if lots is not None:
        lots_val = max(int(lots), 0)
    elif micro_cfg.get("depth_min_lots") is not None:
        try:
            lots_val = max(int(micro_cfg.get("depth_min_lots", 0)), 0)
        except Exception:
            lots_val = depth_required_lots(float(atr_pct or 0.0))
    else:
        lots_val = depth_required_lots(float(atr_pct or 0.0))

    lot_units = max(int(lot_size), 0)
    need_units = lots_val * lot_units

    depth_mult = (
        float(depth_multiplier)
        if depth_multiplier is not None
        else float(
            micro_cfg.get(
                "depth_multiplier",
                getattr(settings.executor, "depth_multiplier", 1.0),
            )
        )
    )
    require_depth_flag = (
        bool(require_depth)
        if require_depth is not None
        else bool(
            micro_cfg.get(
                "require_depth", getattr(settings.executor, "require_depth", False)
            )
        )
    )
    required_units = int(math.ceil(need_units * depth_mult)) if need_units else 0

    base_context: Dict[str, Any] = {
        "trace_id": trace_id,
        "mode": mode,
        "side": side_norm,
        "need_lots": lots_val,
        "lot_size": lot_units,
        "need_units": need_units,
        "depth_multiplier": depth_mult,
        "require_depth": require_depth_flag,
        "atr_pct": float(atr_pct) if atr_pct is not None else None,
    }
    cap_default = float(micro_cfg.get("max_spread_pct", 1.0))

    def emit(event: str, **fields: Any) -> None:
        payload = dict(base_context)
        payload.update(fields)
        structured_log.event(event, **payload)

    def log_outcome(outcome: str, **fields: Any) -> None:
        metrics = dict(fields)
        metrics.setdefault("required_qty", required_units)
        emit("micro_eval", **metrics)
        emit(outcome, **metrics)

    if not q:
        log_outcome(
            "micro_wait",
            reason="no_quote",
            spread_pct=None,
            cap_pct=cap_default,
            depth_ok=None,
            depth_available=None,
            available_bid_qty=None,
            available_ask_qty=None,
            spread_block=None,
            depth_block=None,
            raw_block=True,
            would_block=hard,
        )
        return {
            "spread_pct": None,
            "depth_ok": None,
            "mode": mode,
            "reason": "no_quote",
            "would_block": hard,
        }

    bid5 = _top5_quantities(q.get("bid5_qty"), [])
    ask5 = _top5_quantities(q.get("ask5_qty"), [])

    bid = _as_float(q.get("bid"))
    ask = _as_float(q.get("ask"))

    if not bid5 or not ask5:
        log_outcome(
            "micro_wait",
            reason="no_quote",
            spread_pct=None,
            cap_pct=cap_default,
            depth_ok=None,
            depth_available=None,
            available_bid_qty=sum(bid5),
            available_ask_qty=sum(ask5),
            spread_block=None,
            depth_block=None,
            raw_block=True,
            would_block=hard,
            bid=bid,
            ask=ask,
        )
        return {
            "spread_pct": None,
            "depth_ok": None,
            "mode": mode,
            "reason": "no_quote",
            "would_block": hard,
        }

    if bid <= 0.0 or ask <= 0.0:
        log_outcome(
            "micro_wait",
            reason="no_quote",
            spread_pct=None,
            cap_pct=cap_default,
            depth_ok=None,
            depth_available=None,
            available_bid_qty=sum(bid5),
            available_ask_qty=sum(ask5),
            spread_block=None,
            depth_block=None,
            raw_block=True,
            would_block=hard,
            bid=bid,
            ask=ask,
        )
        return {
            "spread_pct": None,
            "depth_ok": None,
            "mode": mode,
            "reason": "no_quote",
            "would_block": hard,
        }

    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else None
    spread_pct = (
        (ask - bid) / ((bid + ask) / 2.0) * 100.0 if bid > 0 and ask > 0 else None
    )
    cap = cap_for_mid(mid or 0.0, cfg)

    available_bid = sum(bid5)
    available_ask = sum(ask5)
    if side_norm == "SELL":
        depth_available = available_bid
    elif side_norm == "BUY":
        depth_available = available_ask
    else:
        depth_available = min(available_bid, available_ask)

    if require_depth_flag:
        depth_ok = depth_available >= required_units
    else:
        depth_ok = True

    spread_block = spread_pct is None or (spread_pct > cap)
    depth_block = depth_ok is False
    raw_block = spread_block or depth_block
    final_block = raw_block if hard else False
    reason = "spread" if spread_block else "depth" if depth_block else "ok"

    log_outcome(
        "micro_block" if final_block else "micro_pass",
        reason=reason,
        spread_pct=spread_pct,
        cap_pct=cap,
        depth_ok=depth_ok,
        depth_available=depth_available,
        available_bid_qty=available_bid,
        available_ask_qty=available_ask,
        spread_block=spread_block,
        depth_block=depth_block,
        raw_block=raw_block,
        would_block=final_block,
        bid=bid,
        ask=ask,
        mid=mid,
    )

    return {
        "mid": mid,
        "spread_pct": spread_pct,
        "cap_pct": cap,
        "depth_ok": depth_ok,
        "need_lots": lots_val,
        "lot_size": lot_units,
        "required_qty": required_units,
        "depth_available": depth_available,
        "bid_top5": bid5,
        "ask_top5": ask5,
        "bid": bid,
        "ask": ask,
        "side": side_norm,
        "depth_multiplier": depth_mult,
        "require_depth": require_depth_flag,
        "mode": mode,
        "would_block": final_block,
    }


def _as_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _as_int(val: Any) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def _top5_quantities(raw: Any, fallback_levels: Sequence[Any]) -> list[int]:
    """Return sanitized top-5 depth quantities as integers."""

    values: list[int] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in list(raw)[:5]:
            if isinstance(item, Mapping):
                qty = _as_int(item.get("quantity"))
            else:
                qty = _as_int(item)
            values.append(max(qty, 0))
    if values:
        return values

    for lvl in list(fallback_levels)[:5]:
        if isinstance(lvl, Mapping):
            qty = _as_int(lvl.get("quantity"))
        else:
            qty = _as_int(getattr(lvl, "quantity", 0))
        values.append(max(qty, 0))
    return values
