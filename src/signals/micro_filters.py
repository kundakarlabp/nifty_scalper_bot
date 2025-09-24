"""Shared microstructure evaluation helpers used by signals and execution.

This module hosts a lightweight :func:`micro_check` helper that mirrors the
behaviour of the execution level microstructure filters without emitting any
logs.  It normalises configuration coming from either dataclasses or plain
dicts, enforces quote freshness and depth requirements, and returns the
canonical dictionary describing the decision.  The helper can therefore be
used by the signal generation pipeline to reason about gates while keeping the
core execution path single sourced.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

from src.config import (
    MICRO__DEPTH_MULTIPLIER,
    MICRO__REQUIRE_DEPTH,
    MICRO__STALE_MS,
    RISK__EXPOSURE_CAP_PCT,
    settings,
)

__all__ = ["cap_for_mid", "depth_required_lots", "micro_check"]


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
    """Return sanitised top-5 depth quantities as integers."""

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


def _quote_reference_price(data: Mapping[str, Any] | None) -> float:
    """Return a positive reference price from ``data`` if available."""

    if not isinstance(data, Mapping):
        return 0.0

    fallback: float = 0.0
    for key in ("ltp", "mid", "last_price", "close", "close_price"):
        value = data.get(key)
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0.0:
            return price
        if fallback == 0.0 and price != 0.0:
            fallback = price
    return fallback


def _micro_cfg(cfg: Any) -> Dict[str, Any]:
    """Extract the ``micro`` configuration section from ``cfg``."""

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


def micro_check(
    q: Mapping[str, Any] | None,
    *,
    lot_size: int,
    atr_pct: Optional[float],
    cfg: Any,
    side: Optional[str] = None,
    lots: Optional[int] = None,
    depth_multiplier: Optional[float] = None,
    require_depth: Optional[bool] = None,
    mode_override: Optional[str] = None,
    now_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate microstructure metrics and gating conditions without logging."""

    micro_cfg = _micro_cfg(cfg)
    if mode_override is not None:
        mode = str(mode_override).upper() or "HARD"
    else:
        mode = str(micro_cfg.get("mode", "HARD")).upper()
    if mode not in {"HARD", "SOFT"}:
        mode = "HARD"
    side_norm = str(side).upper() if side else None

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

    if depth_multiplier is not None:
        depth_mult = float(depth_multiplier)
    elif micro_cfg.get("depth_multiplier") is not None:
        try:
            depth_mult = float(
                micro_cfg.get("depth_multiplier", MICRO__DEPTH_MULTIPLIER)
            )
        except Exception:
            depth_mult = float(
                getattr(settings, "MICRO__DEPTH_MULTIPLIER", MICRO__DEPTH_MULTIPLIER)
            )
    else:
        depth_mult = float(
            getattr(settings, "MICRO__DEPTH_MULTIPLIER", MICRO__DEPTH_MULTIPLIER)
        )

    if require_depth is not None:
        require_depth_flag = bool(require_depth)
    elif micro_cfg.get("require_depth") is not None:
        require_depth_flag = bool(micro_cfg.get("require_depth"))
    else:
        require_depth_flag = bool(
            getattr(settings, "MICRO__REQUIRE_DEPTH", MICRO__REQUIRE_DEPTH)
        )

    required_units = int(math.ceil(need_units * depth_mult)) if need_units else 0
    risk_cap_pct = float(
        getattr(settings, "RISK__EXPOSURE_CAP_PCT", RISK__EXPOSURE_CAP_PCT)
    )
    stale_ms_limit = float(getattr(settings, "MICRO__STALE_MS", MICRO__STALE_MS))

    result: Dict[str, Any] = {
        "mode": mode,
        "side": side_norm,
        "need_lots": lots_val,
        "lot_size": lot_units,
        "required_qty": required_units,
        "depth_multiplier": depth_mult,
        "require_depth": require_depth_flag,
        "risk_cap_pct": risk_cap_pct,
        "spread_cap_pct": None,
        "spread_pct": None,
        "depth_ok": None,
        "depth_available": None,
        "bid_top5": [],
        "ask_top5": [],
        "bid": None,
        "ask": None,
        "mid": None,
        "would_block": True,
        "reason": "no_quote",
        "last_tick_age_ms": None,
        "available_bid_qty": None,
        "available_ask_qty": None,
        "spread_block": None,
        "depth_block": None,
        "raw_block": True,
    }

    if not isinstance(q, Mapping):
        return result

    bid = _as_float(q.get("bid"))
    ask = _as_float(q.get("ask"))
    bid_qty_raw = _as_int(q.get("bid_qty"))
    ask_qty_raw = _as_int(q.get("ask_qty"))
    bid_qty = bid_qty_raw if bid_qty_raw > 0 else None
    ask_qty = ask_qty_raw if ask_qty_raw > 0 else None
    bid5 = _top5_quantities(q.get("bid5_qty"), q.get("depth", {}).get("buy", []))
    ask5 = _top5_quantities(q.get("ask5_qty"), q.get("depth", {}).get("sell", []))
    fallback_bid_qty = sum(bid5) if bid5 else None
    fallback_ask_qty = sum(ask5) if ask5 else None
    if bid_qty is None and ask_qty is None and not bid5 and not ask5:
        result["reason"] = "no_quote"
        return result
    result.update(
        {
            "bid_top5": bid5,
            "ask_top5": ask5,
            "available_bid_qty": bid_qty if bid_qty is not None else fallback_bid_qty,
            "available_ask_qty": ask_qty if ask_qty is not None else fallback_ask_qty,
            "bid": bid if bid > 0 else None,
            "ask": ask if ask > 0 else None,
        }
    )

    age_ms: float | None = None
    age_raw = q.get("last_tick_age_ms")
    if age_raw is not None:
        try:
            age_ms = float(age_raw)
        except (TypeError, ValueError):
            age_ms = None
    ts_raw = q.get("age_ms") if age_ms is None else None
    if ts_raw is not None:
        try:
            age_ms = float(ts_raw)
        except (TypeError, ValueError):
            age_ms = None
    if age_ms is None:
        ts_val = q.get("ts_ms") or q.get("timestamp")
        if isinstance(ts_val, (int, float)):
            ts_ms_val = float(ts_val)
            if ts_ms_val < 1e12:
                ts_ms_val *= 1000.0
            now_val = float(now_ms if now_ms is not None else time.time() * 1000.0)
            age_ms = max(0.0, now_val - ts_ms_val)
    result["last_tick_age_ms"] = age_ms

    if bid <= 0.0 or ask <= 0.0:
        result["reason"] = "no_quote"
        return result

    if age_ms is not None and age_ms > stale_ms_limit:
        result["reason"] = "stale_quote"
        return result

    ltp = _quote_reference_price(q)
    mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else None
    if mid is None and ltp > 0.0:
        mid = ltp
    result["mid"] = mid

    ref_price = ltp if ltp > 0.0 else (mid or 0.0)
    spread_pct = abs(ask - bid) / max(ref_price, 1e-6) * 100.0
    spread_cap = cap_for_mid(mid or 0.0, cfg)
    result.update({"spread_pct": spread_pct, "spread_cap_pct": spread_cap})

    if side_norm == "SELL":
        depth_available = bid_qty
    elif side_norm == "BUY":
        depth_available = ask_qty
    else:
        candidates = [qty for qty in (bid_qty, ask_qty) if qty is not None]
        depth_available = min(candidates) if candidates else None
    depth_metric = depth_available
    if depth_metric is None:
        if side_norm == "SELL":
            depth_metric = fallback_bid_qty
        elif side_norm == "BUY":
            depth_metric = fallback_ask_qty
        else:
            candidates_display = [
                val for val in (fallback_bid_qty, fallback_ask_qty) if val is not None
            ]
            depth_metric = min(candidates_display) if candidates_display else None
    result["depth_available"] = depth_metric

    if require_depth_flag:
        if depth_metric is None:
            depth_ok = True
        else:
            depth_ok = depth_metric >= required_units
    else:
        depth_ok = True
    result["depth_ok"] = depth_ok

    spread_block = spread_pct > spread_cap if spread_cap is not None else True
    depth_block = not depth_ok if require_depth_flag else False
    raw_block = spread_block or depth_block
    reason = "spread" if spread_block else "depth" if depth_block else "ok"

    result.update(
        {
            "would_block": raw_block,
            "reason": reason,
            "spread_block": spread_block,
            "depth_block": depth_block,
            "raw_block": raw_block,
        }
    )
    return result
