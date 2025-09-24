from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

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
    "micro_from_quote",
    "depth_to_lots",
    "cap_for_mid",
    "depth_required_lots",
    "micro_check",
    "evaluate_micro",
]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
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


def _quote_age_ms(data: Mapping[str, Any]) -> float | None:
    age = data.get("age_ms")
    if age is not None:
        try:
            return float(age)
        except (TypeError, ValueError):
            return None

    ts_val = data.get("ts_ms") or data.get("timestamp")
    if isinstance(ts_val, (int, float)):
        ts_ms_val = float(ts_val)
        if ts_ms_val < 1e12:
            ts_ms_val *= 1000.0
        return max(0.0, time.time() * 1000.0 - ts_ms_val)
    return None


def micro_from_quote(
    q: dict[str, Any] | None, *, lot_size: int, depth_min_lots: int
) -> tuple[float | None, bool | None]:
    """Compute spread% and depth flag from a quote payload."""

    if depth_min_lots <= 0:
        depth_units = 0
        require_depth = False
    else:
        depth_units = max(int(depth_min_lots), 0) * max(int(lot_size), 0)
        require_depth = True

    check = micro_check(
        q,
        lot_size=lot_size,
        depth_required_units=depth_units,
        require_depth=require_depth,
        stale_ms=float(settings.MICRO__STALE_MS),
    )
    spread = check.get("spread_pct")
    if spread is None:
        bid = check.get("bid")
        ask = check.get("ask")
        if (bid is None) ^ (ask is None):
            other_price = ask if bid is None else bid
            if other_price and other_price > 0.0:
                spread = 999.0

    depth_flag = check.get("depth_ok")
    if depth_flag is None and check.get("has_depth"):
        depth_flag = True
    return spread, depth_flag


def depth_to_lots(depth: Any, *, lot_size: int) -> float | None:
    """Normalize depth information to lots."""

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


def _micro_cfg(cfg: Any) -> dict[str, Any]:
    """Extract the ``micro`` configuration section from ``cfg``."""

    if hasattr(cfg, "micro"):
        return cfg.micro or {}
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
    quote: Mapping[str, Any] | None,
    *,
    lot_size: int,
    depth_required_units: int,
    require_depth: bool,
    stale_ms: float,
    side: str | None = None,
) -> dict[str, Any]:
    """Normalise quote microstructure fields and evaluate basic freshness checks."""

    result: dict[str, Any] = {
        "bid": None,
        "ask": None,
        "mid": None,
        "ltp": None,
        "spread_pct": None,
        "has_depth": False,
        "available_bid_qty": 0,
        "available_ask_qty": 0,
        "depth_required": max(int(depth_required_units), 0),
        "depth_ok": None,
        "reason": "no_quote" if quote is None else "ok",
        "last_tick_age_ms": None,
        "side": side.upper() if side else None,
    }

    if not quote:
        return result

    data = quote if isinstance(quote, Mapping) else {}
    depth = data.get("depth") if isinstance(data, Mapping) else None
    if not isinstance(depth, Mapping):
        depth = {}

    buy_levels = depth.get("buy") if isinstance(depth, Mapping) else None
    sell_levels = depth.get("sell") if isinstance(depth, Mapping) else None
    buy_levels = list(buy_levels) if isinstance(buy_levels, Sequence) else []
    sell_levels = list(sell_levels) if isinstance(sell_levels, Sequence) else []

    def _level_price(levels: Sequence[Any]) -> float:
        if not levels:
            return 0.0
        head = levels[0]
        if isinstance(head, Mapping):
            return _as_float(head.get("price"))
        return _as_float(getattr(head, "price", 0.0))

    bid = _as_float(data.get("bid"))
    ask = _as_float(data.get("ask"))
    if bid <= 0.0:
        bid = _level_price(buy_levels)
    if ask <= 0.0:
        ask = _level_price(sell_levels)

    bid5 = _top5_quantities(data.get("bid5_qty"), buy_levels)
    ask5 = _top5_quantities(data.get("ask5_qty"), sell_levels)
    available_bid = sum(bid5)
    available_ask = sum(ask5)

    ltp = _quote_reference_price(data)
    mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else None
    if mid is None and ltp > 0.0:
        mid = ltp

    spread_pct: float | None
    if bid > 0.0 and ask > 0.0 and mid:
        spread_pct = abs(ask - bid) / max(mid, 1e-6) * 100.0
    else:
        spread_pct = None

    age_ms = _quote_age_ms(data) or 0.0
    result.update(
        {
            "bid": bid if bid > 0.0 else None,
            "ask": ask if ask > 0.0 else None,
            "mid": mid,
            "ltp": ltp if ltp > 0.0 else None,
            "spread_pct": spread_pct,
            "available_bid_qty": available_bid,
            "available_ask_qty": available_ask,
            "has_depth": bool(available_bid and available_ask),
            "last_tick_age_ms": age_ms,
        }
    )

    side_norm = result["side"]
    if side_norm == "SELL":
        depth_available = available_bid
    elif side_norm == "BUY":
        depth_available = available_ask
    else:
        depth_available = min(available_bid, available_ask)

    depth_required = result["depth_required"]
    if require_depth:
        if not result["has_depth"]:
            depth_ok = False
        elif depth_required > 0:
            depth_ok = depth_available >= depth_required
        else:
            depth_ok = True
    else:
        depth_ok = True if result["has_depth"] else None

    if result["bid"] is None or result["ask"] is None:
        depth_ok = None

    result["depth_ok"] = depth_ok

    reason = "ok"
    if age_ms and age_ms > max(float(stale_ms), 0.0):
        reason = "stale_quote"
    elif (
        result["bid"] is None
        or result["ask"] is None
        or (not result["has_depth"] and not buy_levels and not sell_levels)
    ):
        reason = "no_quote"
    elif require_depth and depth_ok is False:
        depth_inputs_present = bool(buy_levels or sell_levels or bid5 or ask5)
        if not depth_inputs_present:
            reason = "no_quote"
        else:
            reason = "depth_fail" if depth_available > 0 else "no_depth"

    result["reason"] = reason
    return result


def evaluate_micro(
    q: dict[str, Any] | None,
    *,
    lot_size: int,
    atr_pct: float | None,
    cfg: Any,
    side: str | None = None,
    lots: int | None = None,
    depth_multiplier: float | None = None,
    require_depth: bool | None = None,
    trace_id: str | None = None,
    mode_override: str | None = None,
) -> dict[str, Any]:
    """Evaluate microstructure metrics and gating conditions."""

    micro_cfg = _micro_cfg(cfg)
    mode_raw = mode_override or micro_cfg.get("mode", "HARD")
    mode = str(mode_raw or "HARD").upper()
    if mode not in {"HARD", "SOFT"}:
        mode = "HARD"
    hard_mode = mode == "HARD"
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

    if depth_multiplier is not None:
        depth_mult = float(depth_multiplier)
    elif micro_cfg.get("depth_multiplier") is not None:
        try:
            depth_mult = float(micro_cfg.get("depth_multiplier", settings.MICRO__DEPTH_MULTIPLIER))
        except Exception:
            depth_mult = float(settings.MICRO__DEPTH_MULTIPLIER)
    else:
        depth_mult = float(settings.MICRO__DEPTH_MULTIPLIER)

    if require_depth is not None:
        require_depth_flag = bool(require_depth)
    elif micro_cfg.get("require_depth") is not None:
        require_depth_flag = bool(micro_cfg.get("require_depth"))
    else:
        require_depth_flag = bool(settings.MICRO__REQUIRE_DEPTH)

    need_units = lots_val * lot_units
    required_units = int(math.ceil(need_units * depth_mult)) if need_units else 0

    stale_ms = float(micro_cfg.get("stale_ms", settings.MICRO__STALE_MS))

    base_context: dict[str, Any] = {
        "trace_id": trace_id,
        "mode": mode,
        "side": side_norm,
        "need_lots": lots_val,
        "lot_size": lot_units,
        "depth_multiplier": depth_mult,
        "require_depth": require_depth_flag,
        "atr_pct": float(atr_pct) if atr_pct is not None else None,
        "required_qty": required_units,
    }

    check = micro_check(
        q,
        lot_size=lot_units,
        depth_required_units=required_units,
        require_depth=require_depth_flag,
        stale_ms=stale_ms,
        side=side_norm,
    )

    mid_for_cap = check.get("mid") or check.get("ltp") or 0.0
    cap_pct = cap_for_mid(float(mid_for_cap), cfg)

    spread_pct = check.get("spread_pct")
    spread_block = spread_pct is None or (spread_pct > cap_pct)
    depth_block = require_depth_flag and check.get("depth_ok") is False
    reason = str(check.get("reason") or "ok")

    if reason in {"depth_fail", "no_depth"}:
        reason = "depth"

    if reason == "ok":
        if spread_block:
            reason = "spread"
        elif depth_block:
            reason = "depth"

    hard_fail = reason in {"no_quote", "stale_quote"}
    raw_block = hard_fail or spread_block or depth_block
    would_block = raw_block if hard_mode else (hard_fail or depth_block or spread_block)

    depth_available = min(
        check.get("available_bid_qty", 0), check.get("available_ask_qty", 0)
    )

    payload = {
        **base_context,
        **{
            "cap_pct": cap_pct,
            "spread_pct": spread_pct,
            "spread_block": spread_block,
            "depth_ok": check.get("depth_ok"),
            "depth_available": depth_available,
            "available_bid_qty": check.get("available_bid_qty"),
            "available_ask_qty": check.get("available_ask_qty"),
            "depth_block": depth_block,
            "last_tick_age_ms": check.get("last_tick_age_ms"),
            "reason": reason,
            "raw_block": raw_block,
            "would_block": would_block,
        },
    }

    structured_log.event("micro_eval", **payload)
    if hard_fail:
        structured_log.event("micro_wait", **payload)
    elif would_block:
        structured_log.event("micro_block", **payload)
    else:
        structured_log.event("micro_pass", **payload)

    result = {
        **check,
        "mode": mode,
        "cap_pct": cap_pct,
        "need_lots": lots_val,
        "lot_size": lot_units,
        "required_qty": required_units,
        "depth_multiplier": depth_mult,
        "require_depth": require_depth_flag,
        "depth_available": depth_available,
        "reason": reason,
        "would_block": would_block,
    }
    return result
