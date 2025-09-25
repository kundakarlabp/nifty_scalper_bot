"""Helpers for evaluating microstructure guard conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional during tests
    from src.config import settings
except Exception:  # pragma: no cover - settings may be absent in tests
    settings = None  # type: ignore

LOT_SIZE_DEFAULT = 75
WATCHDOG_STALE_MS_DEFAULT = 3500


def _coerce_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _coerce_int(value: Any) -> Optional[int]:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return None
    return result


def _mask_quantity(raw: Any) -> list[int]:
    values: list[int] = []
    if raw is None:
        return values
    if isinstance(raw, Mapping):
        quantity = raw.get("quantity")
        coerced = _coerce_int(quantity)
        if coerced is not None:
            values.append(max(coerced, 0))
        return values
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in list(raw)[:5]:
            if isinstance(item, Mapping):
                qty = _coerce_int(item.get("quantity"))
            else:
                qty = _coerce_int(item)
            if qty is not None:
                values.append(max(qty, 0))
        return values
    coerced = _coerce_int(raw)
    if coerced is not None:
        values.append(max(coerced, 0))
    return values


def _sum_depth(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        buy = raw.get("buy")
        sell = raw.get("sell")
        if buy is not None or sell is not None:
            buy_vals = _mask_quantity(buy)
            sell_vals = _mask_quantity(sell)
            candidates = []
            if buy_vals:
                candidates.append(sum(buy_vals))
            if sell_vals:
                candidates.append(sum(sell_vals))
            if candidates:
                return min(candidates)
        qty = _coerce_int(raw.get("quantity"))
        if qty is not None:
            return max(qty, 0)
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        vals = _mask_quantity(raw)
        return sum(vals) if vals else None
    qty = _coerce_int(raw)
    if qty is not None:
        return max(qty, 0)
    return None


def _lot_size() -> int:
    try:
        instruments = getattr(settings, "instruments", None)
        if instruments is None:
            return LOT_SIZE_DEFAULT
        lot = getattr(instruments, "nifty_lot_size", LOT_SIZE_DEFAULT)
        coerced = int(lot)
        return coerced if coerced > 0 else LOT_SIZE_DEFAULT
    except Exception:  # pragma: no cover - defensive
        return LOT_SIZE_DEFAULT


def _watchdog_stale_ms() -> int:
    try:
        value = getattr(settings, "WATCHDOG_STALE_MS", None)
        if value is not None:
            coerced = int(float(value))
            if coerced > 0:
                return coerced
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        fallback = getattr(settings, "MICRO__STALE_MS", WATCHDOG_STALE_MS_DEFAULT)
        coerced = int(float(fallback))
        return coerced if coerced > 0 else WATCHDOG_STALE_MS_DEFAULT
    except Exception:  # pragma: no cover - defensive
        return WATCHDOG_STALE_MS_DEFAULT


def _min_age_ms(data_source: Any) -> Optional[int]:
    if data_source is None:
        return None
    getter = getattr(data_source, "min_age_across_subs_ms", None)
    if not callable(getter):
        return None
    try:
        age = getter()
    except Exception:  # pragma: no cover - defensive
        return None
    if age is None:
        return None
    try:
        coerced = int(float(age))
    except (TypeError, ValueError):
        return None
    return max(coerced, 0)


def evaluate_microstructure(
    quote: Mapping[str, Any] | None,
    *,
    required_lots: int,
    spread_cap_pct: float,
    side: str | None = None,
    data_source: Any | None = None,
    trace_id: str | None = None,
) -> Dict[str, Any]:
    """Return a structured microstructure decision for ``quote``.

    Parameters
    ----------
    quote:
        Quote payload containing ``bid``/``ask`` and optional depth fields.
    required_lots:
        Minimum lots desired for execution (after applying multipliers).
    spread_cap_pct:
        Maximum allowed spread ratio (0.0035 ⇒ 0.35%).
    side:
        Optional ``BUY``/``SELL`` hint used when selecting the relevant depth.
    data_source:
        Optional data source used to determine the freshest subscription age.
    trace_id:
        Optional trace identifier propagated through decision metadata.
    """

    lot_size = _lot_size()
    stale_limit = _watchdog_stale_ms()
    side_norm = str(side).upper() if side else None

    decision: Dict[str, Any] = {
        "ok": False,
        "reason": "no_quote",
        "bid": None,
        "ask": None,
        "mid": None,
        "spread_pct": None,
        "spread_cap_pct": float(spread_cap_pct),
        "age_ms": None,
        "depth_ok": None,
        "required_lots": max(int(required_lots), 0),
        "available_lots": None,
        "available_bid_qty": None,
        "available_ask_qty": None,
        "lot_size": lot_size,
        "side": side_norm,
        "trace_id": trace_id,
    }

    if not isinstance(quote, Mapping):
        return decision

    bid = _coerce_float(quote.get("bid"))
    ask = _coerce_float(quote.get("ask"))
    decision.update({"bid": bid, "ask": ask, "source": quote.get("source")})

    if bid is None or ask is None or bid <= 0 or ask <= 0:
        decision["reason"] = "no_quote"
        return decision

    has_age_source = data_source is not None and hasattr(
        data_source, "min_age_across_subs_ms"
    )
    age_ms = _min_age_ms(data_source) if has_age_source else None
    decision["age_ms"] = age_ms
    if has_age_source and age_ms is not None and age_ms > stale_limit:
        decision["reason"] = "stale_quote"
        return decision

    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else None
    decision["mid"] = mid
    if mid is None or mid <= 0:
        decision["reason"] = "no_quote"
        return decision

    spread = (ask - bid) / mid
    decision["spread_pct"] = spread
    if not (spread <= spread_cap_pct):
        decision["reason"] = "wide_spread"
        return decision

    bid_qty = _coerce_int(quote.get("bid_qty"))
    ask_qty = _coerce_int(quote.get("ask_qty"))
    if bid_qty is None or bid_qty <= 0:
        bid_qty = _sum_depth(quote.get("bid5_qty"))
    if ask_qty is None or ask_qty <= 0:
        ask_qty = _sum_depth(quote.get("ask5_qty"))

    depth_raw = quote.get("depth")
    depth_buy: Optional[int] = None
    depth_sell: Optional[int] = None
    if isinstance(depth_raw, Mapping):
        depth_buy = _sum_depth(depth_raw.get("buy"))
        depth_sell = _sum_depth(depth_raw.get("sell"))
    elif isinstance(depth_raw, Sequence) and not isinstance(
        depth_raw, (str, bytes, bytearray)
    ):
        depth_values = list(depth_raw)
        if depth_values:
            depth_buy = _sum_depth(depth_values[0])
        if len(depth_values) > 1:
            depth_sell = _sum_depth(depth_values[1])
    else:
        single_depth = _sum_depth(depth_raw)
        depth_buy = depth_buy if depth_buy is not None else single_depth
        depth_sell = depth_sell if depth_sell is not None else single_depth

    if bid_qty is None or bid_qty <= 0:
        if depth_buy is not None:
            bid_qty = depth_buy
    if ask_qty is None or ask_qty <= 0:
        if depth_sell is not None:
            ask_qty = depth_sell

    decision["available_bid_qty"] = bid_qty
    decision["available_ask_qty"] = ask_qty

    required_units = decision["required_lots"] * lot_size if lot_size > 0 else 0

    if required_units <= 0:
        decision["depth_ok"] = True
        decision["available_lots"] = None
        decision["ok"] = True
        decision["reason"] = None
        return decision

    if side_norm == "SELL":
        depth_units = bid_qty
    elif side_norm == "BUY":
        depth_units = ask_qty
    else:
        candidates = [val for val in (bid_qty, ask_qty) if isinstance(val, int)]
        depth_units = min(candidates) if candidates else None

    if depth_units is not None and lot_size > 0:
        decision["available_lots"] = depth_units / float(lot_size)

    if depth_units is None:
        decision["depth_ok"] = None
        decision["ok"] = True
        decision["reason"] = None
        return decision

    decision["depth_ok"] = depth_units >= required_units
    if decision["depth_ok"]:
        decision["ok"] = True
        decision["reason"] = None
    else:
        decision["ok"] = False
        decision["reason"] = "insufficient_depth"
    return decision


__all__ = ["evaluate_microstructure", "LOT_SIZE_DEFAULT", "WATCHDOG_STALE_MS_DEFAULT"]

