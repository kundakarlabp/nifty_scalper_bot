"""Canonical option-premium risk geometry.

This module is deliberately side-effect free. StrategyRunner calls these
functions directly, so live, paper, replay, and tests use identical geometry
without import-order hooks or class mutation.
"""

from __future__ import annotations

import dataclasses
from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.risk.net_rr_gate import (
    estimate_half_spread,
    minimum_risk_distance_for_net_rr,
)

_TICK_SIZE = 0.05


def _positive_float(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def _metadata(signal: Any) -> dict[str, Any]:
    value = getattr(signal, "metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _premium_domain(metadata: Mapping[str, Any]) -> bool:
    domain = str(
        metadata.get("invalidation_level_domain")
        or metadata.get("risk_domain")
        or metadata.get("atr_domain")
        or ""
    ).strip().lower()
    if domain in {"option_premium", "premium", "options"}:
        return True
    return bool(
        metadata.get("computed_from_premium")
        or metadata.get("premium_risk_contract_applied")
        or _positive_float(metadata.get("premium_stop_distance"))
        or _positive_float(metadata.get("premium_stop_pct"))
        or _positive_float(metadata.get("setup_invalidation_premium"))
        or _positive_float(metadata.get("premium_stop_price"))
    )


def _spread_distance(metadata: Mapping[str, Any], entry_price: float) -> float:
    bid = _positive_float(metadata.get("bid") or metadata.get("best_bid"))
    ask = _positive_float(metadata.get("ask") or metadata.get("best_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return ask - bid
    spread_pct = _positive_float(metadata.get("spread_pct"))
    if spread_pct is not None:
        return entry_price * spread_pct / 100.0
    return 0.0


def _side_geometry_valid(side: str, entry: float, stop: float, target: float) -> bool:
    if side == "BUY":
        return 0.0 < stop < entry < target
    return 0.0 < target < entry < stop


def _same_price(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and abs(left - right) <= _TICK_SIZE


def _risk_distance(
    signal: Any,
    *,
    entry_price: float,
    entry_side: str,
    atr: float,
) -> tuple[float, float, str, bool]:
    metadata = _metadata(signal)
    spread = _spread_distance(metadata, entry_price)
    trusted = _premium_domain(metadata)
    stop = _positive_float(getattr(signal, "stop_loss", None))
    target = _positive_float(getattr(signal, "take_profit", None))
    candidate_stop = _positive_float(metadata.get("candidate_stop_loss"))
    candidate_target = _positive_float(metadata.get("candidate_target"))
    candidate_symbol = str(metadata.get("candidate_symbol") or "").strip().upper()
    signal_symbol = str(getattr(signal, "symbol", "") or "").strip().upper()
    copied_candidate_geometry = bool(
        metadata.get("candidate_selected")
        and candidate_symbol
        and candidate_symbol == signal_symbol
        and _same_price(stop, candidate_stop)
        and _same_price(target, candidate_target)
    )
    explicit_stop = _positive_float(
        metadata.get("setup_invalidation_premium")
        or metadata.get("premium_stop_price")
    )
    explicit_distance = _positive_float(metadata.get("premium_stop_distance"))
    stop_pct = _positive_float(metadata.get("premium_stop_pct"))
    premium_atr = _positive_float(
        metadata.get("premium_atr") or metadata.get("option_atr")
    )
    if premium_atr is None and trusted:
        premium_atr = _positive_float(atr)

    source = "premium_percent_fallback"
    distance: float | None = None
    if copied_candidate_geometry and candidate_stop is not None:
        candidate = (
            entry_price - candidate_stop
            if entry_side == "BUY"
            else candidate_stop - entry_price
        )
        if candidate > 0.0:
            distance = candidate
            source = "selected_candidate_geometry"
            trusted = True
    if distance is None and explicit_stop is not None and trusted:
        candidate = (
            entry_price - explicit_stop
            if entry_side == "BUY"
            else explicit_stop - entry_price
        )
        if candidate > 0.0:
            distance = candidate
            source = "explicit_premium_stop"
    if distance is None and explicit_distance is not None:
        distance = explicit_distance
        source = "premium_stop_distance"
        trusted = True
    if distance is None and stop_pct is not None:
        normalized_pct = stop_pct / 100.0 if stop_pct > 1.0 else stop_pct
        if 0.0 < normalized_pct < 1.0:
            distance = entry_price * normalized_pct
            source = "premium_stop_pct"
            trusted = True
    if distance is None and stop is not None:
        candidate = entry_price - stop if entry_side == "BUY" else stop - entry_price
        untrusted_cap = max(entry_price * 0.30, spread * 4.0, 1.0)
        if candidate > 0.0 and (trusted or candidate <= untrusted_cap):
            distance = candidate
            source = "existing_premium_geometry"
    if distance is None and premium_atr is not None:
        distance = max(premium_atr * 1.2, entry_price * 0.02, spread * 1.5, 1.0)
        source = "premium_atr"
        trusted = True
    if distance is None:
        distance = max(entry_price * 0.10, spread * 1.5, 1.0)

    max_distance = max(
        entry_price * (0.60 if trusted else 0.30),
        spread * 4.0,
        1.0,
    )
    distance = min(max(distance, _TICK_SIZE), max_distance)
    if entry_side == "BUY":
        distance = min(distance, max(entry_price - _TICK_SIZE, _TICK_SIZE))
    rr = _positive_float(metadata.get("premium_target_rr")) or 2.0
    return distance, rr, source, trusted


def apply_premium_risk_contract(signal: Any, premium: float) -> Any:
    """Fill missing option-premium SL/TP from explicit strategy distance."""
    if signal is None or premium <= 0.0:
        return signal
    action = str(getattr(signal, "action", "") or "").upper()
    symbol = str(getattr(signal, "symbol", "") or "").upper()
    if action not in {"BUY", "SELL"} or not symbol.endswith(("CE", "PE")):
        return signal

    metadata = getattr(signal, "metadata", {})
    if not isinstance(metadata, Mapping):
        return signal
    distance = _positive_float(metadata.get("premium_stop_distance"))
    if distance is None:
        return signal
    domain = str(metadata.get("invalidation_level_domain") or "option_premium").lower()
    if domain != "option_premium":
        return signal

    existing_sl = _positive_float(getattr(signal, "stop_loss", None))
    existing_tp = _positive_float(getattr(signal, "take_profit", None))
    rr = _positive_float(metadata.get("premium_target_rr")) or 2.0

    if action == "BUY":
        stop_loss = existing_sl or max(0.05, premium - distance)
        take_profit = existing_tp or premium + distance * rr
        valid = stop_loss < premium < take_profit
    else:
        stop_loss = existing_sl or premium + distance
        take_profit = existing_tp or max(0.05, premium - distance * rr)
        valid = take_profit < premium < stop_loss
    if not valid:
        return signal

    updated_metadata = dict(metadata)
    updated_metadata.setdefault("premium_risk_contract_applied", True)
    updated_metadata.setdefault("premium_risk_source", "premium_stop_distance")
    updated_metadata.setdefault("premium_risk_reference_price", float(premium))
    return dataclasses.replace(
        signal,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        metadata=updated_metadata,
    )


def apply_cost_aware_risk_floor(
    signal: Any,
    *,
    entry_price: float,
    quantity: int,
    half_spread: float | None = None,
) -> Any:
    """Apply the minimum viable distance to distance-anchored long options."""

    entry = float(entry_price or 0.0)
    action = str(getattr(signal, "action", "") or "").upper()
    symbol = str(getattr(signal, "symbol", "") or "").upper()
    contract = symbol.split(":", 1)[-1]
    resolved_contract = bool(
        contract.endswith(("CE", "PE")) and any(char.isdigit() for char in contract[:-2])
    )
    metadata = _metadata(signal)
    if (
        entry <= 0.0
        or int(quantity or 0) <= 0
        or action != "BUY"
        or not resolved_contract
        or str(metadata.get("bracket_anchor_mode") or "distance").lower()
        != "distance"
    ):
        return signal

    stop = _positive_float(getattr(signal, "stop_loss", None))
    target = _positive_float(getattr(signal, "take_profit", None))
    if stop is None or target is None or not (0.0 < stop < entry < target):
        return signal
    current_distance = entry - stop
    rr = _positive_float(metadata.get("premium_target_rr"))
    if rr is None:
        rr = (target - entry) / current_distance
    spread = (
        max(0.0, float(half_spread))
        if half_spread is not None
        else estimate_half_spread(signal, entry)
    )
    floor = minimum_risk_distance_for_net_rr(
        entry_price=entry,
        gross_rr=rr,
        quantity=int(quantity),
        half_spread=spread,
        maximum_distance=entry * 0.60,
    )
    updated = dict(metadata)
    updated["premium_cost_floor_distance"] = floor
    updated["premium_cost_floor_quantity"] = int(quantity)
    updated["premium_cost_floor_half_spread"] = spread
    if floor is None:
        updated["premium_cost_floor_viable"] = False
        return dataclasses.replace(signal, metadata=updated)
    updated["premium_cost_floor_viable"] = True
    if current_distance + 1e-9 >= floor:
        updated["premium_cost_floor_applied"] = False
        return dataclasses.replace(signal, metadata=updated)

    updated["premium_cost_floor_applied"] = True
    updated["premium_cost_floor_original_distance"] = current_distance
    updated["premium_stop_distance"] = floor
    updated["premium_risk_distance"] = floor
    updated["premium_risk_source"] = "cost_aware_minimum"
    return dataclasses.replace(
        signal,
        stop_loss=max(_TICK_SIZE, entry - floor),
        take_profit=entry + floor * rr,
        metadata=updated,
    )


def validate_option_premium_geometry(
    self: Any,
    signal: Any,
    entry_price: float,
    entry_side: str,
    atr: float,
) -> Any:
    """Return domain-safe option SL/TP without using underlying-scale distances."""
    del self
    entry = float(entry_price or 0.0)
    side = str(entry_side or "BUY").upper()
    if entry <= 0.0 or side not in {"BUY", "SELL"}:
        return signal

    metadata = _metadata(signal)
    distance, rr, source, trusted = _risk_distance(
        signal,
        entry_price=entry,
        entry_side=side,
        atr=float(atr or 0.0),
    )
    spread = _spread_distance(metadata, entry)
    existing_stop = _positive_float(getattr(signal, "stop_loss", None))
    existing_target = _positive_float(getattr(signal, "take_profit", None))
    explicit_target = _positive_float(
        metadata.get("setup_target_premium")
        or metadata.get("premium_target_price")
    )
    has_rr_contract = _positive_float(metadata.get("premium_target_rr")) is not None

    if side == "BUY":
        stop = entry - distance
        target = entry + distance * rr
    else:
        stop = entry + distance
        target = max(_TICK_SIZE, entry - distance * rr)

    existing_stop_distance = None
    if existing_stop is not None:
        existing_stop_distance = (
            entry - existing_stop if side == "BUY" else existing_stop - entry
        )
    max_stop_distance = max(
        entry * (0.60 if trusted else 0.30), spread * 4.0, 1.0
    )
    stop_was_usable = bool(
        existing_stop_distance is not None
        and 0.0 < existing_stop_distance <= max_stop_distance
    )
    if stop_was_usable and source == "existing_premium_geometry":
        stop = float(existing_stop)
        distance = float(existing_stop_distance)
        if side == "BUY":
            target = entry + distance * rr
        else:
            target = max(_TICK_SIZE, entry - distance * rr)

    max_target_distance = max(entry, distance * max(rr, 3.0), spread * 8.0)
    if explicit_target is not None:
        explicit_valid = (
            entry < explicit_target <= entry + max_target_distance
            if side == "BUY"
            else max(_TICK_SIZE, entry - max_target_distance)
            <= explicit_target
            < entry
        )
        if explicit_valid:
            target = explicit_target
    elif not has_rr_contract and existing_target is not None and stop_was_usable:
        target_distance = (
            existing_target - entry if side == "BUY" else entry - existing_target
        )
        if 0.0 < target_distance <= max_target_distance:
            target = existing_target

    if not _side_geometry_valid(side, entry, stop, target):
        if side == "BUY":
            stop = max(_TICK_SIZE, entry - distance)
            target = entry + distance * rr
        else:
            stop = entry + distance
            target = max(_TICK_SIZE, entry - distance * rr)

    updated = dict(metadata)
    updated.update(
        {
            "premium_risk_distance": float(distance),
            "premium_risk_domain": "option_premium",
            "premium_risk_source": source,
            "premium_target_rr": float(rr),
            "premium_geometry_validated": True,
        }
    )
    if source == "premium_atr":
        updated.setdefault("premium_atr", float(distance) / 1.2)

    return dataclasses.replace(
        signal,
        stop_loss=max(_TICK_SIZE, float(stop)),
        take_profit=max(_TICK_SIZE, float(target)),
        metadata=updated,
    )


def anchor_option_geometry_to_execution(
    self: Any,
    signal: Any,
    *,
    signal_price: float,
    execution_price: float,
    entry_side: str,
    atr: float,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> Any:
    """Translate valid distance geometry to execution price; never widen it."""
    del sl_mult, tp_mult
    execution = float(execution_price or 0.0)
    reference = float(signal_price or 0.0)
    side = str(entry_side or "BUY").upper()
    if execution <= 0.0 or side not in {"BUY", "SELL"}:
        return signal

    metadata = _metadata(signal)
    mode = str(metadata.get("bracket_anchor_mode") or "distance").lower()
    stop = _positive_float(getattr(signal, "stop_loss", None))
    target = _positive_float(getattr(signal, "take_profit", None))

    # Absolute technical invalidations are not moved. If fill drift invalidates
    # them, the downstream order preflight remains the fail-closed authority.
    if mode == "absolute_level":
        return signal

    delta = execution - reference
    if abs(delta) > _TICK_SIZE:
        if stop is not None:
            stop += delta
        if target is not None:
            target += delta

    if stop is not None and target is not None and _side_geometry_valid(
        side, execution, stop, target
    ):
        return dataclasses.replace(
            signal,
            stop_loss=float(stop),
            take_profit=float(target),
        )

    candidate = dataclasses.replace(
        signal,
        stop_loss=float(stop) if stop is not None else None,
        take_profit=float(target) if target is not None else None,
    )
    return validate_option_premium_geometry(
        self,
        candidate,
        entry_price=execution,
        entry_side=side,
        atr=atr,
    )


__all__ = [
    "anchor_option_geometry_to_execution",
    "apply_cost_aware_risk_floor",
    "apply_premium_risk_contract",
    "validate_option_premium_geometry",
]
