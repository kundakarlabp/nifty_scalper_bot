"""Final transaction-cost-aware reward-to-risk gate for option entries."""

from __future__ import annotations

import math
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Mapping

from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost


@dataclass(slots=True, frozen=True)
class NetRRResult:
    allowed: bool
    net_rr: float
    minimum: float
    gross_reward: float
    gross_risk: float
    net_reward: float
    net_risk: float
    target_cost: float
    stop_cost: float
    half_spread: float


def _positive(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def _quantity(signal: Any) -> int:
    for name in ("quantity", "qty", "order_quantity"):
        with suppress(TypeError, ValueError):
            value = int(float(getattr(signal, name, 0) or 0))
            if value > 0:
                return value
    return 0


def _metadata(signal: Any) -> Mapping[str, Any]:
    value = getattr(signal, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _price(signal: Any, *names: str) -> float | None:
    metadata = _metadata(signal)
    for name in names:
        value = getattr(signal, name, None)
        if value is None:
            value = metadata.get(name)
        parsed = _positive(value)
        if parsed is not None:
            return parsed
    return None


def _half_spread(signal: Any, entry: float) -> float:
    metadata = _metadata(signal)
    bid = _positive(metadata.get("bid") or metadata.get("best_bid"))
    ask = _positive(metadata.get("ask") or metadata.get("best_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return (ask - bid) / 2.0
    absolute = _positive(
        metadata.get("half_spread") or metadata.get("half_spread_points")
    )
    if absolute is not None:
        return absolute
    spread_pct = _positive(metadata.get("spread_pct"))
    if spread_pct is not None:
        return entry * spread_pct / 200.0
    with suppress(TypeError, ValueError):
        fallback_pct = max(
            0.0,
            float(os.getenv("COST_FALLBACK_HALF_SPREAD_PCT", "0.25") or 0.25),
        )
        return entry * fallback_pct / 100.0
    return entry * 0.0025


def _minimum_net_rr() -> float:
    """Return a finite non-negative threshold; malformed config never bypasses."""
    raw = os.getenv("MIN_NET_REWARD_RISK", "1.5")
    try:
        parsed = float(raw or 1.5)
    except (TypeError, ValueError):
        return 1.5
    if not math.isfinite(parsed):
        return 1.5
    return max(0.0, parsed)


def _max_target_uplift_r() -> float:
    """Return the maximum extra gross-R permitted to repair transaction costs."""
    raw = os.getenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    try:
        parsed = float(raw or 0.35)
    except (TypeError, ValueError):
        return 0.35
    if not math.isfinite(parsed):
        return 0.35
    return max(0.0, parsed)


def evaluate_final_net_rr(signal: Any) -> NetRRResult | None:
    """Return final BUY-option net RR, or None when the gate is not applicable."""
    symbol = str(getattr(signal, "symbol", "") or "").strip().upper()
    side = str(
        getattr(signal, "action", "")
        or getattr(signal, "transaction_type", "")
        or getattr(signal, "side", "")
    ).strip().upper()
    if not symbol.endswith(("CE", "PE")) or side != "BUY":
        return None

    entry = _price(
        signal,
        "entry_price",
        "price",
        "limit_price",
        "reference_price",
        "premium_risk_reference_price",
        "current_price",
    )
    stop = _price(signal, "stop_loss", "sl", "stop_price")
    target = _price(signal, "take_profit", "target", "target_price")
    quantity = _quantity(signal)
    if entry is None or stop is None or target is None or quantity <= 0:
        return None
    if not (stop < entry < target):
        return None

    half_spread = _half_spread(signal, entry)
    target_cost = estimate_round_trip_cost(
        entry_price=entry,
        exit_price=target,
        quantity=quantity,
        half_spread=half_spread,
    ).total
    stop_cost = estimate_round_trip_cost(
        entry_price=entry,
        exit_price=stop,
        quantity=quantity,
        half_spread=half_spread,
    ).total
    gross_reward = (target - entry) * quantity
    gross_risk = (entry - stop) * quantity
    net_reward = gross_reward - target_cost
    net_risk = gross_risk + stop_cost
    net_rr = net_reward / net_risk if net_reward > 0.0 and net_risk > 0.0 else 0.0
    minimum = _minimum_net_rr()
    return NetRRResult(
        allowed=net_rr >= minimum,
        net_rr=net_rr,
        minimum=minimum,
        gross_reward=gross_reward,
        gross_risk=gross_risk,
        net_reward=net_reward,
        net_risk=net_risk,
        target_cost=target_cost,
        stop_cost=stop_cost,
        half_spread=half_spread,
    )


def minimum_target_for_net_rr(signal: Any, *, tick_size: float = 0.05) -> float | None:
    """Return the smallest bounded BUY-option target satisfying final net RR.

    The final gate remains authoritative. This helper only compensates a modest
    transaction-cost erosion of an already valid distance-based strategy target.
    If the configured net RR cannot be reached within ``MAX_NET_RR_TARGET_UPLIFT_R``
    additional gross R, ``None`` is returned and the caller must fail closed.
    """
    current = evaluate_final_net_rr(signal)
    if current is None:
        return None
    target = _price(signal, "take_profit", "target", "target_price")
    entry = _price(
        signal,
        "entry_price",
        "price",
        "limit_price",
        "reference_price",
        "premium_risk_reference_price",
        "current_price",
    )
    stop = _price(signal, "stop_loss", "sl", "stop_price")
    quantity = _quantity(signal)
    if target is None or entry is None or stop is None or quantity <= 0:
        return None
    if current.allowed:
        return float(target)

    risk_points = entry - stop
    if risk_points <= 0.0 or target <= entry:
        return None
    current_gross_rr = (target - entry) / risk_points
    cap_rr = current_gross_rr + _max_target_uplift_r()
    cap_target = entry + risk_points * cap_rr
    tick = float(tick_size) if math.isfinite(float(tick_size)) and tick_size > 0 else 0.05
    max_tick_target = math.floor((cap_target + 1e-12) / tick) * tick
    if max_tick_target <= target:
        return None

    half_spread = _half_spread(signal, entry)
    stop_cost = estimate_round_trip_cost(
        entry_price=entry,
        exit_price=stop,
        quantity=quantity,
        half_spread=half_spread,
    ).total
    net_risk = (entry - stop) * quantity + stop_cost
    minimum = _minimum_net_rr()
    if net_risk <= 0.0:
        return None

    def _net_rr_at(candidate: float) -> float:
        target_cost = estimate_round_trip_cost(
            entry_price=entry,
            exit_price=candidate,
            quantity=quantity,
            half_spread=half_spread,
        ).total
        net_reward = (candidate - entry) * quantity - target_cost
        return net_reward / net_risk if net_reward > 0.0 else 0.0

    if _net_rr_at(max_tick_target) + 1e-12 < minimum:
        return None

    low = float(target)
    high = float(max_tick_target)
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if _net_rr_at(midpoint) >= minimum:
            high = midpoint
        else:
            low = midpoint

    candidate = math.ceil((high - 1e-12) / tick) * tick
    candidate = round(candidate, 2)
    if candidate > max_tick_target + 1e-9:
        return None
    if _net_rr_at(candidate) + 1e-12 < minimum:
        return None
    return candidate


__all__ = ["NetRRResult", "evaluate_final_net_rr", "minimum_target_for_net_rr"]
