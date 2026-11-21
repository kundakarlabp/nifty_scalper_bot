"""Purpose: Evaluate hypothetical trade exits using causal executable quotes.

Responsibilities: Replay prespecified stop-retention alternatives and sustained
adverse order flow using same-tick, depth-verified full-lot bid observations;
compare an observed quote peak with the actual broker exit without claiming
hypothetical fills. Missing or incomplete evidence remains unmeasured.
Safety: Pure research; no live order authority, lookahead or assumed stop fills.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _number(value: Any) -> float | None:
    """Read only finite numeric evidence."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _bid_path(samples: Any) -> list[tuple[float, float]]:
    """Return only timestamped, full-lot bid proofs from the same market tick.

    A cached best bid or favorable option LTP is *not* sufficient evidence for
    liquidating the actual position quantity. Older unlabelled records abstain.
    """
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return []
    normalized: dict[float, float] = {}
    for observation in samples:
        if not isinstance(observation, Mapping):
            continue
        if (
            observation.get("source") != "bid"
            or observation.get("full_lot_bid_proven") is not True
        ):
            continue
        timestamp = _number(observation.get("ts"))
        bid = _number(observation.get("bid"))
        top_qty = _number(observation.get("top_bid_qty"))
        required = _number(observation.get("required_qty"))
        quote_age_ms = _number(observation.get("quote_age_ms"))
        market_age_ms = _number(observation.get("market_age_ms"))
        if (
            timestamp is None
            or timestamp <= 0
            or bid is None
            or bid <= 0
            or top_qty is None
            or required is None
            or required <= 0
            or top_qty < required
            or quote_age_ms is None
            or not 0 <= quote_age_ms <= 2000.0
            or market_age_ms is None
            or not 0 <= market_age_ms <= 2000.0
        ):
            continue
        normalized[timestamp] = bid
    return sorted(normalized.items())


def replay_stop_retention(
    samples: Any,
    *,
    entry: float,
    initial_stop: float,
    quantity: int,
    round_trip_cost: float,
    retention: float = 0.40,
    min_lock_after_1p5_r: float = 0.0,
    min_samples: int = 5,
) -> dict[str, Any]:
    """Replay one research stop policy without lookahead or invented fills.

    The stop is evaluated *before* the current tick can raise it. A stop
    breach fills conservatively at the observed bid, not the stop trigger.
    The tiered live stop algorithm has additional ATR/spacing behavior;
    this approximation is not a byte-for-byte replacement or promotion test.
    """
    path = _bid_path(samples)
    risk = entry - initial_stop
    if (
        risk <= 0
        or entry <= 0
        or quantity <= 0
        or round_trip_cost < 0
        or not 0 < retention <= 1
        or len(path) < min_samples
    ):
        return {
            "ready": False,
            "reason": "insufficient_executable_path_or_bad_geometry",
        }
    cost_points = round_trip_cost / quantity
    high = entry
    stop = initial_stop
    peak = 0.0
    for timestamp, bid in path:
        if bid <= stop:
            net_pnl = (bid - entry) * quantity - round_trip_cost
            return {
                "ready": True,
                "status": "simulated_stop_breach",
                "exit_ts": timestamp,
                "exit_bid": round(bid, 4),
                "stop_at_breach": round(stop, 4),
                "net_r": round(net_pnl / (risk * quantity), 4),
                "mfe_r": round(peak / risk, 4),
                "sample_count": len(path),
                "policy": "research_only",
            }
        high = max(high, bid)
        peak = max(0.0, high - entry)
        mfe_r = peak / risk
        if mfe_r < 0.60:
            continue
        if mfe_r >= 1.0:
            candidate = entry + peak * retention
        else:
            candidate = entry + cost_points
        candidate = max(candidate, entry + cost_points + 0.10 * risk)
        if mfe_r >= 1.5 and min_lock_after_1p5_r > 0:
            candidate = max(candidate, entry + min_lock_after_1p5_r * risk)
        # Do not assume that a stop at the current executable bid can be set.
        if candidate < bid - max(cost_points, 0.05):
            stop = max(stop, candidate)
    return {
        "ready": True,
        "status": "unresolved_at_last_sample",
        "last_ts": path[-1][0],
        "last_bid": round(path[-1][1], 4),
        "last_stop": round(stop, 4),
        "mfe_r": round(peak / risk, 4),
        "sample_count": len(path),
        "policy": "research_only",
    }


def compare_profit_retention(
    samples: Any,
    *,
    entry: float,
    initial_stop: float,
    quantity: int,
    round_trip_cost: float,
    samples_truncated: bool = False,
    entry_at: float | None = None,
    exit_at: float | None = None,
    realized_net_pnl: float | None = None,
    cost_source: str | None = None,
) -> dict[str, Any]:
    """Compare quote policies and measure only supported peak-to-fill giveback."""
    variants = {
        "retention_40pct": (0.40, 0.0),
        "retention_50pct": (0.50, 0.0),
        "retention_60pct": (0.60, 0.0),
        "retention_40pct_floor_0p75r_at_1p5r": (0.40, 0.75),
        "retention_40pct_floor_1r_at_1p5r": (0.40, 1.0),
    }
    if samples_truncated:
        return {
            "ready": False,
            "reason": "quote_path_truncated",
            "quote_peak_retention": None,
            "policies": {},
            "policy": "research_only",
        }
    path = _bid_path(samples)
    if not path:
        return {
            "ready": False,
            "reason": "full_lot_bid_path_unverified",
            "policy": "research_only",
            "policies": {},
            "quote_peak_retention": None,
        }
    # Record the best VERIFIED, time-bounded full-size bid. Historical best
    # prices cannot be retroactively filled; do not claim realized lost profit.
    peak_report: dict[str, Any] | None = None
    start = _number(entry_at)
    end = _number(exit_at)
    actual_net = _number(realized_net_pnl)
    if (
        start is not None
        and end is not None
        and 0 < start < end
        and actual_net is not None
        and quantity > 0
        and entry > 0
        and math.isfinite(round_trip_cost)
        and round_trip_cost >= 0
    ):
        held = [(ts, bid) for ts, bid in path if start <= ts <= end]
        if held:
            peak_ts, peak_bid = max(held, key=lambda item: item[1])
            peak_net_proxy = (peak_bid - entry) * quantity - round_trip_cost
            gap = peak_net_proxy - actual_net
            peak_report = {
                "ready": True,
                "provenance": "same_tick_full_lot_top_bid_observed_not_filled",
                "peak_quote_epoch": peak_ts,
                "peak_bid": round(peak_bid, 4),
                "peak_bid_mfe_r": (
                    round((peak_bid - entry) / (entry - initial_stop), 4)
                    if entry > initial_stop
                    else None
                ),
                "peak_net_using_actual_exit_costs_proxy": round(peak_net_proxy, 2),
                "actual_ledger_net_pnl": round(actual_net, 2),
                "recorded_cost_source": str(cost_source or "unverified"),
                "broker_costs_verified": cost_source == "broker_virtual_contract_note",
                "peak_to_realized_cash_gap_proxy": round(max(0.0, gap), 2),
                "positive_quoted_profit_given_back_proxy": round(
                    max(0.0, gap) if peak_net_proxy > 0 else 0.0, 2
                ),
                "observed_held_full_lot_samples": len(held),
                "fill_feasibility_not_proven": True,
                "future_movement_not_substituted": True,
            }
    policies = {
        key: replay_stop_retention(
            samples,
            entry=entry,
            initial_stop=initial_stop,
            quantity=quantity,
            round_trip_cost=round_trip_cost,
            retention=retention,
            min_lock_after_1p5_r=floor,
        )
        for key, (retention, floor) in variants.items()
    }
    return {
        "ready": all(result["ready"] for result in policies.values()),
        "policy": "research_only",
        "verified_full_lot_samples": len(path),
        "quote_peak_retention": peak_report,
        "limitations": [
            "sampled_top_of_book_not_guaranteed_broker_fill",
            "historical_stop_modify_latency_not_replayed",
            "counterfactual_execution_not_broker_verified",
            "constant_actual_exit_costs_used_for_peak_net_proxy",
            "not_full_live_trailing_parity",
        ],
        "policies": policies,
    }


def evaluate_adverse_flow_shadow(
    observations: Any,
    *,
    contract_side: str,
    threshold: float = 1.5,
    required_observations: int = 3,
    max_window_seconds: float = 5.0,
) -> dict[str, Any]:
    """Identify the first *observed* sustained contrary OFI window.

    Returns only advisory evidence, never a live stop or market exit. Requires
    fresh observations, monotonic timestamps and consecutive OFI signs.
    """
    if contract_side not in {"CE", "PE"}:
        return {"triggered": False, "reason": "unknown_side"}
    if not isinstance(observations, Sequence):
        return {"triggered": False, "reason": "no_orderflow_samples"}
    streak: list[tuple[float, float, float | None]] = []
    prior_ts = -1.0
    for obs in observations:
        if not isinstance(obs, Mapping):
            streak.clear()
            continue
        ts = _number(obs.get("ts"))
        ofi = _number(obs.get("ofi_3s_norm"))
        bid = _number(obs.get("exit_bid"))
        fresh = obs.get("fresh") is True
        if ts is None or ts <= prior_ts:
            streak.clear()
            continue
        prior_ts = ts
        if ofi is None:
            streak.clear()
            continue
        adverse = ofi <= -threshold if contract_side == "CE" else ofi >= threshold
        if not fresh or not adverse:
            streak.clear()
            continue
        streak.append((ts, ofi, bid))
        streak = streak[-required_observations:]
        if (
            len(streak) >= required_observations
            and ts - streak[0][0] <= max_window_seconds
        ):
            return {
                "triggered": True,
                "policy": "shadow_only",
                "first_confirmed_ts": ts,
                "ofi_3s_norm": ofi,
                "exit_bid": bid,
                "observations": len(streak),
            }
    return {"triggered": False, "reason": "sustained_fresh_adverse_flow_not_observed"}
