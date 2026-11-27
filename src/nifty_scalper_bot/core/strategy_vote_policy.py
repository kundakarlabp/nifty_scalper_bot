"""Purpose: Apply canonical trigger-versus-context strategy vote policy.

Responsibilities:
- vote_role and partition_votes distinguish trigger from context strategies.
- setup_gate_decision validates actionable structural setup evidence.
- independent_same_side_confirmation checks corroborating evidence.

Boundaries: Pure vote policy; no quote fetching, symbol selection or order entry.
Safety: Context-only votes cannot create triggers; range reversal requires a
separately attested structural and option-flow contract in LIVE pilot mode.
The stronger nested/conditional reversal payoff floor survives final repricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.config.strategy_taxonomy import (
    canonical_signal_family,
    is_context_only_strategy,
    normalize_strategy_name,
    range_vwap_live_enabled,
    range_vwap_local_trend_enabled,
)
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost

_CLOSE_ACTIONS = frozenset({"CLOSE_LONG", "CLOSE_SHORT"})


@dataclass(frozen=True, slots=True)
class SetupGateDecision:
    passed: bool
    reason: str | None = None


def _range_minimum_net_rr(metadata: Mapping[str, Any]) -> float:
    """Preserve the stronger strategy threshold across the broker-plan boundary."""
    structure = metadata.get("range_structure")
    stronger = (
        metadata.get("range_local_trend_pilot") is True
        or metadata.get("range_counter_broad_trend") is True
        or metadata.get("range_conditional_center_exception") is True
        or (
            isinstance(structure, Mapping)
            and structure.get("conditional_center_exception_used") is True
        )
    )
    floor = 1.75 if stronger else 1.5
    value = metadata.get("range_required_net_rr", floor)
    try:
        required = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    return max(floor, required) if isfinite(required) else float("inf")


def verified_range_vwap_reversal(
    metadata: Mapping[str, Any], *, strategy: str | None, side: str
) -> bool:
    """Authorize only the internally generated LIVE range reversal contract.

    This is not a broker-risk override: the runner validates execution price,
    affordability, stop, exposure and protective-order readiness independently.
    """
    if not range_vwap_live_enabled():
        return False
    if normalize_strategy_name(strategy) != "range_vwap_reclaim":
        return False
    if metadata.get("role") != "trigger" or metadata.get("setup_pass") is not True:
        return False
    if metadata.get("range_option_flow_verified") is not True:
        return False
    structure = metadata.get("range_structure")
    economics = metadata.get("option_cost_risk")
    if not isinstance(structure, Mapping) or not isinstance(economics, Mapping):
        return False
    if (
        structure.get("qualified") is not True
        or structure.get("side") != side
        or structure.get("reversal_direction_source") != "two_completed_futures_candles"
        or economics.get("qualified") is not True
        or economics.get("model_only") is not True
    ):
        return False
    if metadata.get("range_local_trend_pilot") is True and not (
        range_vwap_local_trend_enabled()
        and structure.get("local_range_confirmed") is True
        and structure.get("local_reclaim_side") == side
        and structure.get("broader_60bar_trend_side") in {"CE", "PE"}
        and metadata.get("range_spot_broad_trend_side")
        == structure.get("broader_60bar_trend_side")
    ):
        return False
    try:
        risk = float(economics["worst_stop_loss_including_costs"])
        rr = float(economics["net_reward_risk"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        isfinite(risk)
        and isfinite(rr)
        and 0 < risk <= 773.0
        and rr >= _range_minimum_net_rr(metadata)
    )


def range_legacy_context_mismatch_only(
    evidence: Any, *, expected_symbol: str, expected_quote_version: Any
) -> bool:
    """Allow a proven futures reversal to ignore *only* old trend mismatch.

    Actual adverse OrderFlow/depth, unavailable flow and wrong quote version
    must still veto admission in the strategy manager.
    """
    if normalize_strategy_name(getattr(evidence, "strategy", None)) != "order_flow":
        return False
    meta = dict(getattr(evidence, "metadata", {}) or {})
    if (
        meta.get("context_alignment_block_reason") != "underlying_direction_mismatch"
        or meta.get("candidate_symbol") != expected_symbol
        or expected_quote_version in (None, "")
        or str(meta.get("quote_update_version")) != str(expected_quote_version)
        or meta.get("quote_readiness_allowed") is not True
        or meta.get("ofi_ready") is not True
        or meta.get("ofi_3s_complete") is not True
        or meta.get("range_research_holdoff_active") is not False
        or meta.get("microstructure_conflicts_side") is True
        or meta.get("strong_depth_conflicts_side") is True
        or meta.get("ofi_conflicts_side") is True
    ):
        return False
    try:
        span = float(meta["ofi_3s_span_seconds"])
        updates = int(meta["ofi_update_count_3s"])
        ofi_3s = float(meta["ofi_3s_normalized"])
        ofi_1s = float(meta["ofi_1s_normalized"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        isfinite(span)
        and isfinite(ofi_3s)
        and isfinite(ofi_1s)
        and span >= 1.5
        and updates >= 3
        and ofi_3s >= 0.10
        and ofi_1s >= 0
    )


def verified_range_vwap_final_order_risk(
    metadata: Mapping[str, Any],
    *,
    strategy: str,
    symbol: str,
    final_symbol: str,
    broker_quantity: int,
    broker_lot_size: int,
    entry_price: float,
    ask_price: float,
    stop_price: float,
    target_price: float,
) -> bool:
    """Recheck final repriced broker plan, not merely the strategy estimate."""
    if not verified_range_vwap_reversal(
        metadata,
        strategy=strategy,
        side="CE" if symbol.endswith("CE") else "PE" if symbol.endswith("PE") else "",
    ):
        return False
    model = metadata.get("option_cost_risk")
    if not isinstance(model, Mapping):
        return False
    try:
        modeled_lot_size = int(model["observed_contract_lot_size"])
        slippage = float(model["slippage_each_side_bps"]) / 10000.0
        expected_stop = float(model["stop_price_model"])
        modeled_stop_fill = float(model["modeled_stressed_stop_fill_price"])
        exit_spread_stress = float(model["modeled_extra_exit_spread_points"])
        expected_target = float(model["vwap_limited_target_price_model"])
        values = (
            entry_price,
            ask_price,
            stop_price,
            target_price,
            expected_stop,
            modeled_stop_fill,
            exit_spread_stress,
            expected_target,
        )
        prices_valid = all(isfinite(float(value)) for value in values)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        not prices_valid
        or symbol != final_symbol
        or broker_lot_size != modeled_lot_size
        or broker_quantity != modeled_lot_size
        or modeled_lot_size <= 0
        or not 0 <= slippage <= 0.01
        or not 0 < modeled_stop_fill < expected_stop
        or not 0 <= exit_spread_stress < expected_stop
        or not 0 < stop_price < max(entry_price, ask_price) < target_price
        # Never let generic premium geometry silently replace the reachable
        # session-VWAP stop/target with a synthetic wider target.
        or abs(stop_price - expected_stop) > 0.051
        or abs(target_price - expected_target) > 0.051
        or metadata.get("bracket_anchor_mode") != "absolute_level"
    ):
        return False
    stressed_entry = max(entry_price, ask_price) * (1.0 + slippage)
    # The trigger is a bid threshold, not a guaranteed liquidation fill.
    # Preserve the strategy's more adverse scenario when the entry quote drifts.
    stressed_stop = min(
        modeled_stop_fill,
        stop_price * (1.0 - slippage) - exit_spread_stress,
    )
    if stressed_stop <= 0:
        return False
    fees = estimate_round_trip_cost(
        entry_price=stressed_entry,
        exit_price=stressed_stop,
        quantity=broker_quantity,
    ).total
    final_cash_risk = (stressed_entry - stressed_stop) * broker_quantity + fees
    target_fees = estimate_round_trip_cost(
        entry_price=stressed_entry,
        exit_price=target_price,
        quantity=broker_quantity,
    ).total
    net_target_reward = (target_price - stressed_entry) * broker_quantity - target_fees
    return bool(
        isfinite(final_cash_risk)
        and isfinite(net_target_reward)
        and 0 < final_cash_risk <= 773.0
        and net_target_reward / final_cash_risk >= _range_minimum_net_rr(metadata)
    )


def is_permanent_context_only(evidence: Any) -> bool:
    """Return whether a strategy is structurally context-only."""
    return is_context_only_strategy(getattr(evidence, "strategy", None))


def vote_role(evidence: Any) -> str:
    """Return the immutable role for one strategy evidence record."""
    if is_permanent_context_only(evidence):
        return "context"
    metadata = dict(getattr(evidence, "metadata", {}) or {})
    return str(metadata.get("role") or "trigger").strip().lower()


def is_close_signal(signal: Any) -> bool:
    return str(getattr(signal, "action", "") or "").upper() in _CLOSE_ACTIONS


def setup_gate_decision(evidence: Any) -> SetupGateDecision:
    """Evaluate a strategy's own structural setup contract."""
    if vote_role(evidence) == "context":
        return SetupGateDecision(True)

    metadata: Mapping[str, Any] = dict(getattr(evidence, "metadata", {}) or {})
    if metadata.get("side_conflict") is True:
        return SetupGateDecision(False, "strategy_contract_side_conflict")
    if metadata.get("required_data_present") is False:
        return SetupGateDecision(False, "required_data_missing")
    if metadata.get("stale_data_used") is True:
        return SetupGateDecision(False, "stale_data")
    if metadata.get("setup_pass") is not True:
        return SetupGateDecision(
            False,
            str(metadata.get("trigger_block_reason") or "setup_contract_not_passed"),
        )
    if metadata.get("trigger_conditions_met") is False:
        return SetupGateDecision(
            False,
            str(metadata.get("trigger_block_reason") or "trigger_conditions_not_met"),
        )
    return SetupGateDecision(True)


def partition_votes(
    signals: Sequence[tuple[Any, Any]],
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], list[dict[str, Any]]]:
    """Partition setup-valid triggers, context evidence and rejected setups."""
    triggers: list[tuple[Any, Any]] = []
    context: list[tuple[Any, Any]] = []
    rejected: list[dict[str, Any]] = []
    for signal, evidence in signals:
        role = vote_role(evidence)
        if role == "context":
            context.append((signal, evidence))
            continue
        if not is_close_signal(signal):
            decision = setup_gate_decision(evidence)
            if not decision.passed:
                rejected.append(
                    {
                        "strategy": getattr(evidence, "strategy", None),
                        "reason": decision.reason,
                    }
                )
                continue
        triggers.append((signal, evidence))
    return triggers, context, rejected


def independent_same_side_confirmation(
    signals: Sequence[tuple[Any, Any]],
) -> tuple[bool, list[str]]:
    """Return confirmation from a distinct setup-valid trigger evidence family."""
    valid = []
    for signal, evidence in signals:
        if vote_role(evidence) == "context" or is_close_signal(signal):
            continue
        if not setup_gate_decision(evidence).passed:
            continue
        valid.append(evidence)

    if len(valid) < 2:
        return False, []

    primary = valid[0]
    side = str(getattr(primary, "side", "") or "").upper()
    strategy = str(getattr(primary, "strategy", "") or "").strip().lower()
    if side not in {"CE", "PE"} or not strategy:
        return False, []
    family = canonical_signal_family(strategy)

    confirming = sorted(
        {
            str(getattr(item, "strategy", "") or "").strip()
            for item in valid[1:]
            if str(getattr(item, "side", "") or "").upper() == side
            and str(getattr(item, "strategy", "") or "").strip().lower()
            not in {"", strategy}
            and canonical_signal_family(getattr(item, "strategy", None)) != family
        }
    )
    return bool(confirming), confirming


__all__ = [
    "SetupGateDecision",
    "independent_same_side_confirmation",
    "is_close_signal",
    "range_legacy_context_mismatch_only",
    "is_permanent_context_only",
    "partition_votes",
    "setup_gate_decision",
    "vote_role",
    "verified_range_vwap_reversal",
    "verified_range_vwap_final_order_risk",
]
