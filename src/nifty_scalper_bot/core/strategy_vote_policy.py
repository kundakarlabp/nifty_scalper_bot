"""Canonical strategy-evidence policy used by StrategyManager.

There is deliberately no score, confidence threshold, weighted vote, bonus, or
numeric admission model here. Strategies either prove their structural setup or
produce no trigger. Context strategies can corroborate a valid trigger but can
never manufacture one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.config.strategy_taxonomy import (
    canonical_signal_family,
    is_context_only_strategy,
)

_CLOSE_ACTIONS = frozenset({"CLOSE_LONG", "CLOSE_SHORT"})


@dataclass(frozen=True, slots=True)
class SetupGateDecision:
    passed: bool
    reason: str | None = None


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
    "is_permanent_context_only",
    "partition_votes",
    "setup_gate_decision",
    "vote_role",
]
