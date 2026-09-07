"""Canonical strategy-vote policy used by StrategyManager.

This module contains pure policy decisions only. It deliberately does not
monkey-patch StrategyManager or any runtime class. StrategyManager remains the
single owner of orchestration; these helpers make trigger/context semantics
explicit and testable before the legacy runtime adapters are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SCORE_KEYS = ("raw_setup_score", "setup_score", "strategy_score")
_MIN_KEYS = ("setup_min", "setup_min_score", "trigger_min_score", "min_score")
_CONTEXT_ONLY_STRATEGIES = frozenset({"orderflow"})
_CLOSE_ACTIONS = frozenset({"CLOSE_LONG", "CLOSE_SHORT"})


@dataclass(frozen=True, slots=True)
class SetupGateDecision:
    passed: bool
    score: float | None = None
    minimum: float | None = None
    reason: str | None = None


def _float_from(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def vote_role(vote: Any) -> str:
    """Return the effective immutable role for a vote.

    OrderFlow is context-only by architecture, irrespective of malformed or
    legacy metadata. Other strategies retain their declared role.
    """
    strategy = str(getattr(vote, "strategy", "") or "").strip().lower()
    if strategy in _CONTEXT_ONLY_STRATEGIES:
        return "context"
    metadata = dict(getattr(vote, "metadata", {}) or {})
    return str(metadata.get("role") or "trigger").strip().lower()


def is_close_signal(signal: Any) -> bool:
    return str(getattr(signal, "action", "") or "").upper() in _CLOSE_ACTIONS


def setup_gate_decision(vote: Any) -> SetupGateDecision:
    """Evaluate the strategy's own setup contract without changing thresholds."""
    if vote_role(vote) == "context":
        return SetupGateDecision(True)

    metadata = dict(getattr(vote, "metadata", {}) or {})
    score = _float_from(metadata, _SCORE_KEYS)
    minimum = _float_from(metadata, _MIN_KEYS)
    explicit_pass = metadata.get("setup_pass")
    block_reason = str(metadata.get("trigger_block_reason") or "").strip() or None
    has_contract = score is not None or minimum is not None or explicit_pass is not None

    if not has_contract:
        return SetupGateDecision(True, score, minimum)
    if explicit_pass is False:
        return SetupGateDecision(False, score, minimum, block_reason or "setup_pass_false")
    if block_reason in {"weak_score", "setup_below_minimum", "setup_failed"}:
        return SetupGateDecision(False, score, minimum, block_reason)
    if score is not None and minimum is not None and score < minimum:
        return SetupGateDecision(False, score, minimum, "setup_below_minimum")
    return SetupGateDecision(True, score, minimum)


def partition_votes(signals: Sequence[tuple[Any, Any]]) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], list[dict[str, Any]]]:
    """Partition valid triggers/context and reject failed setup contracts.

    Close signals are never blocked by entry setup policy.
    """
    triggers: list[tuple[Any, Any]] = []
    context: list[tuple[Any, Any]] = []
    rejected: list[dict[str, Any]] = []
    for signal, vote in signals:
        role = vote_role(vote)
        if role == "context":
            context.append((signal, vote))
            continue
        if not is_close_signal(signal):
            decision = setup_gate_decision(vote)
            if not decision.passed:
                rejected.append({
                    "strategy": getattr(vote, "strategy", None),
                    "score": decision.score,
                    "minimum": decision.minimum,
                    "reason": decision.reason,
                })
                continue
        triggers.append((signal, vote))
    return triggers, context, rejected


def independent_same_side_confirmation(signals: Sequence[tuple[Any, Any]]) -> tuple[bool, list[str]]:
    """Return bounded confirmation from a distinct same-side trigger strategy."""
    trigger_votes = [
        vote
        for signal, vote in signals
        if vote_role(vote) != "context" and not is_close_signal(signal)
    ]
    if len(trigger_votes) < 2:
        return False, []
    try:
        best = max(trigger_votes, key=lambda vote: float(getattr(vote, "score", 0.0) or 0.0))
    except Exception:
        return False, []
    best_side = str(getattr(best, "side", "") or "").upper()
    best_strategy = str(getattr(best, "strategy", "") or "").strip().lower()
    if best_side not in {"CE", "PE"} or not best_strategy:
        return False, []
    confirming = sorted({
        str(getattr(vote, "strategy", "") or "").strip()
        for vote in trigger_votes
        if str(getattr(vote, "side", "") or "").upper() == best_side
        and str(getattr(vote, "strategy", "") or "").strip().lower() not in {"", best_strategy}
    })
    return bool(confirming), confirming


__all__ = [
    "SetupGateDecision",
    "independent_same_side_confirmation",
    "is_close_signal",
    "partition_votes",
    "setup_gate_decision",
    "vote_role",
]
