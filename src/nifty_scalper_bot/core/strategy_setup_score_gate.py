"""Compatibility helpers for the canonical strategy-vote policy.

Runtime ownership lives directly in :mod:`strategy_manager`; this module contains
no import-time installers or class mutation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from nifty_scalper_bot.core.strategy_vote_policy import (
    is_permanent_context_only,
    setup_gate_decision,
    vote_role,
)


def setup_gate_result(vote: Any) -> tuple[bool, float | None, float | None, str | None]:
    decision = setup_gate_decision(vote)
    return decision.passed, decision.score, decision.minimum, decision.reason


def enforce_context_only_role(signal: Any, vote: Any) -> tuple[Any, bool]:
    """Normalize malformed permanent context-only metadata without mutation hooks."""
    if not is_permanent_context_only(vote) or getattr(signal, "action", None) in {
        "CLOSE_LONG",
        "CLOSE_SHORT",
    }:
        return signal, False
    enforced = {
        "role": "context",
        "can_trigger": False,
        "trigger_conditions_met": False,
        "trigger_eligible": False,
        "trigger_block_reason": "context_only_role",
        "trigger_disqualified_by": "context_only_role",
        "context_role": "confirmation",
    }
    vote_metadata = dict(getattr(vote, "metadata", {}) or {})
    signal_metadata = dict(getattr(signal, "metadata", {}) or {})
    changed = any(
        vote_metadata.get(key) != value or signal_metadata.get(key) != value
        for key, value in enforced.items()
    )
    vote_metadata.update(enforced)
    signal_metadata.update(enforced)
    vote.metadata = vote_metadata
    with_metadata = getattr(signal, "with_metadata", None)
    if callable(with_metadata):
        signal = with_metadata(**enforced)
    elif dataclasses.is_dataclass(signal):
        signal = dataclasses.replace(signal, metadata=signal_metadata)
    else:
        signal.metadata = signal_metadata
    return signal, changed


def filter_context_promotions(
    context_votes: list[tuple[Any, Any]],
) -> tuple[list[tuple[Any, Any]], list[str]]:
    """Return promotable context votes plus permanent-context exclusions."""
    eligible: list[tuple[Any, Any]] = []
    blocked: list[str] = []
    for signal, vote in context_votes:
        if is_permanent_context_only(vote):
            blocked.append(str(getattr(vote, "strategy", "") or "unknown"))
        else:
            eligible.append((signal, vote))
    return eligible, blocked


__all__ = [
    "enforce_context_only_role",
    "filter_context_promotions",
    "setup_gate_result",
    "vote_role",
]
