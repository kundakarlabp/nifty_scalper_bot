"""Compatibility wiring for the canonical strategy-vote policy.

All setup/role policy lives in :mod:`strategy_vote_policy`. This module is only
the import-time integration seam for the legacy StrategyManager and therefore
must not define independent thresholds, roles, or confirmation semantics.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Mapping

from nifty_scalper_bot.core.strategy_vote_policy import (
    partition_votes,
    setup_gate_decision,
    vote_role,
)
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


def setup_gate_result(vote: Any) -> tuple[bool, float | None, float | None, str | None]:
    """Compatibility tuple view of the canonical setup-gate decision."""
    decision = setup_gate_decision(vote)
    return decision.passed, decision.score, decision.minimum, decision.reason


def enforce_context_only_role(signal: Any, vote: Any) -> tuple[Any, bool]:
    """Normalize permanent context-only metadata at the runtime seam."""
    strategy = str(getattr(vote, "strategy", "") or "").strip().lower()
    if (
        vote_role(vote) != "context"
        or strategy != "orderflow"
        or getattr(signal, "action", None) in {"CLOSE_LONG", "CLOSE_SHORT"}
    ):
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
    """Exclude permanent context-only votes from legacy promotion machinery."""
    eligible: list[tuple[Any, Any]] = []
    blocked: list[str] = []
    for signal, vote in context_votes:
        strategy = str(getattr(vote, "strategy", "") or "").strip()
        if vote_role(vote) == "context" and strategy.lower() == "orderflow":
            blocked.append(strategy or "unknown")
        else:
            eligible.append((signal, vote))
    return eligible, blocked


def apply_patches() -> None:
    """Wire canonical policy into the legacy StrategyManager seam idempotently."""
    from nifty_scalper_bot.core import strategy_manager as strategy_module

    cls = strategy_module.StrategyManager

    if not getattr(cls, "_hard_setup_score_gate_installed", False):
        original_combine = cls._combine_strategy_votes

        def _combine_strategy_votes(
            self: Any,
            *,
            symbol: str,
            signals: list[tuple[Any, Any]],
            indicators: Mapping[str, Any],
            no_vote_reason_counts: Mapping[str, int] | None = None,
        ) -> Any:
            normalized: list[tuple[Any, Any]] = []
            corrected: list[str] = []
            for signal, vote in signals:
                signal, role_changed = enforce_context_only_role(signal, vote)
                if role_changed:
                    corrected.append(str(getattr(vote, "strategy", "unknown")))
                normalized.append((signal, vote))

            triggers, contexts, rejected = partition_votes(normalized)
            eligible = triggers + contexts

            if corrected:
                log_throttled(
                    LOGGER,
                    f"context_role_enforced:{str(symbol).upper()}",
                    "PERMANENT_CONTEXT_ONLY_ROLE_ENFORCED symbol=%s strategies=%s",
                    symbol,
                    corrected,
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "PERMANENT_CONTEXT_ONLY_ROLE_ENFORCED",
                        "symbol": str(symbol).upper(),
                        "strategies": corrected,
                    },
                )

            entry_trigger_exists = any(
                getattr(signal, "action", None) not in {"CLOSE_LONG", "CLOSE_SHORT"}
                for signal, _vote in triggers
            )
            if rejected and not entry_trigger_exists:
                log_throttled(
                    LOGGER,
                    f"hard_setup_score_gate:{str(symbol).upper()}",
                    "HARD_SETUP_SCORE_GATE_BLOCKED symbol=%s rejected=%s",
                    symbol,
                    rejected,
                    interval_sec=30.0,
                    level=logging.INFO,
                    extra={
                        "event": "HARD_SETUP_SCORE_GATE_BLOCKED",
                        "symbol": str(symbol).upper(),
                        "rejected": rejected,
                    },
                )
                return None

            return original_combine(
                self,
                symbol=symbol,
                signals=eligible,
                indicators=indicators,
                no_vote_reason_counts=no_vote_reason_counts,
            )

        cls._hard_setup_score_gate_original_combine = original_combine
        cls._combine_strategy_votes = _combine_strategy_votes
        cls._hard_setup_score_gate_installed = True

    if not getattr(cls, "_permanent_context_only_gate_installed", False):
        original_promotion = cls._try_context_promotion

        def _try_context_promotion(
            self: Any,
            symbol: str,
            context_votes: list[tuple[Any, Any]],
            indicators: Mapping[str, Any],
            mode_profile: dict[str, Any],
        ) -> Any:
            eligible, blocked = filter_context_promotions(context_votes)
            if blocked:
                log_throttled(
                    LOGGER,
                    f"context_promotion_blocked:{str(symbol).upper()}",
                    "PERMANENT_CONTEXT_ONLY_PROMOTION_BLOCKED symbol=%s strategies=%s",
                    symbol,
                    blocked,
                    interval_sec=30.0,
                    level=logging.INFO,
                    extra={
                        "event": "PERMANENT_CONTEXT_ONLY_PROMOTION_BLOCKED",
                        "symbol": str(symbol).upper(),
                        "strategies": blocked,
                    },
                )
            if not eligible:
                return None
            return original_promotion(self, symbol, eligible, indicators, mode_profile)

        cls._permanent_context_only_original_promotion = original_promotion
        cls._try_context_promotion = _try_context_promotion
        cls._permanent_context_only_gate_installed = True


__all__ = [
    "apply_patches",
    "enforce_context_only_role",
    "filter_context_promotions",
    "setup_gate_result",
]
