"""Fail-closed setup-quality and context-role gates for strategy combination.

Context may confirm or veto a valid trigger, but it must never convert a setup
that failed its own strategy threshold into an executable entry. OrderFlow is a
permanent confirmation-only strategy and cannot be promoted into a trigger.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)

_SCORE_KEYS = ("raw_setup_score", "setup_score", "strategy_score")
_MIN_KEYS = ("setup_min", "setup_min_score", "trigger_min_score", "min_score")
_PERMANENT_CONTEXT_ONLY_STRATEGIES = frozenset({"orderflow"})


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


def setup_gate_result(vote: Any) -> tuple[bool, float | None, float | None, str | None]:
    """Return whether an explicit trigger setup contract passed.

    Votes without setup score/minimum metadata retain legacy behaviour.
    """
    metadata = dict(getattr(vote, "metadata", {}) or {})
    if str(metadata.get("role") or "trigger").strip().lower() == "context":
        return True, None, None, None

    score = _float_from(metadata, _SCORE_KEYS)
    minimum = _float_from(metadata, _MIN_KEYS)
    explicit_pass = metadata.get("setup_pass")
    block_reason = str(metadata.get("trigger_block_reason") or "").strip() or None
    has_contract = score is not None or minimum is not None or explicit_pass is not None

    if not has_contract:
        return True, score, minimum, None
    if explicit_pass is False:
        return False, score, minimum, block_reason or "setup_pass_false"
    if block_reason in {"weak_score", "setup_below_minimum", "setup_failed"}:
        return False, score, minimum, block_reason
    if score is not None and minimum is not None and score < minimum:
        return False, score, minimum, "setup_below_minimum"
    return True, score, minimum, None


def _enforce_permanent_context_only_role(signal: Any, vote: Any) -> bool:
    """Normalize permanent context strategies before the combiner sees them."""
    strategy = str(getattr(vote, "strategy", "") or "").strip()
    if strategy.lower() not in _PERMANENT_CONTEXT_ONLY_STRATEGIES:
        return False
    if getattr(signal, "action", None) in {"CLOSE_LONG", "CLOSE_SHORT"}:
        return False

    metadata = dict(getattr(vote, "metadata", {}) or {})
    signal_metadata = dict(getattr(signal, "metadata", {}) or {})
    changed = bool(
        str(metadata.get("role") or "trigger").strip().lower() != "context"
        or metadata.get("can_trigger") is not False
        or metadata.get("trigger_conditions_met") is not False
    )
    enforced = {
        "role": "context",
        "can_trigger": False,
        "trigger_conditions_met": False,
        "trigger_eligible": False,
        "trigger_block_reason": "context_only_role",
        "trigger_disqualified_by": "context_only_role",
        "context_role": "confirmation",
    }
    metadata.update(enforced)
    signal_metadata.update(enforced)
    setattr(vote, "metadata", metadata)
    setattr(signal, "metadata", signal_metadata)
    return changed


def _remove_permanent_context_only_promotions(
    context_votes: list[tuple[Any, Any]],
) -> tuple[list[tuple[Any, Any]], list[str]]:
    """Remove strategies that are never allowed to originate an entry."""
    eligible: list[tuple[Any, Any]] = []
    blocked: list[str] = []
    for signal, vote in context_votes:
        strategy = str(getattr(vote, "strategy", "") or "").strip()
        if strategy.lower() in _PERMANENT_CONTEXT_ONLY_STRATEGIES:
            blocked.append(strategy or "unknown")
            continue
        eligible.append((signal, vote))
    return eligible, blocked


def apply_patches() -> None:
    from nifty_scalper_bot.core import strategy_manager as strategy_module

    cls = strategy_module.StrategyManager

    if not getattr(cls, "_hard_setup_score_gate_installed", False):
        original = cls._combine_strategy_votes

        def _combine_strategy_votes(
            self: Any,
            *,
            symbol: str,
            signals: list[tuple[Any, Any]],
            indicators: Mapping[str, Any],
            no_vote_reason_counts: Mapping[str, int] | None = None,
        ) -> Any:
            eligible: list[tuple[Any, Any]] = []
            rejected: list[dict[str, Any]] = []
            role_corrections: list[str] = []

            for signal, vote in signals:
                if _enforce_permanent_context_only_role(signal, vote):
                    role_corrections.append(
                        str(getattr(vote, "strategy", "unknown") or "unknown")
                    )
                metadata = dict(getattr(vote, "metadata", {}) or {})
                role = str(metadata.get("role") or "trigger").strip().lower()
                is_close = getattr(signal, "action", None) in {
                    "CLOSE_LONG",
                    "CLOSE_SHORT",
                }
                if role != "context" and not is_close:
                    passed, score, minimum, reason = setup_gate_result(vote)
                    if not passed:
                        rejected.append(
                            {
                                "strategy": getattr(vote, "strategy", None),
                                "score": score,
                                "minimum": minimum,
                                "reason": reason,
                            }
                        )
                        continue
                eligible.append((signal, vote))

            if role_corrections:
                log_throttled(
                    LOGGER,
                    f"permanent_context_role:{str(symbol).upper()}",
                    "PERMANENT_CONTEXT_ONLY_ROLE_ENFORCED symbol=%s strategies=%s",
                    symbol,
                    role_corrections,
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "PERMANENT_CONTEXT_ONLY_ROLE_ENFORCED",
                        "symbol": str(symbol).upper(),
                        "strategies": role_corrections,
                    },
                )

            eligible_trigger_exists = any(
                str((getattr(vote, "metadata", {}) or {}).get("role") or "trigger")
                .strip()
                .lower()
                != "context"
                and getattr(signal, "action", None)
                not in {"CLOSE_LONG", "CLOSE_SHORT"}
                for signal, vote in eligible
            )
            if rejected and not eligible_trigger_exists:
                # Never pass context-only votes to the legacy promotion path after an
                # explicit trigger failed its own setup threshold.
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

            return original(
                self,
                symbol=symbol,
                signals=eligible,
                indicators=indicators,
                no_vote_reason_counts=no_vote_reason_counts,
            )

        cls._hard_setup_score_gate_original_combine = original
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
            eligible, blocked = _remove_permanent_context_only_promotions(
                context_votes
            )
            if blocked:
                log_throttled(
                    LOGGER,
                    f"permanent_context_only:{str(symbol).upper()}",
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
            return original_promotion(
                self,
                symbol,
                eligible,
                indicators,
                mode_profile,
            )

        cls._permanent_context_only_original_promotion = original_promotion
        cls._try_context_promotion = _try_context_promotion
        cls._permanent_context_only_gate_installed = True


__all__ = [
    "apply_patches",
    "setup_gate_result",
    "_enforce_permanent_context_only_role",
    "_remove_permanent_context_only_promotions",
]
