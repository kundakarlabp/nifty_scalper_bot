"""Fail-closed setup-quality gate for strategy vote combination.

Context may confirm or veto a valid trigger, but it must never convert a setup
that failed its own strategy threshold into an executable entry.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)

_SCORE_KEYS = ("raw_setup_score", "setup_score", "strategy_score")
_MIN_KEYS = ("setup_min", "setup_min_score", "trigger_min_score", "min_score")


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


def apply_patches() -> None:
    from nifty_scalper_bot.core import strategy_manager as strategy_module

    cls = strategy_module.StrategyManager
    if getattr(cls, "_hard_setup_score_gate_installed", False):
        return

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

        for signal, vote in signals:
            metadata = dict(getattr(vote, "metadata", {}) or {})
            role = str(metadata.get("role") or "trigger").strip().lower()
            is_close = getattr(signal, "action", None) in {"CLOSE_LONG", "CLOSE_SHORT"}
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

        eligible_trigger_exists = any(
            str((getattr(vote, "metadata", {}) or {}).get("role") or "trigger")
            .strip()
            .lower()
            != "context"
            and getattr(signal, "action", None) not in {"CLOSE_LONG", "CLOSE_SHORT"}
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


__all__ = ["apply_patches", "setup_gate_result"]
