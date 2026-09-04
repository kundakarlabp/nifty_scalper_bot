"""Fail-closed confirmation for rapid underlying direction reversals.

The gate accepts the first valid direction immediately. A CE<->PE reversal is
withheld until the new direction is both sufficiently confident and persistent
for a minimum number of observations and wall-clock duration. During the
transition no direction is exposed to option-entry strategies, preventing a
single noisy context update from immediately reversing live exposure intent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from nifty_scalper_bot.config.env_utils import parse_float_env, parse_int_env


@dataclass(slots=True, frozen=True)
class DirectionStabilityDecision:
    direction: str | None
    confidence: float
    pending: bool
    candidate_direction: str | None = None
    candidate_updates: int = 0
    candidate_age_seconds: float = 0.0


@dataclass(slots=True)
class _DirectionState:
    accepted_direction: str | None = None
    candidate_direction: str | None = None
    candidate_since: float = 0.0
    candidate_updates: int = 0


class DirectionStabilityGate:
    """Confirm CE/PE flips while remaining fail-closed during transition."""

    def __init__(
        self,
        *,
        confirm_seconds: float = 5.0,
        confirm_updates: int = 3,
        min_confidence: float = 0.60,
    ) -> None:
        self.confirm_seconds = max(0.0, float(confirm_seconds))
        self.confirm_updates = max(1, int(confirm_updates))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self._states: dict[str, _DirectionState] = {}

    @classmethod
    def from_env(cls) -> "DirectionStabilityGate":
        return cls(
            confirm_seconds=parse_float_env(
                os.getenv("DIRECTION_FLIP_CONFIRM_SECONDS"), 5.0
            ),
            confirm_updates=parse_int_env(
                os.getenv("DIRECTION_FLIP_CONFIRM_UPDATES"), 3
            ),
            min_confidence=parse_float_env(
                os.getenv("DIRECTION_FLIP_MIN_CONFIDENCE"), 0.60
            ),
        )

    def observe(
        self,
        role: str,
        direction: str | None,
        confidence: float,
        *,
        now: float | None = None,
    ) -> DirectionStabilityDecision:
        key = str(role or "context").strip().lower() or "context"
        state = self._states.setdefault(key, _DirectionState())
        timestamp = time.time() if now is None else float(now)
        normalized = str(direction or "").strip().upper()
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            conf = 0.0

        if normalized not in {"CE", "PE"}:
            state.candidate_direction = None
            state.candidate_since = 0.0
            state.candidate_updates = 0
            return DirectionStabilityDecision(None, 0.0, False)

        if state.accepted_direction is None:
            state.accepted_direction = normalized
            return DirectionStabilityDecision(normalized, conf, False)

        if normalized == state.accepted_direction:
            state.candidate_direction = None
            state.candidate_since = 0.0
            state.candidate_updates = 0
            return DirectionStabilityDecision(normalized, conf, False)

        if conf < self.min_confidence:
            state.candidate_direction = None
            state.candidate_since = 0.0
            state.candidate_updates = 0
            return DirectionStabilityDecision(
                None,
                0.0,
                True,
                candidate_direction=normalized,
                candidate_updates=0,
                candidate_age_seconds=0.0,
            )

        if state.candidate_direction != normalized:
            state.candidate_direction = normalized
            state.candidate_since = timestamp
            state.candidate_updates = 1
        else:
            state.candidate_updates += 1

        age = max(0.0, timestamp - state.candidate_since)
        if (
            state.candidate_updates >= self.confirm_updates
            and age >= self.confirm_seconds
        ):
            state.accepted_direction = normalized
            state.candidate_direction = None
            state.candidate_since = 0.0
            state.candidate_updates = 0
            return DirectionStabilityDecision(normalized, conf, False)

        return DirectionStabilityDecision(
            None,
            0.0,
            True,
            candidate_direction=normalized,
            candidate_updates=state.candidate_updates,
            candidate_age_seconds=age,
        )


__all__ = ["DirectionStabilityDecision", "DirectionStabilityGate"]
