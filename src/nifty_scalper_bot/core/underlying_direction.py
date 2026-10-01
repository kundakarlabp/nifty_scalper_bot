"""Canonical arbitration for NIFTY underlying direction.

Option-premium data may trigger a setup but must never authorize NIFTY direction.
Only fresh spot/futures observations participate. Futures is the primary
price-discovery source when both underlying sources agree; spot is confirmation.
Any fresh disagreement remains fail-closed. This treats opposing underlying
views as a transition/reversal warning instead of forcing a CE/PE decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_VALID_DIRECTIONS = {"CE", "PE"}


class UnderlyingDirectionState(str, Enum):
    """Semantic underlying state; CE/PE remains the execution compatibility view."""

    CONFIRMED_BULL = "CONFIRMED_BULL"
    CONFIRMED_BEAR = "CONFIRMED_BEAR"
    TRANSITION = "TRANSITION"
    UNAVAILABLE = "UNAVAILABLE"


def _state_for_bias(bias: str | None) -> UnderlyingDirectionState:
    if bias == "CE":
        return UnderlyingDirectionState.CONFIRMED_BULL
    if bias == "PE":
        return UnderlyingDirectionState.CONFIRMED_BEAR
    return UnderlyingDirectionState.UNAVAILABLE



@dataclass(frozen=True, slots=True)
class UnderlyingDirectionObservation:
    """Atomic direction evidence from one underlying market-data source."""

    bias: str
    confidence: float
    age_seconds: float
    source: str

    def __post_init__(self) -> None:
        bias = str(self.bias or "").upper()
        if bias not in _VALID_DIRECTIONS:
            raise ValueError(f"invalid underlying direction: {self.bias!r}")
        if self.age_seconds < 0:
            raise ValueError("direction observation age cannot be negative")
        object.__setattr__(self, "bias", bias)
        object.__setattr__(
            self, "confidence", max(0.0, min(1.0, float(self.confidence)))
        )
        object.__setattr__(self, "age_seconds", float(self.age_seconds))
        object.__setattr__(self, "source", str(self.source or "unknown"))


@dataclass(frozen=True, slots=True)
class UnderlyingDirectionResolution:
    """Result of reconciling fresh spot and futures direction observations."""

    observation: UnderlyingDirectionObservation | None
    conflict: bool = False
    confirming_source: str | None = None
    state: UnderlyingDirectionState = UnderlyingDirectionState.UNAVAILABLE
    reason: str = "unavailable"

    @property
    def executable_bias(self) -> str | None:
        """Backward-compatible CE/PE expression of the resolved state."""
        return self.observation.bias if self.observation is not None else None


def arbitrate_underlying_direction(
    spot: UnderlyingDirectionObservation | None,
    futures: UnderlyingDirectionObservation | None,
) -> UnderlyingDirectionResolution:
    """Resolve direction while preserving atomic confidence/freshness provenance.

    Futures is primary only when both sources agree. This reflects its empirical
    price-discovery role without turning that prior into an unconditional
    override. Opposing credible observations are treated as a possible
    transition/reversal and fail closed.
    """

    if spot is not None and futures is not None:
        if spot.bias == futures.bias:
            return UnderlyingDirectionResolution(
                observation=futures,
                confirming_source=spot.source,
                state=_state_for_bias(futures.bias),
                reason="spot_futures_agree",
            )

        # Canonical execution invariant: two fresh underlying authorities that
        # disagree are a transition. Confidence dominance is diagnostic only;
        # it must never manufacture an executable CE/PE bias from conflict.
        return UnderlyingDirectionResolution(
            observation=None,
            conflict=True,
            state=UnderlyingDirectionState.TRANSITION,
            reason="fresh_spot_futures_disagreement",
        )

    observation = futures if futures is not None else spot
    return UnderlyingDirectionResolution(
        observation=observation,
        state=_state_for_bias(observation.bias if observation else None),
        reason=(
            "single_fresh_underlying_source"
            if observation
            else "no_fresh_underlying_source"
        ),
    )
