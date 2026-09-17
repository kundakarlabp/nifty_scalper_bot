"""Canonical arbitration for NIFTY underlying direction.

Option-premium data may trigger a setup but must never authorize NIFTY direction.
Only fresh spot/futures observations participate. Futures is the primary
price-discovery source when both underlying sources agree; spot is confirmation.
Disagreement remains fail-closed unless one source is materially stronger and
the opposing source is genuinely weak. A published old side that is already
carrying the confirmed side as its own reversal candidate is treated as
hysteresis lag, not as an independent contradiction.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DIRECTIONS = {"CE", "PE"}
_DOMINANCE_GAP = 0.20
_MIN_DOMINANT_CONFIDENCE = 0.70
_MAX_WEAK_DISAGREEMENT_CONFIDENCE = 0.60


@dataclass(frozen=True, slots=True)
class UnderlyingDirectionObservation:
    """Atomic direction evidence from one underlying market-data source."""

    bias: str
    confidence: float
    age_seconds: float
    source: str
    reversal_candidate: str | None = None
    reversal_observations: int = 0

    def __post_init__(self) -> None:
        bias = str(self.bias or "").upper()
        if bias not in _VALID_DIRECTIONS:
            raise ValueError(f"invalid underlying direction: {self.bias!r}")
        if self.age_seconds < 0:
            raise ValueError("direction observation age cannot be negative")
        candidate = str(self.reversal_candidate or "").upper()
        if candidate not in _VALID_DIRECTIONS:
            candidate = ""
        object.__setattr__(self, "bias", bias)
        object.__setattr__(
            self,
            "confidence",
            max(0.0, min(1.0, float(self.confidence))),
        )
        object.__setattr__(self, "age_seconds", float(self.age_seconds))
        object.__setattr__(self, "source", str(self.source or "unknown"))
        object.__setattr__(self, "reversal_candidate", candidate or None)
        object.__setattr__(
            self,
            "reversal_observations",
            max(0, int(self.reversal_observations or 0)),
        )


@dataclass(frozen=True, slots=True)
class UnderlyingDirectionResolution:
    """Result of reconciling fresh spot and futures direction observations."""

    observation: UnderlyingDirectionObservation | None
    conflict: bool = False
    confirming_source: str | None = None
    decision: str = "unresolved"


def _transition_candidate_alignment(
    first: UnderlyingDirectionObservation,
    second: UnderlyingDirectionObservation,
) -> UnderlyingDirectionResolution | None:
    """Resolve one-source hysteresis lag without weakening conflict thresholds."""

    for leading, lagging in ((first, second), (second, first)):
        if leading.bias == lagging.bias:
            continue
        if leading.confidence < _MIN_DOMINANT_CONFIDENCE:
            continue
        if lagging.confidence > _MAX_WEAK_DISAGREEMENT_CONFIDENCE:
            continue
        if lagging.reversal_candidate != leading.bias:
            continue
        if lagging.reversal_observations < 1:
            continue
        return UnderlyingDirectionResolution(
            observation=leading,
            confirming_source=f"{lagging.source}:transition_candidate",
            decision="transition_candidate_alignment",
        )
    return None


def arbitrate_underlying_direction(
    spot: UnderlyingDirectionObservation | None,
    futures: UnderlyingDirectionObservation | None,
) -> UnderlyingDirectionResolution:
    """Resolve direction while preserving atomic confidence/freshness provenance.

    Futures is primary only when both sources agree. Opposing credible current
    observations remain fail-closed. A source whose published side is retained
    only by its own hysteresis may confirm the opposite side through its explicit
    reversal candidate, preventing a false cross-source conflict during a real
    transition. Numeric dominance and weak-opposition thresholds are unchanged.
    """

    if spot is not None and futures is not None:
        if spot.bias == futures.bias:
            return UnderlyingDirectionResolution(
                observation=futures,
                confirming_source=spot.source,
                decision="agreement_futures_primary",
            )

        transition = _transition_candidate_alignment(spot, futures)
        if transition is not None:
            return transition

        confidence_gap = abs(spot.confidence - futures.confidence)
        if confidence_gap < _DOMINANCE_GAP:
            return UnderlyingDirectionResolution(
                observation=None,
                conflict=True,
                decision="conflict_comparable",
            )

        stronger = spot if spot.confidence > futures.confidence else futures
        weaker = futures if stronger is spot else spot
        if stronger.confidence < _MIN_DOMINANT_CONFIDENCE:
            return UnderlyingDirectionResolution(
                observation=None,
                conflict=True,
                decision="conflict_dominant_insufficient",
            )
        if weaker.confidence > _MAX_WEAK_DISAGREEMENT_CONFIDENCE:
            return UnderlyingDirectionResolution(
                observation=None,
                conflict=True,
                decision="conflict_credible_opposition",
            )

        return UnderlyingDirectionResolution(
            observation=stronger,
            confirming_source=f"{weaker.source}:weak_disagreement",
            decision="weak_disagreement_override",
        )

    observation = futures if futures is not None else spot
    if futures is not None:
        decision = "single_futures"
    elif spot is not None:
        decision = "single_spot"
    else:
        decision = "unresolved"
    return UnderlyingDirectionResolution(
        observation=observation,
        decision=decision,
    )
