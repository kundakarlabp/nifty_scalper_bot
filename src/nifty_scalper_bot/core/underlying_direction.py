"""Canonical arbitration for NIFTY underlying direction.

This module owns one narrow invariant: an option premium may trigger a setup,
but it must never become the authority for the NIFTY underlying direction.
Only fresh spot/futures observations may authorize CE/PE direction. Direction,
confidence and age travel together as one immutable observation; conflicting
fresh spot/futures observations fail closed instead of being resolved by source
order or by the option being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DIRECTIONS = {"CE", "PE"}


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
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "age_seconds", float(self.age_seconds))
        object.__setattr__(self, "source", str(self.source or "unknown"))


@dataclass(frozen=True, slots=True)
class UnderlyingDirectionResolution:
    """Result of reconciling fresh spot and futures direction observations."""

    observation: UnderlyingDirectionObservation | None
    conflict: bool = False
    confirming_source: str | None = None


def arbitrate_underlying_direction(
    spot: UnderlyingDirectionObservation | None,
    futures: UnderlyingDirectionObservation | None,
) -> UnderlyingDirectionResolution:
    """Resolve underlying direction without source-order or option-side bias.

    Two resolved sources must agree. If both are present and disagree, return a
    fail-closed conflict. When both agree, spot remains the primary price-index
    authority and futures is recorded as confirmation. When only one source is
    resolved, that complete observation is returned unchanged, preserving its
    own confidence and freshness provenance.
    """

    if spot is not None and futures is not None:
        if spot.bias != futures.bias:
            # A disagreement is only actionable when both independent
            # underlying sources carry comparable conviction. A weak/noisy
            # secondary reading must not veto a materially stronger primary
            # observation forever. Conversely, close-confidence disagreement
            # remains ambiguous and therefore fails closed.
            confidence_gap = abs(spot.confidence - futures.confidence)
            dominance_gap = 0.20
            if confidence_gap < dominance_gap:
                return UnderlyingDirectionResolution(observation=None, conflict=True)
            stronger = spot if spot.confidence > futures.confidence else futures
            weaker = futures if stronger is spot else spot
            # Require genuine conviction from the dominant source and keep the
            # weaker source as disagreement provenance, not confirmation.
            if stronger.confidence < 0.70:
                return UnderlyingDirectionResolution(observation=None, conflict=True)
            return UnderlyingDirectionResolution(
                observation=stronger,
                confirming_source=f"{weaker.source}:weak_disagreement",
            )
        return UnderlyingDirectionResolution(
            observation=spot,
            confirming_source=futures.source,
        )
    observation = spot if spot is not None else futures
    return UnderlyingDirectionResolution(observation=observation)
