"""Canonical arbitration for NIFTY underlying direction.

Option-premium data may trigger a setup but must never authorize NIFTY direction.
Only fresh spot/futures observations participate.  Futures is the primary
price-discovery source when both underlying sources agree; spot is confirmation.
Disagreement remains fail-closed unless one source is materially stronger, so a
source prior can never manufacture a trade from ambiguous evidence.
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
    """Resolve direction while preserving atomic confidence/freshness provenance.

    Futures is primary only when both sources agree.  This reflects its
    price-discovery role without turning that empirical prior into an override.
    When sources disagree, the existing conservative evidence rule is retained:
    comparable conviction fails closed and only a materially stronger,
    high-conviction observation can override weak disagreement.
    """

    if spot is not None and futures is not None:
        if spot.bias == futures.bias:
            return UnderlyingDirectionResolution(
                observation=futures,
                confirming_source=spot.source,
            )

        confidence_gap = abs(spot.confidence - futures.confidence)
        dominance_gap = 0.20
        if confidence_gap < dominance_gap:
            return UnderlyingDirectionResolution(observation=None, conflict=True)
        stronger = spot if spot.confidence > futures.confidence else futures
        weaker = futures if stronger is spot else spot
        if stronger.confidence < 0.70:
            return UnderlyingDirectionResolution(observation=None, conflict=True)
        return UnderlyingDirectionResolution(
            observation=stronger,
            confirming_source=f"{weaker.source}:weak_disagreement",
        )

    observation = futures if futures is not None else spot
    return UnderlyingDirectionResolution(observation=observation)
