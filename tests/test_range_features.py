from __future__ import annotations

"""Tests for the range feature helpers."""

from dataclasses import dataclass

from src.features.range import range_score


@dataclass
class DummyFeatures:
    mom_norm: float | None
    atr_pct: float | None


def test_range_score_in_band_low_momentum() -> None:
    features = DummyFeatures(mom_norm=0.1, atr_pct=0.05)

    score = range_score(features)

    assert 0.0 <= score <= 1.0
    # Low momentum and ATR within the preferred band should nearly max out the score.
    assert score == 0.9


def test_range_score_out_of_band_penalty() -> None:
    features = DummyFeatures(mom_norm=-0.25, atr_pct=0.5)

    score = range_score(features)

    # Outside the ATR band we expect the penalty multiplier.
    assert score == (1.0 - abs(features.mom_norm)) * 0.5


def test_range_score_defaults_to_zero() -> None:
    # Missing attributes should default to zero and stay within bounds.
    class Empty:  # pragma: no cover - structure definition only
        pass

    score = range_score(Empty())

    assert score == 0.5
