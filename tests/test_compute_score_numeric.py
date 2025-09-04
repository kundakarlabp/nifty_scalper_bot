"""Tests for numeric score attributes in compute_score."""

from types import SimpleNamespace

from src.strategies.scalping_strategy import compute_score


def test_compute_score_accepts_numeric_trend_score() -> None:
    """Numeric ``trend_score`` should be returned as-is."""

    feats = SimpleNamespace(trend_score=0.7)
    assert compute_score(feats, "TREND") == 0.7


def test_compute_score_accepts_numeric_range_score() -> None:
    """Numeric ``range_score`` should be returned as-is."""

    feats = SimpleNamespace(range_score=0.5)
    assert compute_score(feats, "RANGE") == 0.5

