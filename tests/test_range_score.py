import types

from src.strategies.scalping_strategy import compute_score
from src.features import range_score


def test_compute_score_uses_range_score():
    feats = types.SimpleNamespace(mom_norm=0.1, atr_pct=0.05, range_score=range_score)
    score = compute_score(feats, "RANGE")
    assert 0.0 < score <= 1.0
