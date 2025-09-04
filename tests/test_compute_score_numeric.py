"""Tests for numeric behaviour in compute_score."""

from types import SimpleNamespace

import pandas as pd

from src.strategies.scalping_strategy import compute_score


def _make_df(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"close": list(range(n)), "atr": [1.0] * n})


def test_compute_score_trend() -> None:
    df = _make_df()
    cfg = SimpleNamespace(ema_fast=3, ema_slow=7, warmup_bars_min=20)
    score, details = compute_score(df, "TREND", cfg)
    assert 0.0 <= score <= 1.0
    assert details and details.regime == "TREND"


def test_compute_score_unknown_regime_returns_zero() -> None:
    df = _make_df()
    cfg = SimpleNamespace(ema_fast=3, ema_slow=7, warmup_bars_min=20)
    score, details = compute_score(df, "UNKNOWN", cfg)
    assert score == 0.0
    assert details is None
