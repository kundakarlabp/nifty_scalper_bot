"""Basic tests for compute_score helpers."""

import pandas as pd

from src.strategies.scalping_strategy import compute_score
from src.strategies.strategy_config import StrategyConfig


def test_compute_score_trend_returns_breakdown() -> None:
    df = pd.DataFrame({"close": [i for i in range(30)], "atr": [1.0] * 30})
    cfg = StrategyConfig.load("config/strategy.yaml")
    score, details = compute_score(df, "TREND", cfg)
    assert 0.0 <= score <= 1.0
    assert details is not None
    assert "ema_slope" in details.parts

