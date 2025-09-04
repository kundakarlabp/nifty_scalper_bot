import pandas as pd

from src.strategies.scalping_strategy import compute_score
from src.strategies.strategy_config import StrategyConfig


def test_compute_score_range_returns_breakdown() -> None:
    df = pd.DataFrame({"close": [100 + i for i in range(25)], "atr": [1.0] * 25})
    cfg = StrategyConfig.load("config/strategy.yaml")
    score, details = compute_score(df, "RANGE", cfg)
    assert 0.0 <= score <= 1.0
    assert details is not None
    assert "mr_dist" in details.parts
