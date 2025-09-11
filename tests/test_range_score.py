from types import SimpleNamespace

import pandas as pd

from src.strategies.scalping_strategy import compute_score


def test_compute_score_uses_range_score() -> None:
    df = pd.DataFrame({
        "close": [100.0] * 29 + [102.0],
        "atr": [1.0] * 30,
    })
    cfg = SimpleNamespace(bb_period=20, enable_range_scoring=True, warmup_bars_min=15)
    score, details = compute_score(df, "RANGE", cfg)
    assert 0.0 <= score <= 1.0
    assert details and details.regime == "RANGE"
