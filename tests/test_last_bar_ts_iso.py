from __future__ import annotations

import pandas as pd

from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def test_last_bar_ts_isoformat() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        },
        index=[pd.Timestamp("2024-01-01T09:15:00")],
    )
    strategy = EnhancedScalpingStrategy()
    plan = strategy.generate_signal(df)
    assert plan["last_bar_ts"] == df.index[-1].to_pydatetime().isoformat()
