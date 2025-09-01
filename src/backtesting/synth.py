from __future__ import annotations

"""Synthetic data utilities for backtesting."""

from datetime import datetime
import pandas as pd


def make_synth_1m(
    start: datetime,
    minutes: int = 300,
    base: float = 22500.0,
    drift: float = 0.2,
    amp: float = 5.0,
) -> pd.DataFrame:
    """Generate a simple synthetic 1-minute OHLCV dataset.

    The price follows a deterministic wave so tests remain repeatable.
    """
    idx = pd.date_range(start=start, periods=minutes, freq="1min", tz="Asia/Kolkata")
    close: list[float] = []
    x = base
    for i in range(minutes):
        x += drift + (amp if i % 10 < 5 else -amp)
        close.append(x)
    s = pd.Series(close, index=idx, name="close")
    df = pd.DataFrame(
        {
            "open": s.shift(1).fillna(s.iloc[0]),
            "high": s + 2.0,
            "low": s - 2.0,
            "close": s,
            "volume": 1000,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df
