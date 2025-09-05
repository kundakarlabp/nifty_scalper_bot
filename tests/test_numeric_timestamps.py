from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtesting.data_feed import SpotFeed
from src.backtesting.data_source import BacktestCsvSource

def _make_epoch_df(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1min")
    epoch = ts.view("int64") // 10**9
    return pd.DataFrame(
        {
            "timestamp": epoch,
            "open": range(100, 100 + periods),
            "high": range(101, 101 + periods),
            "low": range(99, 99 + periods),
            "close": range(100, 100 + periods),
            "volume": [1] * periods,
        }
    )

def test_spotfeed_from_csv_epoch_seconds(tmp_path):
    df = _make_epoch_df("2025-08-01 09:15", 2)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    feed = SpotFeed.from_csv(str(csv_path))
    assert feed.df.index[0] == pd.Timestamp("2025-08-01 09:15")

def test_backtest_csv_source_epoch_seconds(tmp_path):
    df = _make_epoch_df("2025-08-01 09:15", 25)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    src = BacktestCsvSource(csv_path, symbol="NIFTY")
    expected = datetime(2025, 8, 1, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert src.current_datetime == expected
