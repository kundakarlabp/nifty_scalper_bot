from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.data_feed import SpotFeed


@pytest.mark.parametrize("ts_col", ["timestamp", "date", "time", "datetime"])
def test_from_csv_detects_timestamp_column(tmp_path, ts_col: str) -> None:
    df = pd.DataFrame(
        {
            ts_col: ["2025-01-01 09:15", "2025-01-01 09:16"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [1, 1],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    feed = SpotFeed.from_csv(str(path))
    assert feed.df.index[0] == pd.Timestamp("2025-01-01 09:15")


def test_from_csv_adj_close_alias(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 09:15"],
            "open": [1],
            "high": [1],
            "low": [1],
            "Adj Close": [123],
            "volume": [5],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    feed = SpotFeed.from_csv(str(path))
    assert feed.df.loc[pd.Timestamp("2025-01-01 09:15"), "close"] == 123


def test_from_csv_missing_volume(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 09:15", "2025-01-01 09:16"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    feed = SpotFeed.from_csv(str(path))
    assert "volume" in feed.df.columns
    assert (feed.df["volume"] == 0).all()
