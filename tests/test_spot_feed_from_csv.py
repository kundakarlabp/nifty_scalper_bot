import pandas as pd
import pytest

from src.backtesting.data_feed import SpotFeed


def _write_csv(tmp_path, content: str):
    path = tmp_path / "data.csv"
    path.write_text(content)
    return path

@pytest.mark.parametrize(
    "col,value",
    [
        ("timestamp", "2025-04-09 09:15"),
        ("Timestamp", "2025-04-09 09:15"),
        ("datetime", "2025-04-09 09:15"),
        ("Date", "2025-04-09"),
    ],
)
def test_from_csv_timestamp_variants(tmp_path, col, value):
    csv = f"{col},Open,High,Low,Close,Volume\n{value},1,2,0.5,1.5,10\n"
    feed = SpotFeed.from_csv(str(_write_csv(tmp_path, csv)))
    assert len(feed.df) == 1
    assert feed.df.index[0] == pd.to_datetime(value)


def test_from_csv_date_time_columns(tmp_path):
    csv = "Date,Time,Open,High,Low,Close,Volume\n2025-04-09,09:15,1,2,0.5,1.5,10\n"
    feed = SpotFeed.from_csv(str(_write_csv(tmp_path, csv)))
    assert len(feed.df) == 1
    assert feed.df.index[0] == pd.to_datetime("2025-04-09 09:15")


def test_from_csv_adj_close_alias(tmp_path):
    csv = "timestamp,Open,High,Low,Adj Close,Volume\n2025-04-09 09:15,1,2,0.5,1.5,10\n"
    feed = SpotFeed.from_csv(str(_write_csv(tmp_path, csv)))
    assert "close" in feed.df.columns
    assert feed.df.iloc[0]["close"] == 1.5


def test_from_csv_missing_volume(tmp_path):
    csv = "timestamp,Open,High,Low,Close\n2025-04-09 09:15,1,2,0.5,1.5\n"
    feed = SpotFeed.from_csv(str(_write_csv(tmp_path, csv)))
    assert "volume" in feed.df.columns
    assert feed.df["volume"].tolist() == [0]
