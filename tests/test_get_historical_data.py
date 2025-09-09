from datetime import datetime, timedelta

import pandas as pd

from src.data.source import DataSource, get_historical_data


class StubDataSource(DataSource):
    def __init__(self, max_bars: int) -> None:
        self.max_bars = max_bars
        self.calls: list[tuple[datetime, datetime]] = []

    def fetch_ohlc(self, token, start: datetime, end: datetime, timeframe: str):
        self.calls.append((start, end))
        diff = int((end - start).total_seconds() // 60)
        bars = min(diff, self.max_bars)
        idx = pd.date_range(end - timedelta(minutes=bars), periods=bars, freq="1min")
        data = {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
        return pd.DataFrame(data, index=idx)

    def get_last_price(self, symbol):
        return 1.0


def test_get_historical_data_returns_warmup():
    ds = StubDataSource(max_bars=100)
    end = datetime(2024, 1, 1, 10, 0)
    df = get_historical_data(ds, token=1, end=end, timeframe="minute", warmup_bars=50)
    assert len(ds.calls) == 1
    assert df is not None
    assert len(df) == 50


def test_get_historical_data_retries_on_short_data():
    ds = StubDataSource(max_bars=20)
    end = datetime(2024, 1, 1, 10, 0)
    df = get_historical_data(ds, token=1, end=end, timeframe="minute", warmup_bars=50)
    # Should make four attempts as data never reaches warmup
    assert len(ds.calls) == 4
    assert df is not None
    assert len(df) == 20


def test_get_historical_data_floors_end_time():
    """End timestamps with seconds should be floored to the last minute."""
    ds = StubDataSource(max_bars=100)
    end = datetime(2024, 1, 1, 10, 0, 30)
    df = get_historical_data(ds, token=1, end=end, timeframe="minute", warmup_bars=10)

    assert df is not None
    assert len(ds.calls) == 1
    call_start, call_end = ds.calls[0]
    assert call_end == datetime(2024, 1, 1, 10, 0)
    assert call_start == datetime(2024, 1, 1, 9, 50)
    assert len(df) == 10
