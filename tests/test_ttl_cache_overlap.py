from datetime import datetime, timedelta
import pandas as pd

from src.data.source import LiveKiteSource, WARMUP_BARS
from src.data.types import HistStatus


class FakeKite:
    def __init__(self) -> None:
        self.calls = 0

    def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
        self.calls += 1
        start = frm - timedelta(minutes=60)
        end = to + timedelta(minutes=60)
        idx = pd.date_range(start, end, freq="1min", inclusive="left")
        return [
            {
                "date": ts.to_pydatetime(),
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 0,
            }
            for ts in idx
        ]

    def ltp(self, instruments):
        return {str(instruments[0]): {"last_price": 1}}


def test_overlapping_requests_hit_cache():
    kite = FakeKite()
    src = LiveKiteSource(kite)
    start = datetime(2024, 1, 1, 9, 0)
    end = datetime(2024, 1, 1, 9, 30)
    res1 = src.fetch_ohlc(123, start, end, "minute")
    assert res1.status is HistStatus.OK
    df1 = res1.df
    assert kite.calls == 1

    start2 = datetime(2024, 1, 1, 9, 15)
    end2 = datetime(2024, 1, 1, 9, 45)
    res2 = src.fetch_ohlc(123, start2, end2, "minute")
    assert res2.status is HistStatus.OK
    df2 = res2.df
    assert kite.calls == 1  # cache hit
    assert len(df2) >= WARMUP_BARS
    assert df2.index.min() <= start2
    assert df2.index.max() >= end2
