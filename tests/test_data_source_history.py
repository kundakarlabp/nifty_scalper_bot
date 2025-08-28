from datetime import datetime, timedelta

import pandas as pd
from freezegun import freeze_time

from src.data.source import LiveKiteSource


class FakeKite:
    def __init__(self):
        self.calls = []

    def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
        self.calls.append((frm, to))
        idx = pd.date_range(frm, to, freq='1min', inclusive='left')
        return [
            {
                'date': ts.to_pydatetime(),
                'open': 1,
                'high': 1,
                'low': 1,
                'close': 1,
                'volume': 0,
            }
            for ts in idx
        ]

    def ltp(self, instruments):
        return {str(instruments[0]): {'last_price': 1}}


def test_fetch_ohlc_chunks_multiple_calls():
    kite = FakeKite()
    src = LiveKiteSource(kite)
    end = datetime(2024, 1, 4)
    start = end - timedelta(days=3)
    df = src.fetch_ohlc(123, start, end, 'minute')
    assert len(kite.calls) > 1  # chunked into multiple requests
    assert len(df) == 3 * 24 * 60
    assert df.index.min() == pd.Timestamp(start)
    assert df.index.max() == pd.Timestamp(end) - pd.Timedelta(minutes=1)


def test_fetch_ohlc_cached_window_clipped():
    kite = FakeKite()
    src = LiveKiteSource(kite)
    end = datetime(2024, 1, 4, 0, 10)
    start = end - timedelta(minutes=10)
    warm_start = start - timedelta(minutes=5)

    # Initial call fetches wider window (warmup)
    src.fetch_ohlc(123, warm_start, end, 'minute')
    assert len(kite.calls) == 1

    # Second call should hit cache and be clipped to requested window
    df = src.fetch_ohlc(123, start, end, 'minute')
    assert len(kite.calls) == 1  # served from cache
    assert df.index.min() == pd.Timestamp(start)
    assert df.index.max() == pd.Timestamp(end) - pd.Timedelta(minutes=1)


@freeze_time("2024-01-01 09:30:00")
def test_fetch_ohlc_network_failure_falls_back_to_ltp():
    class BoomKite:
        def historical_data(self, *args, **kwargs):
            raise Exception("boom")

        def ltp(self, instruments):
            return {str(instruments[0]): {"last_price": 1}}

    src = LiveKiteSource(BoomKite())
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=1)
    df = src.fetch_ohlc(123, start, end, "minute")
    assert len(df) == 1
    assert df.iloc[0].close == 1
