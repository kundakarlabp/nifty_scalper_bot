from datetime import datetime, timedelta

import pandas as pd

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
