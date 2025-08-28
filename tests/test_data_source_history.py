from datetime import datetime, timedelta

import pandas as pd
from freezegun import freeze_time

from src.config import settings
from src.data.source import LiveKiteSource

WARMUP_BARS = int(
    max(settings.data.lookback_minutes, settings.strategy.min_bars_for_signal)
)


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

    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end) - pd.Timedelta(minutes=1)
    window = df.loc[expected_start:expected_end]
    assert len(window) == 3 * 24 * 60
    assert df.index.min() <= expected_start
    assert df.index.max() >= expected_end
    assert len(df) >= WARMUP_BARS


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
