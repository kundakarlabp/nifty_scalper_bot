import pandas as pd
from datetime import datetime, timedelta
from freezegun import freeze_time

from src.data.source import LiveKiteSource, DataException
from src.data.types import HistStatus
from src.utils.market_time import prev_session_bounds
from src.utils.time_windows import TZ


@freeze_time("2024-01-03 03:30:00")
def test_fetch_ohlc_preopen_refetch_prev_session():
    class PreOpenKite:
        def __init__(self):
            self.calls = []

        def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
            self.calls.append((frm, to))
            if frm.date() != datetime(2024, 1, 2).date():
                raise DataException(
                    "Historical fetch API call should not be made before the market opens"
                )
            idx = pd.date_range(frm, to, freq="1min", inclusive="left")
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

    kite = PreOpenKite()
    src = LiveKiteSource(kite)
    start = datetime(2024, 1, 3, 9, 0)
    end = start + timedelta(minutes=10)
    res = src.fetch_ohlc(123, start, end, "minute")
    assert res.status is HistStatus.OK
    df = res.df
    assert not df.empty
    prev_start, prev_end = prev_session_bounds(datetime(2024, 1, 3, 9, 0, tzinfo=TZ))
    call_start, call_end = kite.calls[-1]
    assert call_start.date() == prev_start.date()
    assert call_end.date() == prev_end.date()
    prev_start_naive = prev_start.replace(tzinfo=None)
    prev_end_naive = prev_end.replace(tzinfo=None) - timedelta(minutes=1)
    assert df.index.min() == prev_start_naive
    assert df.index.max() == prev_end_naive
