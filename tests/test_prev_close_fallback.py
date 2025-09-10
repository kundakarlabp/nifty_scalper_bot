from datetime import datetime, timedelta

import pandas as pd
from freezegun import freeze_time

from src.data.source import LiveKiteSource


@freeze_time("2024-01-04 09:20:00")
def test_prev_close_fallback_when_ltp_missing() -> None:
    class MockKite:
        def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
            idx = pd.date_range(frm, to, freq="1min", inclusive="left")
            return [
                {
                    "date": ts.to_pydatetime(),
                    "open": 100,
                    "high": 100,
                    "low": 100,
                    "close": 100,
                    "volume": 0,
                }
                for ts in idx
            ]

    kite = MockKite()
    src = LiveKiteSource(kite)
    src.cb_hist.allow = lambda: False  # type: ignore[assignment]
    src.get_last_price = lambda _token: None

    start = datetime(2024, 1, 4, 9, 0)
    end = start + timedelta(minutes=10)
    df = src._fetch_ohlc_df(token=1, start=start, end=end, timeframe="minute")

    assert not df.empty
    assert (df["close"] == 100).all()
    assert getattr(src, "_last_hist_reason") == "synthetic_prev_close"
