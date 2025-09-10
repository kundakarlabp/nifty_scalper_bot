from unittest.mock import Mock
import pandas as pd

from src.data.source import LiveKiteSource, WARMUP_BARS


def test_synthetic_bars_from_last_session() -> None:
    kite = Mock()
    kite.historical_data.return_value = [
        {
            "date": "2025-04-08 15:29:00",
            "open": 22500,
            "high": 22510,
            "low": 22490,
            "close": 22505,
            "volume": 100,
        }
    ]
    source = LiveKiteSource(kite)
    source.cb_hist.allow = lambda: False  # simulate broker unavailable
    source.get_last_price = lambda _token: None

    start = pd.Timestamp("2025-04-09 09:15")
    end = pd.Timestamp("2025-04-09 09:45")
    df = source._fetch_ohlc_df(token=256265, start=start, end=end, timeframe="minute")

    assert not df.empty
    assert df.iloc[-1]["close"] == 22505
    assert len(df) >= WARMUP_BARS
