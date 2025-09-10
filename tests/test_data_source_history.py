from datetime import datetime, timedelta

import pandas as pd
from freezegun import freeze_time

from src.config import settings
from src.data.source import (
    LiveKiteSource,
    WARMUP_BARS,
    _normalize_ohlc_df,
    render_last_bars,
)
from src.data.types import HistStatus


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
    res = src.fetch_ohlc(123, start, end, 'minute')
    assert res.status is HistStatus.OK
    df = res.df
    assert len(kite.calls) > 1  # chunked into multiple requests

    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end) - pd.Timedelta(minutes=1)
    window = df.loc[expected_start:expected_end]
    assert len(window) == 3 * 24 * 60
    assert df.index.min() <= expected_start
    assert df.index.max() >= expected_end
    assert len(df) >= WARMUP_BARS


def test_have_min_bars_returns_bool() -> None:
    kite = FakeKite()
    src = LiveKiteSource(kite)
    res = src.have_min_bars(10)
    assert isinstance(res, bool)
    assert res
    bars = src.get_recent_bars(5)
    assert len(bars) == 5


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
    res = src.fetch_ohlc(123, start, end, "minute")
    assert res.status is HistStatus.OK
    assert len(res.df) >= WARMUP_BARS
    assert (res.df.close == 1).all()


def test_normalize_drops_invalid_rows() -> None:
    raw = pd.DataFrame(
        {
            "open": [1, -1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [0, 0],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        }
    )
    out = _normalize_ohlc_df(raw)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_render_last_bars_outputs_string(monkeypatch) -> None:
    kite = FakeKite()
    src = LiveKiteSource(kite)
    monkeypatch.setattr(settings.instruments, "instrument_token", 123)
    out = render_last_bars(src, n=1)
    assert isinstance(out, str)
    assert "O=" in out
