from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.source import LiveKiteSource, WARMUP_BARS


def test_fetch_ohlc_warmup(monkeypatch):
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=WARMUP_BARS)

    ist = timezone(timedelta(hours=5, minutes=30))
    index = pd.date_range(start, periods=WARMUP_BARS, freq="1min", tz=ist)
    data = {
        "Open": list(range(WARMUP_BARS)),
        "High": list(range(WARMUP_BARS)),
        "Low": list(range(WARMUP_BARS)),
        "Close": list(range(WARMUP_BARS)),
        "Volume": [0] * WARMUP_BARS,
    }
    df = pd.DataFrame(data, index=index)
    df = df.rename(columns=str.lower)
    df.index = df.index.tz_localize(None)

    monkeypatch.setattr("src.data.source._fetch_ohlc_yf", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= WARMUP_BARS
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    assert out.index.tz is None


def test_fetch_ohlc_timezone_aware_inputs(monkeypatch):
    ist = timezone(timedelta(hours=5, minutes=30))
    start = datetime(2024, 1, 1, 9, 0, tzinfo=ist)
    end = start + timedelta(minutes=WARMUP_BARS)

    index = pd.date_range(start, periods=WARMUP_BARS, freq="1min", tz=ist)
    data = {
        "Open": list(range(WARMUP_BARS)),
        "High": list(range(WARMUP_BARS)),
        "Low": list(range(WARMUP_BARS)),
        "Close": list(range(WARMUP_BARS)),
        "Volume": [0] * WARMUP_BARS,
    }
    df = pd.DataFrame(data, index=index)
    df = df.rename(columns=str.lower)
    df.index = df.index.tz_localize(None)

    monkeypatch.setattr("src.data.source._fetch_ohlc_yf", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= WARMUP_BARS
    assert out.index.tz is None

