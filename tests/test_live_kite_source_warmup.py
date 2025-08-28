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

    monkeypatch.setattr("src.data.source._yf_symbol", lambda token: "TEST")
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= WARMUP_BARS

