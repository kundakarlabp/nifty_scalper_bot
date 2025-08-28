from datetime import datetime, timedelta

import pandas as pd

from src.data.source import LiveKiteSource
from src.config import settings


def test_fetch_ohlc_warmup(monkeypatch):
    warmup_bars = settings.strategy.min_bars_for_signal
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=warmup_bars)

    index = pd.date_range(start, periods=warmup_bars, freq="1min")
    data = {
        "Open": list(range(warmup_bars)),
        "High": list(range(warmup_bars)),
        "Low": list(range(warmup_bars)),
        "Close": list(range(warmup_bars)),
        "Volume": [0] * warmup_bars,
    }
    df = pd.DataFrame(data, index=index)

    monkeypatch.setattr("src.data.source._yf_symbol", lambda token: "TEST")
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= warmup_bars

