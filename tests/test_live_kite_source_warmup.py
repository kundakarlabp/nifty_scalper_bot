from datetime import datetime, timedelta, timezone
import importlib

import pandas as pd


def _setup(monkeypatch):
    """Reload data source with yfinance enabled."""

    monkeypatch.setenv("YFINANCE_DISABLE", "false")
    monkeypatch.setenv("DATA__WARMUP_DISABLE", "false")
    import src.data.source as src_module

    return importlib.reload(src_module)


def test_fetch_ohlc_warmup(monkeypatch):
    src_module = _setup(monkeypatch)
    LiveKiteSource = src_module.LiveKiteSource
    WARMUP_BARS = src_module.WARMUP_BARS

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

    monkeypatch.setattr(src_module, "_yf_symbol", lambda token: "TEST")
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= WARMUP_BARS
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    assert out.index.tz is None


def test_fetch_ohlc_timezone_aware_inputs(monkeypatch):
    src_module = _setup(monkeypatch)
    LiveKiteSource = src_module.LiveKiteSource
    WARMUP_BARS = src_module.WARMUP_BARS

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

    monkeypatch.setattr(src_module, "_yf_symbol", lambda token: "TEST")
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: df)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")

    assert out is not None
    assert len(out) >= WARMUP_BARS
    assert out.index.tz is None

