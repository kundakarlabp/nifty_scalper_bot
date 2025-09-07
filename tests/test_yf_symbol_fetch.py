from datetime import datetime, timedelta
import importlib

import pandas as pd
import pytest


def _reload(monkeypatch):
    monkeypatch.setenv("YFINANCE_DISABLE", "false")
    monkeypatch.setenv("DATA__WARMUP_DISABLE", "false")
    import src.data.source as src_module

    return importlib.reload(src_module)


def test_yf_symbol_maps_index(monkeypatch):
    src = _reload(monkeypatch)
    assert src._yf_symbol("NSE:NIFTY 50") == "^NSEI"


def test_fetch_ohlc_yf_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    src = _reload(monkeypatch)

    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(src, "yf", type("T", (), {"download": fake_download}))
    start = datetime(2024, 1, 1)
    end = start + timedelta(minutes=1)
    with pytest.raises(ValueError):
        src._fetch_ohlc_yf("^NSEI", start, end, "minute")


def test_fetch_ohlc_yf_returns_minute_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    src = _reload(monkeypatch)

    def fake_download(symbol, start, end, interval, progress):
        assert interval == "1m"
        idx = pd.date_range("2024-01-01 09:15", periods=2, freq="1min")
        data = {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0, 1],
            "Close": [1, 2],
            "Volume": [10, 20],
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(src, "yf", type("T", (), {"download": fake_download}))
    start = datetime(2024, 1, 1, 9, 15)
    end = start + timedelta(minutes=2)
    df = src._fetch_ohlc_yf("^NSEI", start, end, "minute")
    assert "volume" in df.columns
    assert (df.index[1] - df.index[0]).total_seconds() == 60

