from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.source import (
    LiveKiteSource,
    WARMUP_BARS,
    _fetch_ohlc_yf,
    _yf_symbol,
)


def test_fetch_ohlc_yfinance_fallback(monkeypatch):
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=2)

    def fake_download(symbol, start, end, interval, progress=False):
        idx = pd.date_range(start, periods=WARMUP_BARS, freq="1min")
        data = {
            "Open": [1] * WARMUP_BARS,
            "High": [1] * WARMUP_BARS,
            "Low": [1] * WARMUP_BARS,
            "Close": [1] * WARMUP_BARS,
            "Volume": [0] * WARMUP_BARS,
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr("yfinance.download", fake_download)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")
    assert out is not None and len(out) >= WARMUP_BARS
    assert (out.close == 1).all()


def test_get_last_price_yfinance_fallback(monkeypatch):
    hist = pd.DataFrame({"Close": [100]}, index=[pd.Timestamp("2024-01-01")])

    class DummyTicker:
        def history(self, *args, **kwargs):
            return hist

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: DummyTicker())

    src = LiveKiteSource(kite=None)
    price = src.get_last_price("NSE:FOO")
    assert price == 100.0


def test_fetch_ohlc_kite_error_uses_yfinance(monkeypatch):
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=2)

    def fake_download(symbol, start, end, interval, progress=False):
        idx = pd.date_range(start, periods=WARMUP_BARS, freq="1min")
        data = {
            "Open": [1] * WARMUP_BARS,
            "High": [1] * WARMUP_BARS,
            "Low": [1] * WARMUP_BARS,
            "Close": [1] * WARMUP_BARS,
            "Volume": [0] * WARMUP_BARS,
        }
        return pd.DataFrame(data, index=idx)

    class BoomKite:
        def historical_data(self, *args, **kwargs):
            raise Exception("no sub")

    monkeypatch.setattr("yfinance.download", fake_download)

    src = LiveKiteSource(BoomKite())
    out = src.fetch_ohlc(123, start, end, "minute")
    assert out is not None and len(out) >= WARMUP_BARS


def test_fetch_ohlc_yfinance_timezone(monkeypatch):
    """Ensure naive IST inputs are converted to UTC for yfinance calls."""
    captures = {}

    ist = timezone(timedelta(hours=5, minutes=30))

    def fake_download(symbol, start, end, interval, progress=False):
        captures["start"] = start
        captures["end"] = end
        captures["interval"] = interval
        idx = pd.date_range(
            pd.Timestamp("2024-01-01 09:15", tz=ist),
            periods=WARMUP_BARS,
            freq="1min",
        )
        data = {
            "Open": [1] * WARMUP_BARS,
            "High": [1] * WARMUP_BARS,
            "Low": [1] * WARMUP_BARS,
            "Close": [1] * WARMUP_BARS,
            "Volume": [0] * WARMUP_BARS,
        }
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr("yfinance.download", fake_download)

    start = datetime(2024, 1, 1, 9, 15)
    end = start + timedelta(minutes=1)
    df = _fetch_ohlc_yf("FOO", start, end, "minute")

    assert captures["start"].tzinfo == timezone.utc
    assert captures["start"].hour == 3 and captures["start"].minute == 45
    assert df is not None
    assert df.index[0] == pd.Timestamp("2024-01-01 09:15")


def test_yf_symbol_numeric_token_returns_none():
    from src.config import settings

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings.instruments, "spot_symbol", None, raising=False)
    monkeypatch.setattr(settings.instruments, "trade_symbol", None, raising=False)
    try:
        assert _yf_symbol(123456) is None
    finally:
        monkeypatch.undo()
