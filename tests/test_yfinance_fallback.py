from datetime import datetime, timedelta

import pandas as pd

from src.data.source import LiveKiteSource, WARMUP_BARS


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

