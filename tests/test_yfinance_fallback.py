from datetime import datetime, timedelta

import pandas as pd

from src.data.source import LiveKiteSource


def test_fetch_ohlc_yfinance_fallback(monkeypatch):
    start = datetime(2024, 1, 1, 9, 0)
    end = start + timedelta(minutes=2)
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [0, 0],
        },
        index=[start, start + timedelta(minutes=1)],
    )

    def fake_download(*args, **kwargs):
        return df

    monkeypatch.setattr("yfinance.download", fake_download)

    src = LiveKiteSource(kite=None)
    out = src.fetch_ohlc(123, start, end, "minute")
    assert out is not None and len(out) == 2
    assert out.iloc[0].close == 1


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
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [0, 0],
        },
        index=[start, start + timedelta(minutes=1)],
    )

    class BoomKite:
        def historical_data(self, *args, **kwargs):
            raise Exception("no sub")

    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: df)

    src = LiveKiteSource(BoomKite())
    out = src.fetch_ohlc(123, start, end, "minute")
    assert out is not None and len(out) == 2

