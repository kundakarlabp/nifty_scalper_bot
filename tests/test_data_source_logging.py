import logging
from types import SimpleNamespace
from datetime import datetime
import pandas as pd

from src.data.source import DataSource
from src.data import source as source_mod


class EmptySource(DataSource):
    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str):
        return pd.DataFrame()

    def ensure_backfill(self, *, required_bars: int, token: int = 0, timeframe: str = "minute") -> None:
        raise RuntimeError("backfill failed")

    def get_last_price(self, symbol_or_token):
        return None


def test_logs_backfill_failure(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        source_mod,
        "settings",
        SimpleNamespace(instruments=SimpleNamespace(instrument_token=123)),
        raising=False,
    )
    ds = EmptySource()
    with caplog.at_level(logging.WARNING):
        out = ds.get_last_bars(5)
    assert out is None
    assert "ensure_backfill failed" in caplog.text


class VWAPSource(DataSource):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str):
        return self._df

    def get_last_price(self, symbol_or_token):
        return None


def test_logs_vwap_failure(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        source_mod,
        "settings",
        SimpleNamespace(instruments=SimpleNamespace(instrument_token=123)),
        raising=False,
    )
    idx = pd.date_range("2024-01-01 09:15", periods=100, freq="1min")
    data = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    df = pd.DataFrame(data, index=idx)
    ds = VWAPSource(df)

    def _raise(*args, **kwargs):
        raise RuntimeError("vwap fail")

    monkeypatch.setattr(source_mod, "calculate_vwap", _raise)

    with caplog.at_level(logging.DEBUG):
        out = ds.get_last_bars(5)
    assert out is not None
    assert "vwap" not in out.columns
    assert "calculate_vwap failed" in caplog.text
