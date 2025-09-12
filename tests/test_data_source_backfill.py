import pandas as pd
from datetime import datetime

from types import SimpleNamespace

from src.data.source import DataSource
from src.data.types import HistResult, HistStatus
from src.data import source as source_mod


class DummySource(DataSource):
    def __init__(self) -> None:
        self.fetch_calls = 0
        self.backfill_calls = 0
        self._df = pd.DataFrame()

    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str) -> HistResult:
        self.fetch_calls += 1
        status = HistStatus.OK if not self._df.empty else HistStatus.NO_DATA
        return HistResult(status=status, df=self._df)

    def ensure_backfill(self, *, required_bars: int, token: int = 0, timeframe: str = "minute") -> None:
        self.backfill_calls += 1
        start = datetime(2024, 1, 1, 9, 15)
        idx = pd.date_range(start, periods=required_bars, freq="1min")
        data = {
            "open": [1.0] * required_bars,
            "high": [1.0] * required_bars,
            "low": [1.0] * required_bars,
            "close": [1.0] * required_bars,
            "volume": [1] * required_bars,
        }
        self._df = pd.DataFrame(data, index=idx)

    def get_last_price(self, symbol_or_token):
        return None


def test_get_last_bars_backfills_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        source_mod,
        "settings",
        SimpleNamespace(instruments=SimpleNamespace(instrument_token=123)),
        raising=False,
    )
    ds = DummySource()
    out = ds.get_last_bars(5)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 5
    assert ds.backfill_calls == 1
    assert ds.fetch_calls >= 2
