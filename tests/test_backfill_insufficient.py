from datetime import datetime
import pandas as pd
import src.data.source as source


def _make_df(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="1min")
    data = {
        "Open": [1.0] * n,
        "High": [1.0] * n,
        "Low": [1.0] * n,
        "Close": [1.0] * n,
        "Volume": [0] * n,
    }
    return pd.DataFrame(data, index=idx)


def test_backfill_falls_back_when_hist_short(monkeypatch):
    ds = source.LiveKiteSource(kite=None)
    seeded: list[pd.DataFrame] = []
    ds.seed_ohlc = lambda df: seeded.append(df)

    class KiteStub:
        def historical_data(self, token, from_dt, to_dt, timeframe):
            return [
                {
                    "date": datetime(2024, 1, 1, 9, 15),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 0,
                }
            ]

    ds.kite = KiteStub()

    class YFStub:
        def download(self, *args, **kwargs):
            return _make_df(2)

    monkeypatch.setattr(source, "yf", YFStub())

    ds.ensure_backfill(required_bars=2, token=123, timeframe="minute")
    assert seeded and len(seeded[-1]) >= 2
