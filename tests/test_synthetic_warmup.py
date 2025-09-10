"""Tests for the synthetic warm-up fallback bars."""

from unittest.mock import Mock

import pandas as pd


def test_synthetic_warmup_fallback(monkeypatch) -> None:
    import src.boot.synthetic_warmup  # noqa: F401  # ensure patches applied
    from src.data.source import LiveKiteSource

    kite = Mock()
    calls = {"n": 0}

    def hist_data(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return []  # broker returns empty for live fetch
        return [{"close": 22505}]  # previous session close

    kite.historical_data.side_effect = hist_data
    ds = LiveKiteSource(kite)
    ds.get_last_price = lambda _token: None

    df = ds._fetch_ohlc_df(
        token=256265,
        start=pd.Timestamp("2025-04-09 09:15"),
        end=pd.Timestamp("2025-04-09 09:45"),
        timeframe="minute",
    )
    assert df is not None
    if hasattr(df, "empty"):
        assert not df.empty
    res = ds.have_min_bars(30)
    assert isinstance(res, bool)
    assert res

