import pandas as pd
from datetime import datetime, timedelta, timezone

from src.features.health import check
from src.features.indicators import atr_pct


def _sample_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range(datetime.now(timezone.utc) - timedelta(minutes=n), periods=n, freq="1min")
    data = {
        "high": pd.Series(range(1, n + 1), index=idx),
        "low": pd.Series(range(n), index=idx),
        "close": pd.Series(range(1, n + 1), index=idx),
    }
    return pd.DataFrame(data)


def test_atr_pct_returns_value():
    df = _sample_df(15)
    val = atr_pct(df, period=14)
    assert val is not None and val > 0


def test_atr_pct_insufficient_bars_returns_none():
    df = _sample_df(5)
    assert atr_pct(df, period=14) is None


def test_feature_health_flags_short_bars():
    df = _sample_df(5)
    fh = check(df, df.index[-1].to_pydatetime(), atr_period=14)
    assert not fh.bars_ok
    assert "bars_short" in fh.reasons


def test_feature_health_flags_stale_data():
    df = _sample_df(20)
    old_ts = df.index[-1].to_pydatetime() - timedelta(minutes=10)
    fh = check(df, old_ts, atr_period=14, max_age_s=60)
    assert not fh.fresh_ok
    assert "data_stale" in fh.reasons

