from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.ohlc_builder import calc_bar_lag_s


def _df_with_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=index
    )


def test_calc_bar_lag_naive_df_aware_now() -> None:
    idx = pd.date_range("2024-01-01 09:15", periods=1, freq="T")
    df = _df_with_index(idx)
    now = datetime(2024, 1, 1, 9, 16, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert calc_bar_lag_s(df, now) == 60


def test_calc_bar_lag_aware_df_naive_now() -> None:
    idx = pd.date_range(
        "2024-01-01 09:15", periods=1, freq="T", tz=ZoneInfo("Asia/Kolkata")
    )
    df = _df_with_index(idx)
    now = datetime(2024, 1, 1, 9, 16)
    assert calc_bar_lag_s(df, now) == 60
