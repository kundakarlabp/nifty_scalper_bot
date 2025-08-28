import pandas as pd

from src.utils.indicators import calculate_atr


def test_calculate_atr_returns_series():
    data = {
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1000, 1000, 1000, 1000],
    }
    df = pd.DataFrame(data)

    atr_series = calculate_atr(df, period=3)

    # The function should now return a Series labelled 'atr'
    assert isinstance(atr_series, pd.Series)
    assert atr_series.name == "atr"
    assert len(atr_series) == len(df)


def test_calculate_atr_with_series_inputs_returns_series():
    data = {
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
    }
    df = pd.DataFrame(data)

    atr_series = calculate_atr(df["high"], df["low"], df["close"], period=3)

    assert isinstance(atr_series, pd.Series)
    assert atr_series.name == "atr"
    assert len(atr_series) == len(df)
