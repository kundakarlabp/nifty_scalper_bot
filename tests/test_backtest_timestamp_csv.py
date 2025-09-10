import pandas as pd
from io import StringIO


def test_flexible_timestamp_detection():
    csv = "Date,Time,Open,High,Low,Close,Volume\n2025-04-09,09:15,100,101,99,100.5,10\n"
    import src.backtesting.data_feed as dfmod

    df = pd.read_csv(StringIO(csv))
    df = dfmod._normalize_cols(df)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 1
