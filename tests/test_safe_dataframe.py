from datetime import datetime

import pandas as pd

from src.data.source import _safe_dataframe


def test_safe_dataframe_handles_malformed_rows() -> None:
    rows = [
        {
            "Date": "2024-01-01 09:15:00+05:30",
            "Open": "100",
            "High": "101",
            "Low": "99",
            "Close": "100",
            "Volume": "10",
            "Extra": "x",
        },
        {
            "date": "2024-01-01 09:16:00+05:30",
            "OPEN": 102,
            "HIGH": 103,
            "LOW": 101,
            "CLOSE": 102,
            "VOLUME": 15,
        },
    ]
    df = _safe_dataframe(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2024-01-01 09:15:00")
    assert df.index[1] == pd.Timestamp("2024-01-01 09:16:00")
