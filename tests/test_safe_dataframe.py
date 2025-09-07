import pandas as pd

from src.data.source import _safe_dataframe


def test_safe_dataframe_filters_invalid_rows_and_dedupes():
    rows = [
        {"date": "2023-01-01 09:30:00+05:30", "open": 500, "high": 520, "low": 490, "close": 510, "volume": 50},
        {"date": "2023-01-01 09:15:00+05:30", "Open": "100", "High": "110", "Low": "90", "Close": "105", "Volume": "1000"},
        {"date": "2023-01-01 09:15:00+05:30", "Open": "200", "High": "220", "Low": "190", "Close": "210", "Volume": "2000"},
        {"date": "2023-01-01 09:20:00+05:30", "open": 300, "high": 290, "low": 310, "close": 305, "volume": 500},
        {"date": "2023-01-01 09:25:00+05:30", "open": -10, "high": -5, "low": -20, "close": -15, "volume": 100},
        {"date": "2023-01-01 09:35:00+05:30", "open": 400, "high": 410, "low": 390, "close": float("inf"), "volume": 100},
    ]

    df = _safe_dataframe(rows)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tolist() == [
        pd.Timestamp("2023-01-01 09:15:00"),
        pd.Timestamp("2023-01-01 09:30:00"),
    ]
    first = df.loc[pd.Timestamp("2023-01-01 09:15:00")]
    assert first.to_dict() == {"open": 200.0, "high": 220.0, "low": 190.0, "close": 210.0, "volume": 2000.0}


def test_safe_dataframe_missing_columns_returns_empty():
    assert _safe_dataframe([{"open": 1, "high": 2}]).empty

