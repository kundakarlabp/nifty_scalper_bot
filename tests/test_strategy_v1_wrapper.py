import pandas as pd

from src.strategies.v1 import v1


def _dummy_df(rows: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="1T")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1_000] * rows,
        },
        index=idx,
    )


def test_v1_wrapper_returns_plan_dict() -> None:
    df = _dummy_df()
    plan = v1(df=df, current_price=100.0, spot_df=df)
    assert isinstance(plan, dict)
