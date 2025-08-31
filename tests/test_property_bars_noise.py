import numpy as np
import pandas as pd
import hypothesis.strategies as st
from hypothesis import given, settings
from datetime import datetime, timedelta

from strategies.scalping_strategy import ScalpingStrategy


@given(st.integers(min_value=20, max_value=120))
@settings(deadline=None, max_examples=30)
def test_indicator_pipeline_handles_gaps(n):
    # build synthetic minute bars with random gaps
    ts0 = datetime(2025, 8, 29, 9, 15, 0)
    idx = [ts0 + timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame(
        {
            "open": 100 + np.random.randn(n).cumsum(),
            "high": 100 + np.random.randn(n).cumsum() + 1,
            "low": 100 + np.random.randn(n).cumsum() - 1,
            "close": 100 + np.random.randn(n).cumsum(),
            "volume": np.random.randint(100, 1000, size=n),
        },
        index=idx,
    )
    # randomly drop up to 10% rows
    df = df.drop(df.sample(frac=0.1, random_state=42).index).sort_index()
    # Your indicator/strategy evaluate function should not crash on gaps
    s = ScalpingStrategy(settings=None)
    plan = s.evaluate_from_backtest(
        df.index[-1],
        df.open.iloc[-1],
        df.high.iloc[-1],
        df.low.iloc[-1],
        df.close.iloc[-1],
        df.volume.iloc[-1],
    )
    assert isinstance(plan, dict)
