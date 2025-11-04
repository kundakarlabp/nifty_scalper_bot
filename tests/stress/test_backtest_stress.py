from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.nifty_scalper_bot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
)
from tests.conftest import DeterministicStrategy


@pytest.mark.stress
def test_backtest_handles_large_dataset(tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=123)
    index = pd.date_range("2023-01-01", periods=10_000, freq="5T")
    base = 100 + np.cumsum(rng.normal(scale=0.2, size=len(index)))
    data = pd.DataFrame(
        {
            "open": base,
            "high": base + rng.normal(scale=0.1, size=len(index)),
            "low": base - rng.normal(scale=0.1, size=len(index)),
            "close": base,
            "volume": rng.integers(500, 2_000, size=len(index)),
        },
        index=index,
    )
    output_dir = tmp_path / "stress_reports"
    output_dir.mkdir()
    engine = BacktestEngine(
        data,
        DeterministicStrategy(),
        BacktestConfig(output_directory=output_dir, fixed_quantity=5),
    )
    result = engine.run()
    assert result.equity_curve["equity"].notna().all()
    assert len(result.orders) > 0
