from __future__ import annotations

import time

import pytest
from src.nifty_scalper_bot.backtesting.backtest_engine import BacktestEngine


@pytest.mark.performance
def test_backtest_completes_within_time(backtest_engine: BacktestEngine) -> None:
    start = time.perf_counter()
    backtest_engine.run()
    duration = time.perf_counter() - start
    assert duration < 2.0, f"Backtest took too long: {duration}"  # sanity budget for CI
