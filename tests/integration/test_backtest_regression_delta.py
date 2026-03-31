from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nifty_scalper_bot.backtesting.backtest_engine import BacktestConfig, BacktestEngine


@dataclass
class _StaticStrategy:
    name: str = "Static"

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=market_data.index)
        signals.iloc[0:3] = 1
        signals.iloc[3:6] = -1
        return signals


def _sample_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 09:15", periods=10, freq="T")
    return pd.DataFrame({"close": [100 + i for i in range(10)]}, index=index)


def test_backtest_regression_delta(tmp_path) -> None:
    data = _sample_data()
    config = BacktestConfig(output_directory=tmp_path)
    baseline_engine = BacktestEngine(data, _StaticStrategy(), config)
    baseline_result = baseline_engine.run()
    comparison_engine = BacktestEngine(data, _StaticStrategy(), config)
    comparison_result = comparison_engine.run()
    baseline_equity = baseline_result.equity_curve["equity"].iloc[-1]
    comparison_equity = comparison_result.equity_curve["equity"].iloc[-1]
    delta = abs(comparison_equity - baseline_equity) / max(baseline_equity, 1.0)
    assert delta <= 0.01
    assert baseline_result.report_path and baseline_result.report_path.exists()
    assert baseline_result.markdown_path and baseline_result.markdown_path.exists()
