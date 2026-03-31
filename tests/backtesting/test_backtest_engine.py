from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest
from src.nifty_scalper_bot.backtesting import backtest_engine as be
from src.nifty_scalper_bot.backtesting.backtest_engine import BacktestEngine


@pytest.mark.unit
def test_backtest_engine_generates_trades(backtest_engine: BacktestEngine) -> None:
    result = backtest_engine.run()
    assert result.trades, "Expected at least one completed trade"
    assert all(trade.pnl is not None for trade in result.trades)


@pytest.mark.unit
def test_performance_metrics_are_computed(backtest_engine: BacktestEngine) -> None:
    result = backtest_engine.run()
    metrics = result.performance
    expected_keys = {
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "volatility",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade_duration_days",
    }
    assert expected_keys.issubset(metrics), metrics
    assert metrics["volatility"] >= 0


@pytest.mark.unit
def test_equity_curve_alignment(
    backtest_engine: BacktestEngine, sample_price_data: pd.DataFrame
) -> None:
    result = backtest_engine.run()
    assert len(result.equity_curve) == len(sample_price_data)
    assert (
        pytest.approx(result.equity_curve["equity"].iloc[0])
        == backtest_engine.config.initial_cash
    )


@pytest.mark.unit
def test_discover_history_files_includes_cab_directory(tmp_path: Path) -> None:
    cab_dir = tmp_path / "cab"
    cab_dir.mkdir()
    sample = cab_dir / "NIFTY_20240101.csv"
    sample.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T09:15:00+05:30,100,101,99,100.5,1000\n",
        encoding="utf-8",
    )
    files = be._discover_history_files("NIFTY", tmp_path)
    assert sample in files


@pytest.mark.unit
def test_load_market_data_filters_symbol_and_localises_timezone(tmp_path: Path) -> None:
    csv_path = tmp_path / "nifty_options_all.csv"
    timestamps = pd.date_range(
        end=pd.Timestamp.now(tz="Asia/Kolkata"), periods=20, freq="5min"
    )
    symbols = ["NIFTY" if i % 2 == 0 else "BANKNIFTY" for i in range(len(timestamps))]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "open": 200 + pd.Series(range(len(timestamps))) * 0.1,
            "high": 200.5 + pd.Series(range(len(timestamps))) * 0.1,
            "low": 199.5 + pd.Series(range(len(timestamps))) * 0.1,
            "close": 200.2 + pd.Series(range(len(timestamps))) * 0.1,
            "volume": 1000,
        }
    )
    expected_rows = int((frame["symbol"].str.upper() == "NIFTY").sum())
    frame.to_csv(csv_path, index=False)
    loaded = be._load_market_data("NIFTY", 2, tmp_path)
    assert not loaded.empty
    assert len(loaded) == expected_rows
    assert loaded.index.tz == timezone.utc
    assert "symbol" not in loaded.columns
    assert "close" in loaded.columns
