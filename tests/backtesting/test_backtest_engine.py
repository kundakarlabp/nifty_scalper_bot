from __future__ import annotations

import json
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


@pytest.mark.parametrize(
    ("prices", "signals", "slippage", "expected_return", "expected_drawdown"),
    [
        ([100.0], [1], 0.0, -0.001, -0.001),
        ([100.0], [1], 0.01, -0.00201, -0.00201),
        ([100.0, 100.0, 100.0], [1, 0, 0], 0.0, -0.002, -0.002),
        ([100.0, 120.0, 120.0], [1, 0, 0], 0.0, 0.0178, -0.001),
        ([100.0, 100.0], [0, 0], 0.0, 0.0, 0.0),
    ],
)
def test_backtest_includes_first_entry_cost_in_returns_and_drawdown(
    tmp_path: Path, prices, signals, slippage, expected_return, expected_drawdown
) -> None:
    data = pd.DataFrame(
        {"close": prices},
        index=pd.date_range(
            "2026-09-24T09:30:00+05:30", periods=len(prices), freq="min"
        ),
    )

    class _Strategy:
        name = "initial_cost_regression"

        def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
            return pd.Series(signals, index=market_data.index)

    result = BacktestEngine(
        data,
        _Strategy(),
        be.BacktestConfig(
            initial_cash=10_000.0,
            fixed_quantity=10,
            commission_pct=0.01,
            slippage_pct=slippage,
            risk_free_rate=0.0,
            output_directory=tmp_path,
            generate_visualizations=False,
        ),
    ).run()

    assert result.performance["total_return"] == pytest.approx(expected_return)
    assert result.performance["max_drawdown"] == pytest.approx(expected_drawdown)
    assert (1.0 + result.equity_curve["returns"]).prod() - 1.0 == pytest.approx(
        expected_return
    )
    assert len(result.equity_curve) == len(data)
    exported = json.loads(result.json_path.read_text())
    assert exported["pnl"]["total"] == pytest.approx(expected_return * 10_000.0)
    assert sum(exported["pnl"]["daily"].values()) == pytest.approx(
        exported["pnl"]["total"]
    )


def test_daily_pnl_includes_entry_cost_and_overnight_changes(
    tmp_path: Path, deterministic_strategy: be.StrategyProtocol
) -> None:
    data = pd.DataFrame(
        {"close": [100.0, 110.0, 120.0, 115.0]},
        index=pd.to_datetime(
            [
                "2026-09-24T09:30:00+05:30",
                "2026-09-24T10:00:00+05:30",
                "2026-09-25T09:30:00+05:30",
                "2026-09-28T09:30:00+05:30",
            ]
        ),
    )
    result = BacktestEngine(
        data,
        deterministic_strategy,
        be.BacktestConfig(
            initial_cash=10_000.0,
            fixed_quantity=10,
            commission_pct=0.01,
            slippage_pct=0.0,
            allow_short=False,
            output_directory=tmp_path,
            generate_visualizations=False,
        ),
    ).run()

    exported = json.loads(result.json_path.read_text())
    assert exported["pnl"]["daily"] == pytest.approx(
        {"2026-09-24": -11.0, "2026-09-25": 100.0, "2026-09-28": -61.5}
    )
    assert exported["pnl"]["total"] == pytest.approx(27.5)
    assert sum(exported["pnl"]["daily"].values()) == pytest.approx(27.5)


def test_backtest_rejects_missing_timestamps_before_strategy_evaluation() -> None:
    class _Strategy:
        name = "must_not_run"

        def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
            raise AssertionError("invalid data reached strategy")

    data = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex([pd.NaT]))
    with pytest.raises(ValueError, match="missing timestamps"):
        BacktestEngine(data, _Strategy())


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
