from __future__ import annotations

import pytest
from src.nifty_scalper_bot.backtesting.backtest_engine import BacktestEngine


@pytest.mark.integration
def test_backtest_generates_report(backtest_engine: BacktestEngine) -> None:
    result = backtest_engine.run()
    assert result.report_path is not None
    assert result.report_path.exists()
    # Ensure report references chart snippet
    content = result.report_path.read_text(encoding="utf-8")
    assert "Interactive Chart" in content


@pytest.mark.integration
def test_orders_and_trades_are_consistent(backtest_engine: BacktestEngine) -> None:
    result = backtest_engine.run()
    last_order_balance = result.orders[-1].cash_balance
    last_equity = result.equity_curve["equity"].iloc[-1]
    position = result.equity_curve["position"].iloc[-1]
    last_price = backtest_engine.data[backtest_engine.config.price_column].iloc[-1]
    assert (
        pytest.approx(last_order_balance + position * last_price, rel=1e-5)
        == last_equity
    )
