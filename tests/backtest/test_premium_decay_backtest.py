"""Backtests for premium decay strategy helper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.backtest.premium_decay_backtest import (
    BacktestBar,
    run_premium_decay_backtest,
)
from nifty_scalper_bot.options.strike_selector import SelectedContract


def _contract(symbol: str, option_type: str, delta: float) -> SelectedContract:
    """Return a synthetic contract for backtest scenarios."""

    return SelectedContract(
        symbol=symbol,
        option_type=option_type,
        strike=20000.0,
        expiry=datetime.now(timezone.utc),
        ltp=50.0,
        delta=delta,
        metadata={"lot_size": 50},
    )


def test_run_premium_decay_backtest_simple() -> None:
    """Backtest should open and close at least one position on benign data."""

    start = datetime(2024, 5, 1, 9, 45, tzinfo=timezone.utc)
    ce = _contract("SIMCE", "CE", 0.24)
    pe = _contract("SIMPE", "PE", -0.23)
    bars = [
        BacktestBar(
            timestamp=start,
            indicators={"atr": 12.0, "adx": 10.0},
            option_chain=[ce, pe],
            option_prices={"SIMCE": 50.0, "SIMPE": 50.0},
            iv=0.22,
        ),
        BacktestBar(
            timestamp=start + timedelta(minutes=5),
            indicators={"atr": 11.5, "adx": 9.5},
            option_chain=[ce, pe],
            option_prices={"SIMCE": 20.0, "SIMPE": 18.0},
            iv=0.21,
        ),
    ]
    result = run_premium_decay_backtest(bars)
    assert result["trades"] >= 1
    assert result["pnl"] >= 0.0
