from __future__ import annotations

from nifty_scalper_bot.backtest.execution_sim import ExecutionSimulator


def test_wider_spread_results_in_worse_fill() -> None:
    sim = ExecutionSimulator(aggressiveness=0.6)
    narrow = sim.simulate_market_order(
        side="BUY",
        qty=50,
        bid=100.0,
        ask=101.0,
        size_at_best=60,
        ts_ns=1,
    )
    wide = sim.simulate_market_order(
        side="BUY",
        qty=50,
        bid=100.0,
        ask=103.0,
        size_at_best=60,
        ts_ns=2,
    )
    assert wide.average_price > narrow.average_price


def test_partial_fill_when_depth_insufficient() -> None:
    sim = ExecutionSimulator(aggressiveness=0.6)
    result = sim.simulate_market_order(
        side="BUY",
        qty=100,
        bid=100.0,
        ask=101.0,
        size_at_best=40,
        ts_ns=1,
    )
    assert result.remaining_qty == 60
    assert result.fills[0].qty == 40


def test_commissions_reduce_net_result() -> None:
    sim = ExecutionSimulator(aggressiveness=0.6)
    result = sim.simulate_market_order(
        side="BUY",
        qty=50,
        bid=100.0,
        ask=101.0,
        size_at_best=50,
        ts_ns=1,
    )
    gross_turnover = sum(fill.price * fill.qty for fill in result.fills)
    assert result.fees > 0
    assert gross_turnover - result.fees < gross_turnover
