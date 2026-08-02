from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


def _manager(monkeypatch, *, cost_points: float = 1.0) -> RuntimeBracketManager:
    monkeypatch.setenv("TRAIL_MIN_LOCKED_PROFIT_R", "0.10")
    manager = object.__new__(RuntimeBracketManager)
    manager._trail_tier1_pct = 1.0
    manager._trail_tier2_pct = 2.0
    manager._trail_tier3_pct = 4.0
    manager._trail_tier4_pct = 6.0
    manager._trail_tier2_r = 1.0
    manager._trail_tier3_r = 2.0
    manager._trail_tier4_r = 3.0
    manager._calculate_momentum = lambda _symbol: 0.0
    manager._breakeven_cost_per_unit = lambda _bracket: cost_points
    manager._trail_activation_r = lambda _bracket: 0.75
    manager._min_locked_profit_r = lambda: 0.10
    return manager


def _trail(manager, bracket, *, ltp, profit_pct, high_water, atr=0.0):
    """Canonical trail decision: tier ladder, then the net-profit floor."""
    candidate = manager._calculate_tiered_trailing_sl(
        bracket=bracket, ltp=ltp, profit_pct=profit_pct,
        high_water=high_water, atr=atr,
    )
    if candidate is None:
        return None
    return manager._apply_min_profit_floor(bracket, candidate, ltp)


def _bracket(*, side: str, entry: float, sl: float, high: float, low: float):
    return SimpleNamespace(
        symbol="NFO:NIFTYTEST",
        side=side,
        entry_price=entry,
        sl_trigger_price=sl,
        initial_sl_trigger_price=sl,
        highest_ltp=high,
        lowest_ltp=low,
        trailing_config={},
        quantity=65,
    )


def test_buy_tier1_locks_cost_plus_point_one_r(monkeypatch) -> None:
    manager = _manager(monkeypatch, cost_points=1.0)
    bracket = _bracket(side="BUY", entry=100.0, sl=90.0, high=108.0, low=100.0)

    candidate = _trail(
        manager, bracket, ltp=108.0, profit_pct=8.0,
        high_water=108.0, atr=0.0,
    )

    assert candidate == pytest.approx(102.0)


def test_sell_tier1_locks_cost_plus_point_one_r(monkeypatch) -> None:
    manager = _manager(monkeypatch, cost_points=1.0)
    bracket = _bracket(side="SELL", entry=100.0, sl=110.0, high=100.0, low=92.0)

    candidate = _trail(
        manager, bracket, ltp=92.0, profit_pct=8.0,
        high_water=92.0, atr=0.0,
    )

    assert candidate == pytest.approx(98.0)


def test_tier4_uses_r_metric_not_premium_percentage(monkeypatch) -> None:
    manager = _manager(monkeypatch, cost_points=0.0)
    bracket = _bracket(
        side="BUY",
        entry=1000.0,
        sl=995.0,
        high=1015.0,
        low=1000.0,
    )

    candidate = _trail(
        manager, bracket, ltp=1015.0, profit_pct=1.5,
        high_water=1015.0, atr=0.0,
    )

    assert candidate == pytest.approx(1009.0)


def test_floor_is_not_installed_before_activation(monkeypatch) -> None:
    manager = _manager(monkeypatch, cost_points=1.0)
    bracket = _bracket(side="BUY", entry=100.0, sl=90.0, high=107.0, low=100.0)

    candidate = _trail(
        manager, bracket, ltp=107.0, profit_pct=7.0,
        high_water=107.0, atr=0.0,
    )

    assert candidate is None


def test_floor_is_rejected_when_execution_room_is_insufficient(monkeypatch) -> None:
    manager = _manager(monkeypatch, cost_points=4.0)
    bracket = _bracket(side="BUY", entry=100.0, sl=90.0, high=108.0, low=100.0)

    candidate = _trail(
        manager, bracket, ltp=108.0, profit_pct=8.0,
        high_water=108.0, atr=0.0,
    )

    assert candidate is None
