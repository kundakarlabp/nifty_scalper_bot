from __future__ import annotations
# ruff: noqa
import pandas as pd
"""
Tests for the PositionSizing module.

These tests ensure that the position sizing logic returns sensible lot
sizes based on the configured risk and correctly enforces daily
drawdown limits.
"""

from __future__ import annotations

from src.risk.position_sizing import PositionSizing


def test_basic_sizing() -> None:
    # Account of ₹100k, 1 % risk per trade, lot size 50
    ps = PositionSizing(
        account_size=100_000,
        risk_per_trade=0.01,
        daily_risk=0.05,
        max_drawdown=0.05,
        lot_size=50,
        min_lots=1,
        max_lots=10,
    )
    # Entry at 100, stop at 90 (10 points risk).  Risk per lot = 500
    result = ps.calculate_position_size(100, 90, signal_confidence=10)
    assert result is not None
    assert result["quantity"] == 2, "With ₹1k risk budget and ₹500 per lot, quantity should be 2"


def test_confidence_scaling() -> None:
    ps = PositionSizing(
        account_size=100_000,
        risk_per_trade=0.01,
        daily_risk=0.05,
        max_drawdown=0.05,
        lot_size=50,
        min_lots=1,
        max_lots=10,
    )
    # Lower confidence should reduce quantity
    high_conf = ps.calculate_position_size(100, 90, signal_confidence=10)
    low_conf = ps.calculate_position_size(100, 90, signal_confidence=5)
    assert low_conf is not None and high_conf is not None
    assert low_conf["quantity"] < high_conf["quantity"], "Higher confidence should produce larger size"


def test_daily_risk_limit() -> None:
    ps = PositionSizing(
        account_size=100_000,
        risk_per_trade=0.01,
        daily_risk=0.05,
        max_drawdown=0.05,
        lot_size=50,
        min_lots=1,
        max_lots=10,
    )
    # Simulate losses approaching the daily risk limit (₹5k)
    ps.daily_loss = 4_800  # Already down ₹4.8k
    # Each new trade would risk 2 lots * 10 pts * 50 = ₹1000
    result = ps.calculate_position_size(100, 90, signal_confidence=10)
    # Should not take the trade because 4.8k + 1k > 5k
    assert result is None, "Daily risk limit should prevent new positions"