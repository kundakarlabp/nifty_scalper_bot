# tests/test_position_sizing.py

"""Unit tests for :mod:`src.risk.position_sizing`."""

from src.risk.position_sizing import PositionSizer


def _sizer(
    risk_per_trade: float = 0.01,
    min_lots: int = 1,
    max_lots: int = 10,
    max_position_size_pct: float = 1.0,
) -> PositionSizer:
    """Helper to create a ``PositionSizer`` with sensible defaults."""

    return PositionSizer(
        risk_per_trade=risk_per_trade,
        min_lots=min_lots,
        max_lots=max_lots,
        max_position_size_pct=max_position_size_pct,
    )


def test_basic_sizing():
    sizer = _sizer()
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=50,
        equity=100_000.0,
    )
    assert qty == 50
    assert lots == 1


def test_max_lots_clamp():
    sizer = _sizer(max_lots=5)
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=195.0,
        lot_size=25,
        equity=100_000.0,
    )
    assert qty == 125
    assert lots == 5


def test_returns_zero_when_budget_insufficient():
    """Even with a higher min_lots, zero should be returned when one lot is unaffordable."""
    sizer = _sizer(min_lots=3)
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=25,
        equity=10_000.0,
    )
    assert qty == 0
    assert lots == 0


def test_min_lots_only_enforced_when_affordable():
    sizer = _sizer(min_lots=3)
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=50,
        equity=200_000.0,
    )
    assert qty == 150
    assert lots == 3

