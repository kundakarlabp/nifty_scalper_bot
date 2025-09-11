# tests/test_position_sizing.py

"""Unit tests for :mod:`src.risk.position_sizing`."""

from src.risk.position_sizing import PositionSizer
from hypothesis import given, settings, strategies as st, assume


def _sizer(
    risk_per_trade: float = 0.01,
    min_lots: int = 1,
    max_lots: int = 10,
    max_position_size_pct: float = 1.0,
    exposure_basis: str = "premium",
) -> PositionSizer:
    """Helper to create a ``PositionSizer`` with sensible defaults."""

    return PositionSizer(
        risk_per_trade=risk_per_trade,
        min_lots=min_lots,
        max_lots=max_lots,
        max_position_size_pct=max_position_size_pct,
        exposure_basis=exposure_basis,
    )


def test_basic_sizing():
    sizer = _sizer()
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=50,
        equity=100_000.0,
        spot_sl_points=40.0,
        delta=0.5,
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
        spot_sl_points=5.0,
        delta=0.5,
    )
    assert qty == 125
    assert lots == 5


def test_returns_zero_when_budget_insufficient():
    """Even with a higher min_lots, zero should be returned when one lot is unaffordable."""
    sizer = _sizer(min_lots=3)
    qty, lots, diag = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=25,
        equity=4_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 0
    assert lots == 0
    assert diag["block_reason"] == "exposure_cap"


def test_min_lots_only_enforced_when_affordable():
    sizer = _sizer(min_lots=3)
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=50,
        equity=150_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 150
    assert lots == 3


def test_min_lots_rescue_when_affordable():
    sizer = _sizer(risk_per_trade=1e-6)
    qty, lots, _ = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=195.0,
        lot_size=50,
        equity=100_000.0,
        spot_sl_points=5.0,
        delta=0.5,
    )
    assert qty == 50
    assert lots == 1


def test_underlying_basis_caps_by_spot():
    sizer = _sizer(max_position_size_pct=0.05, exposure_basis="underlying")
    qty, lots, diag = sizer.size_from_signal(
        entry_price=10.0,
        stop_loss=5.0,
        lot_size=50,
        equity=100_000.0,
        spot_price=100.0,
        spot_sl_points=5.0,
        delta=0.5,
    )
    assert qty == 50
    assert lots == 1
    assert diag["unit_notional"] * lots == 100.0 * 50 * 1


@given(
    entry=st.floats(min_value=50.0, max_value=500.0),
    stop=st.floats(min_value=1.0, max_value=500.0),
    lot_size=st.integers(min_value=1, max_value=200),
    equity=st.floats(min_value=1_000.0, max_value=1_000_000.0),
    risk=st.floats(min_value=0.001, max_value=0.05),
    min_lots=st.integers(min_value=1, max_value=5),
    max_lots=st.integers(min_value=5, max_value=20),
    max_pos=st.floats(min_value=0.05, max_value=0.5),
)
@settings(max_examples=25, deadline=None)
def test_position_sizer_properties(entry, stop, lot_size, equity, risk, min_lots, max_lots, max_pos):
    assume(max_lots >= min_lots)
    sizer = PositionSizer(
        risk_per_trade=risk,
        min_lots=min_lots,
        max_lots=max_lots,
        max_position_size_pct=max_pos,
    )
    qty, lots, _ = sizer.size_from_signal(
        entry_price=entry,
        stop_loss=stop,
        lot_size=lot_size,
        equity=equity,
        spot_sl_points=abs(entry - stop),
        delta=0.5,
    )
    assert lots >= 0
    assert lots <= max_lots
    assert qty == lots * lot_size

