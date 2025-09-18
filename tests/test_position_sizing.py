# tests/test_position_sizing.py

"""Unit tests for :mod:`src.risk.position_sizing`."""

from types import SimpleNamespace

import pytest
from src.risk.position_sizing import (
    PositionSizer,
    _mid_from_quote,
    lots_from_premium_cap,
)
from src.config import settings as cfg
from hypothesis import given, settings, strategies as st, assume


@pytest.fixture(autouse=True)
def _reset_risk_per_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure module-level defaults use the canonical risk-per-trade."""

    monkeypatch.setattr(cfg.risk, "risk_per_trade", 0.01, raising=False)


def _sizer(
    risk_per_trade: float | None = None,
    min_lots: int = 1,
    max_lots: int = 10,
    max_position_size_pct: float = 1.0,
    exposure_basis: str = "premium",
) -> PositionSizer:
    """Helper to create a ``PositionSizer`` with sensible defaults."""

    kwargs = dict(
        min_lots=min_lots,
        max_lots=max_lots,
        max_position_size_pct=max_position_size_pct,
        exposure_basis=exposure_basis,
    )
    if risk_per_trade is not None:
        kwargs["risk_per_trade"] = risk_per_trade
    return PositionSizer(**kwargs)


def test_mid_from_quote_variants():
    """_mid_from_quote should derive price from multiple quote shapes."""
    assert _mid_from_quote({"mid": 100}) == 100.0
    assert _mid_from_quote({"bid": 90, "ask": 110}) == 100.0
    assert _mid_from_quote({"ltp": 123.45}) == 123.45
    assert _mid_from_quote({}) == 0.0


def test_lots_from_premium_cap(monkeypatch):
    """lots_from_premium_cap respects equity-based caps."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.10, raising=False)
    monkeypatch.setattr(
        cfg.risk, "exposure_cap_pct_of_equity", 0.10, raising=False
    )
    runner = SimpleNamespace(equity_amount=10_000)
    lots, unit_notional, cap, eq_source = lots_from_premium_cap(
        runner, {"mid": 100}, 25, 10
    )
    assert unit_notional == 2_500
    assert cap == 1_000
    assert lots == 0
    assert eq_source == "live"


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
    assert diag["block_reason"] == "cap_lt_one_lot"
    assert diag["min_equity_needed"] > diag["equity"]
    assert diag["basis"] == "premium"
    assert diag["lots"] == diag["lots_final"]
    assert diag["cap"] == diag["exposure_cap"]
    assert diag["cap_abs"] is None


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
    qty, lots, diag = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=195.0,
        lot_size=50,
        equity=100_000.0,
        spot_sl_points=5.0,
        delta=0.5,
    )
    assert qty == 50
    assert lots == 1
    assert diag["block_reason"] == ""


def test_cap_abs_in_diag(monkeypatch):
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 5_000.0, raising=False)
    monkeypatch.setattr(
        cfg.risk, "exposure_cap_pct_of_equity", 0.40, raising=False
    )
    monkeypatch.setattr(cfg.risk, "exposure_cap_abs", 5_000.0, raising=False)
    sizer = _sizer()
    qty, lots, diag = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=50,
        equity=200_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 50
    assert lots == 1
    assert diag["block_reason"] == ""
    assert diag["cap_abs"] == 5_000.0


def test_risk_per_trade_uses_settings(monkeypatch):
    """The default sizing configuration should mirror the live settings object."""
    monkeypatch.setattr(cfg.risk, "risk_per_trade", 0.0125, raising=False)
    sizer = PositionSizer()
    assert sizer.params.risk_per_trade == 0.0125


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


def test_allow_min_one_lot_when_cap_small(monkeypatch):
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.01, raising=False)
    monkeypatch.setattr(cfg.risk, "exposure_cap_pct_of_equity", 0.01, raising=False)
    monkeypatch.setattr(cfg.risk, "allow_min_one_lot", True, raising=False)
    sizer = _sizer()
    qty, lots, diag = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=75,
        equity=100_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 75
    assert lots == 1
    assert diag["block_reason"] == ""


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

