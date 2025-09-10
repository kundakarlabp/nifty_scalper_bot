from datetime import datetime
from zoneinfo import ZoneInfo

from hypothesis import assume, given, settings, strategies as st, HealthCheck
from unittest.mock import patch
from src.risk.limits import Exposure, LimitConfig, RiskEngine


def _basic_args():
    return dict(
        equity_rupees=0.0,
        plan={},
        exposure=Exposure(),
        intended_symbol="SYM",
        intended_lots=1,
        lot_size=1,
        entry_price=100.0,
        stop_loss_price=90.0,
    )


def test_daily_dd_blocks():
    cfg = LimitConfig(max_daily_dd_R=1.0)
    eng = RiskEngine(cfg)
    eng.state.session_date = datetime.now(ZoneInfo(cfg.tz)).date().isoformat()
    eng.state.cum_R_today = -1.2
    ok, reason, _ = eng.pre_trade_check(**_basic_args())
    assert not ok and reason == "daily_dd_hit"


def test_max_lots_symbol():
    cfg = LimitConfig(max_lots_per_symbol=2)
    eng = RiskEngine(cfg)
    exp = Exposure(lots_by_symbol={"SYM": 2})
    ok, reason, _ = eng.pre_trade_check(
        **{**_basic_args(), "exposure": exp}
    )
    assert not ok and reason == "max_lots_symbol"


def test_max_notional():
    cfg = LimitConfig(max_notional_rupees=1000.0)
    eng = RiskEngine(cfg)
    exp = Exposure(notional_rupees=900.0)
    ok, reason, _ = eng.pre_trade_check(
        **{**_basic_args(), "exposure": exp, "intended_lots": 1, "lot_size": 1, "entry_price": 200.0}
    )
    assert not ok and reason == "max_notional"


def test_gamma_mode_cap(monkeypatch):
    cfg = LimitConfig(max_gamma_mode_lots=1)
    eng = RiskEngine(cfg)
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=ZoneInfo(cfg.tz))  # Tuesday
    monkeypatch.setattr(eng, "_now", lambda: dt)
    ok, reason, _ = eng.pre_trade_check(
        **{**_basic_args(), "intended_lots": 2}
    )
    assert not ok and reason == "gamma_mode_lot_cap"


@given(
    current_notional=st.floats(min_value=0, max_value=1_000_000),
    entry=st.floats(min_value=50, max_value=500),
    lot_size=st.integers(min_value=1, max_value=200),
    lots=st.integers(min_value=1, max_value=5),
    max_notional=st.floats(min_value=50_000, max_value=2_000_000),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_pre_trade_notional_ok(current_notional, entry, lot_size, lots, max_notional):
    cfg = LimitConfig(max_notional_rupees=max_notional, max_lots_per_symbol=100)
    eng = RiskEngine(cfg)
    with patch.object(eng, "_now", return_value=datetime(2024, 1, 1, tzinfo=ZoneInfo(cfg.tz))):
        exposure = Exposure(notional_rupees=current_notional)
        intended_notional = entry * lot_size * lots
        assume(current_notional + intended_notional <= max_notional)
        ok, reason, details = eng.pre_trade_check(
            **{
                **_basic_args(),
                "exposure": exposure,
                "intended_lots": lots,
                "lot_size": lot_size,
                "entry_price": entry,
            }
        )
        assert ok and reason == ""
        assert details.get("R_rupees_est") == round(abs(entry - 90.0) * lot_size * lots, 2)


@given(
    current_notional=st.floats(min_value=0, max_value=1_000_000),
    entry=st.floats(min_value=50, max_value=500),
    lot_size=st.integers(min_value=1, max_value=200),
    lots=st.integers(min_value=1, max_value=5),
    max_notional=st.floats(min_value=50_000, max_value=2_000_000),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_pre_trade_notional_block(current_notional, entry, lot_size, lots, max_notional):
    cfg = LimitConfig(max_notional_rupees=max_notional, max_lots_per_symbol=100)
    eng = RiskEngine(cfg)
    with patch.object(eng, "_now", return_value=datetime(2024, 1, 1, tzinfo=ZoneInfo(cfg.tz))):
        exposure = Exposure(notional_rupees=current_notional)
        intended_notional = entry * lot_size * lots
        assume(current_notional + intended_notional > max_notional)
        ok, reason, _ = eng.pre_trade_check(
            **{
                **_basic_args(),
                "exposure": exposure,
                "intended_lots": lots,
                "lot_size": lot_size,
                "entry_price": entry,
            }
        )
        assert not ok and reason == "max_notional"
