from datetime import datetime
import math
from zoneinfo import ZoneInfo

from hypothesis import assume, given, settings, strategies as st, HealthCheck
from unittest.mock import patch

import pytest

from src.config import InstrumentConfig, settings as app_settings
from src.risk.limits import Exposure, LimitConfig, RiskEngine


def _basic_args():
    return dict(
        equity_rupees=10_000_000.0,
        plan={},
        exposure=Exposure(),
        intended_symbol="SYM",
        intended_lots=1,
        lot_size=1,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=100.0,
        option_mid_price=100.0,
        quote={"mid": 100.0},
    )


@pytest.fixture(autouse=True)
def _fixed_now(monkeypatch):
    monkeypatch.setattr(RiskEngine, "_now", lambda self: FIXED_NOW)


FIXED_NOW = datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def test_daily_dd_blocks():
    cfg = LimitConfig(max_daily_dd_R=1.0)
    eng = RiskEngine(cfg)
    eng.state.session_date = FIXED_NOW.date().isoformat()
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
    cfg = LimitConfig(max_notional_rupees=1000.0, exposure_basis="underlying")
    eng = RiskEngine(cfg)
    exp = Exposure(notional_rupees=900.0)
    ok, reason, _ = eng.pre_trade_check(
        **{
            **_basic_args(),
            "exposure": exp,
            "intended_lots": 1,
            "lot_size": 1,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "spot_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    assert not ok and reason == "max_notional"


def test_cap_lt_one_lot():
    cfg = LimitConfig(exposure_basis="premium")
    eng = RiskEngine(cfg)
    args = _basic_args()
    plan = args["plan"]
    args.update(
        {
            "equity_rupees": 0.0,
            "intended_lots": 1,
            "lot_size": 25,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "cap_lt_one_lot"
    assert details["unit_notional"] == 200.0 * 25
    assert details["cap"] <= details["unit_notional"]
    assert plan["reason_block"] == "cap_lt_one_lot"
    assert any(
        r.startswith("cap_lt_one_lot") for r in plan.get("reasons", [])
    )
    assert "cap_abs" in details


def test_equity_based_premium_cap(monkeypatch):
    cfg = LimitConfig(exposure_basis="premium")
    eng = RiskEngine(cfg)
    args = _basic_args()
    plan = args["plan"]
    args.update(
        {
            "equity_rupees": 40_000.0,
            "intended_lots": 1,
            "lot_size": 25,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    ok, reason, _ = eng.pre_trade_check(**args)
    assert ok and reason == ""
    cap_pct = getattr(app_settings, "RISK__EXPOSURE_CAP_PCT", 0.0)
    cap_rupees = cap_pct * args["equity_rupees"]
    unit_notional = args["option_mid_price"] * args["lot_size"]
    expected_lots = max(1, math.floor(cap_rupees / unit_notional))
    assert plan.get("qty_lots") == expected_lots


def test_allow_min_one_lot_override(monkeypatch):
    cfg = LimitConfig(exposure_basis="premium")
    eng = RiskEngine(cfg)
    monkeypatch.setattr(app_settings, "RISK__EXPOSURE_CAP_PCT", 0.20, raising=False)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.20, raising=False)
    monkeypatch.setattr(app_settings.risk, "exposure_cap_pct_of_equity", 0.20, raising=False)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_ABS", 0.0, raising=False)
    monkeypatch.setattr(app_settings.risk, "allow_min_one_lot", True, raising=False)
    args = _basic_args()
    plan = args["plan"]
    args.update(
        {
            "equity_rupees": 40_000.0,
            "intended_lots": 3,
            "lot_size": 50,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    ok, reason, details = eng.pre_trade_check(**args)
    assert ok and reason == ""
    assert plan.get("qty_lots") == 1
    assert details.get("allow_min_one_lot") is True


def test_allow_min_one_lot_disabled_blocks(monkeypatch):
    cfg = LimitConfig(exposure_basis="premium")
    eng = RiskEngine(cfg)
    monkeypatch.setattr(app_settings, "RISK__EXPOSURE_CAP_PCT", 0.20, raising=False)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.20, raising=False)
    monkeypatch.setattr(app_settings.risk, "exposure_cap_pct_of_equity", 0.20, raising=False)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_ABS", 0.0, raising=False)
    monkeypatch.setattr(app_settings.risk, "allow_min_one_lot", False, raising=False)
    args = _basic_args()
    plan = args["plan"]
    args.update(
        {
            "equity_rupees": 40_000.0,
            "intended_lots": 2,
            "lot_size": 50,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "cap_lt_one_lot"
    assert plan["reason_block"] == "cap_lt_one_lot"
    assert "unit_notional" in details


def test_equity_cap_limits_aggregate_exposure(monkeypatch):
    cfg = LimitConfig(max_notional_rupees=1_000_000.0, exposure_basis="premium")
    eng = RiskEngine(cfg)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    monkeypatch.setattr(app_settings, "RISK__EXPOSURE_CAP_PCT", 0.20, raising=False)
    monkeypatch.setattr(app_settings, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.20, raising=False)
    monkeypatch.setattr(app_settings.risk, "exposure_cap_pct_of_equity", 0.20, raising=False)
    exposure = Exposure(notional_rupees=9_000.0)
    args = _basic_args()
    plan = args["plan"]
    args.update(
        {
            "equity_rupees": 50_000.0,
            "exposure": exposure,
            "intended_lots": 1,
            "lot_size": 50,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "spot_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    ok, reason, details = eng.pre_trade_check(**args)
    assert plan.get("qty_lots") == 1
    assert not ok and reason == "max_notional"
    assert details["cur"] == pytest.approx(exposure.notional_rupees)
    assert details["add"] == pytest.approx(200.0 * 50)


def test_gamma_mode_cap(monkeypatch):
    cfg = LimitConfig(max_gamma_mode_lots=1)
    eng = RiskEngine(cfg)
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=ZoneInfo(cfg.tz))  # Tuesday
    monkeypatch.setattr(eng, "_now", lambda: dt)
    ok, reason, _ = eng.pre_trade_check(
        **{**_basic_args(), "intended_lots": 2}
    )
    assert not ok and reason == "gamma_mode_lot_cap"


def test_notional_underlying_vs_premium():
    """Ensure notional is computed based on exposure basis."""
    spot = 60.0
    premium = 10.0
    exp_under = Exposure(notional_rupees=100.0)
    cfg_under = LimitConfig(max_notional_rupees=150.0, exposure_basis="underlying")
    eng_under = RiskEngine(cfg_under)
    ok, reason, _ = eng_under.pre_trade_check(
        **{
            **_basic_args(),
            "exposure": exp_under,
            "spot_price": spot,
            "option_mid_price": premium,
            "entry_price": premium,
            "quote": {"mid": premium},
        }
    )
    assert not ok and reason == "max_notional"

    exp_prem = Exposure(notional_rupees=100.0)
    cfg_prem = LimitConfig(max_notional_rupees=150.0, exposure_basis="premium")
    eng_prem = RiskEngine(cfg_prem)
    ok2, reason2, _ = eng_prem.pre_trade_check(
        **{
            **_basic_args(),
            "exposure": exp_prem,
            "spot_price": spot,
            "option_mid_price": premium,
            "entry_price": premium,
            "quote": {"mid": premium},
        }
    )
    assert ok2 and reason2 == ""


def test_notional_cap_disabled_when_none():
    cfg = LimitConfig(
        max_notional_rupees=None,
        max_lots_per_symbol=100,
        exposure_basis="underlying",
    )
    eng = RiskEngine(cfg)
    ok, reason, details = eng.pre_trade_check(
        **{
            **_basic_args(),
            "exposure": Exposure(notional_rupees=2_000_000.0),
            "intended_lots": 10,
            "lot_size": 50,
            "entry_price": 200.0,
            "option_mid_price": 200.0,
            "spot_price": 200.0,
            "quote": {"mid": 200.0},
        }
    )
    assert ok and reason == ""


def test_daily_premium_loss_blocks():
    cfg = LimitConfig(max_daily_loss_rupees=100.0)
    eng = RiskEngine(cfg)
    eng.state.session_date = FIXED_NOW.date().isoformat()
    eng.state.cum_loss_rupees = -150.0
    ok, reason, _ = eng.pre_trade_check(**_basic_args())
    assert not ok and reason == "daily_premium_loss"


def test_volatility_loss_cap_blocks():
    cfg = LimitConfig(
        max_daily_loss_rupees=200.0,
        volatility_ref_atr_pct=2.0,
        volatility_loss_min_multiplier=0.3,
        volatility_loss_max_multiplier=1.5,
    )
    eng = RiskEngine(cfg)
    eng.state.session_date = FIXED_NOW.date().isoformat()
    eng.state.cum_loss_rupees = -120.0
    args = _basic_args()
    args["plan"] = {"atr_pct": 5.0}
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "volatility_loss_cap"
    assert details["threshold"] < abs(cfg.max_daily_loss_rupees)


def test_instrument_specific_lot_cap(monkeypatch):
    cfg = LimitConfig(max_lots_per_symbol=6)
    eng = RiskEngine(cfg)
    inst = InstrumentConfig(
        symbol="BANKNIFTY",
        spot_symbol="NSE:BANKNIFTY",
        trade_symbol="BANKNIFTY",
        trade_exchange="NFO",
        instrument_token=999,
        spot_token=999,
        nifty_lot_size=15,
        strike_range=200,
        min_lots=1,
        max_lots=2,
    )
    monkeypatch.setitem(app_settings.instruments.additional, "BANKNIFTY", inst)
    args = _basic_args()
    args.update({"intended_symbol": "BANKNIFTY"})
    args["exposure"] = Exposure(lots_by_symbol={"BANKNIFTY": 2})
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "lot_cap"
    assert details["cap"] == 2


def test_regime_lot_clamp_blocks():
    cfg = LimitConfig(max_lots_per_symbol=4, regime_lot_multipliers={"RANGE": 0.5})
    eng = RiskEngine(cfg)
    exp = Exposure(lots_by_symbol={"SYM": 1})
    args = _basic_args()
    args["plan"] = {"regime": "range"}
    args["exposure"] = exp
    args["intended_lots"] = 2
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "lot_cap"
    assert details["cap"] == 2


def test_regime_delta_cap_blocks():
    cfg = LimitConfig(
        max_portfolio_delta_units=100,
        regime_delta_multipliers={"RANGE": 0.4},
    )
    eng = RiskEngine(cfg)
    args = _basic_args()
    args["plan"] = {"regime": "RANGE"}
    args["portfolio_delta_units"] = 50.0
    args["planned_delta_units"] = 0.0
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "delta_cap"
    assert details["cap"] == 40


def test_var_limit_blocks():
    cfg = LimitConfig(var_lookback_trades=5, max_var_rupees=85.0, var_confidence=0.8)
    eng = RiskEngine(cfg)
    eng.state.session_date = FIXED_NOW.date().isoformat()
    eng.state.pnl_history_rupees = [-30.0, -50.0, -80.0, -120.0, -60.0]
    args = _basic_args()
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "var_limit"
    assert details["threshold"] == pytest.approx(85.0)


def test_cvar_limit_blocks():
    cfg = LimitConfig(
        var_lookback_trades=5,
        max_var_rupees=500.0,
        max_cvar_rupees=75.0,
        cvar_confidence=0.6,
    )
    eng = RiskEngine(cfg)
    eng.state.session_date = FIXED_NOW.date().isoformat()
    eng.state.pnl_history_rupees = [-40.0, -60.0, -90.0, -120.0, -80.0]
    args = _basic_args()
    ok, reason, details = eng.pre_trade_check(**args)
    assert not ok and reason == "cvar_limit"
    assert details["threshold"] == pytest.approx(75.0)


def test_mid_price_from_quote():
    cfg = LimitConfig(max_notional_rupees=150.0, exposure_basis="premium")
    eng = RiskEngine(cfg)
    exp = Exposure(notional_rupees=145.0)
    quote = {"bid": 9.0, "ask": 11.0}
    ok, reason, _ = eng.pre_trade_check(
        **{
            **_basic_args(),
            "equity_rupees": 375.0,
            "exposure": exp,
            "quote": quote,
            "option_mid_price": None,
            "entry_price": 0.0,
            "spot_price": 0.0,
        }
    )
    assert not ok and reason == "max_notional"


@given(
    current_notional=st.floats(min_value=0, max_value=1_000_000),
    entry=st.floats(min_value=50, max_value=500),
    lot_size=st.integers(min_value=1, max_value=200),
    lots=st.integers(min_value=1, max_value=5),
    max_notional=st.floats(min_value=50_000, max_value=2_000_000),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_pre_trade_notional_ok(current_notional, entry, lot_size, lots, max_notional):
    cfg = LimitConfig(
        max_notional_rupees=max_notional,
        max_lots_per_symbol=100,
        exposure_basis="underlying",
    )
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
                "option_mid_price": entry,
                "spot_price": entry,
                "quote": {"mid": entry},
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
    cfg = LimitConfig(
        max_notional_rupees=max_notional,
        max_lots_per_symbol=100,
        exposure_basis="underlying",
    )
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
                "option_mid_price": entry,
                "spot_price": entry,
                "quote": {"mid": entry},
            }
        )
        assert not ok and reason == "max_notional"
