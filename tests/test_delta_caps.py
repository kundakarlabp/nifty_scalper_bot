from datetime import datetime
from zoneinfo import ZoneInfo

from src.risk.limits import LimitConfig, RiskEngine, Exposure


def _engine(cfg: LimitConfig) -> RiskEngine:
    eng = RiskEngine(cfg)
    eng._now = lambda: datetime(2024, 1, 1, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata"))  # type: ignore[method-assign]
    return eng


def test_portfolio_delta_cap():
    cfg = LimitConfig()
    eng = _engine(cfg)
    exposure = Exposure()
    ok, reason, _ = eng.pre_trade_check(
        equity_rupees=1_000_000.0,
        plan={},
        exposure=exposure,
        intended_symbol="NIFTY",
        intended_lots=1,
        lot_size=50,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=100.0,
        option_mid_price=100.0,
        quote={"mid": 100.0},
        portfolio_delta_units=101.0,
    )
    assert not ok and reason == "delta_cap"


def test_delta_cap_on_add():
    cfg = LimitConfig()
    eng = _engine(cfg)
    exposure = Exposure()
    ok, reason, _ = eng.pre_trade_check(
        equity_rupees=1_000_000.0,
        plan={},
        exposure=exposure,
        intended_symbol="NIFTY",
        intended_lots=1,
        lot_size=50,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=100.0,
        option_mid_price=100.0,
        quote={"mid": 100.0},
        planned_delta_units=20.0,
        portfolio_delta_units=90.0,
    )
    assert not ok and reason == "delta_cap_on_add"
