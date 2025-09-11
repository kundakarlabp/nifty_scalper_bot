from datetime import datetime
from zoneinfo import ZoneInfo

from src.risk.limits import Exposure, LimitConfig, RiskEngine


def _args():
    return dict(
        equity_rupees=0.0,
        plan={},
        exposure=Exposure(),
        intended_symbol="SYM",
        intended_lots=1,
        lot_size=1,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=100.0,
        option_mid_price=100.0,
    )


def test_skip_next_open(monkeypatch):
    cfg = LimitConfig(max_daily_dd_R=1.0, skip_next_open_after_two_daily_caps=True)
    eng = RiskEngine(cfg)

    dt1 = datetime(2025, 1, 1, 10, 0, tzinfo=ZoneInfo(cfg.tz))
    monkeypatch.setattr(eng, "_now", lambda: dt1)
    eng.on_trade_closed(pnl_R=-1.5)
    eng.pre_trade_check(**_args())

    dt2 = datetime(2025, 1, 2, 10, 0, tzinfo=ZoneInfo(cfg.tz))
    monkeypatch.setattr(eng, "_now", lambda: dt2)
    eng.on_trade_closed(pnl_R=-1.5)
    ok, reason, _ = eng.pre_trade_check(**_args())
    assert not ok and reason == "daily_dd_hit"
    assert eng.state.skip_next_open_date == "2025-01-03"

    dt3 = datetime(2025, 1, 3, 10, 0, tzinfo=ZoneInfo(cfg.tz))
    monkeypatch.setattr(eng, "_now", lambda: dt3)
    ok, reason, _ = eng.pre_trade_check(**_args())
    assert not ok and reason == "skip_next_open"
