from datetime import datetime
from zoneinfo import ZoneInfo

from src.risk.limits import Exposure, LimitConfig, RiskEngine


def _args():
    return {
        "equity_rupees": 1_000_000.0,
        "plan": {},
        "exposure": Exposure(),
        "intended_symbol": "SYM",
        "intended_lots": 1,
        "lot_size": 1,
        "entry_price": 100.0,
        "stop_loss_price": 95.0,
        "spot_price": 100.0,
        "quote": {"mid": 100.0},
        "option_mid_price": 100.0,
    }


def test_no_new_after_blocks(monkeypatch):
    cfg = LimitConfig(no_new_after_hhmm="10:00")
    eng = RiskEngine(cfg)
    dt = datetime(2024, 1, 1, 10, 1, tzinfo=ZoneInfo(cfg.tz))
    monkeypatch.setattr(eng, "_now", lambda: dt)
    ok, reason, _ = eng.pre_trade_check(**_args())
    assert not ok and reason == "session_closed"
