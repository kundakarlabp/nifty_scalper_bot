from src.risk.limits import Exposure, LimitConfig, RiskEngine


def test_posttrade_updates_and_cooloff():
    cfg = LimitConfig(max_consec_losses=3, cooloff_minutes=60)
    eng = RiskEngine(cfg)
    eng.on_trade_closed(pnl_R=1.0)
    assert eng.state.trades_today == 1
    assert eng.state.cum_R_today == 1.0

    eng.on_trade_closed(pnl_R=-0.5)
    eng.on_trade_closed(pnl_R=-0.5)
    eng.on_trade_closed(pnl_R=-0.5)
    ok, reason, _ = eng.pre_trade_check(
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
        quote={"mid": 100.0},
    )
    assert not ok and reason == "loss_cooloff"
    assert eng.state.cooloff_until is not None


def test_roll10_down(monkeypatch):
    cfg = LimitConfig(roll10_pause_R=-0.2, roll10_pause_minutes=60, max_daily_dd_R=100.0)
    eng = RiskEngine(cfg)
    for _ in range(10):
        eng.on_trade_closed(pnl_R=-0.3)
    ok, reason, det = eng.pre_trade_check(
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
        quote={"mid": 100.0},
    )
    assert not ok and reason == "roll10_down"
    assert "avg10R" in det
