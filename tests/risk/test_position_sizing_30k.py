from nifty_scalper_bot.risk.position_sizing import PositionSizer


def test_30k_conservative():
    r = PositionSizer().size(equity=30000, risk_per_trade_pct=0.005, entry=120, stop_loss=112, lot_size=25, confidence=0.8, regime_multiplier=1.0, max_lots=3, available_margin=50000, margin_per_lot=15000)
    assert not r.allowed
    assert r.qty == 0
    assert r.reason == "insufficient_risk_budget_for_one_lot"


def test_invalid_sl():
    assert not PositionSizer().size(equity=30000, risk_per_trade_pct=0.005, entry=100, stop_loss=101, lot_size=25, confidence=0.8, regime_multiplier=1.0, max_lots=1, available_margin=50000, margin_per_lot=15000).allowed


def test_margin_reject():
    assert not PositionSizer().size(equity=30000, risk_per_trade_pct=0.005, entry=120, stop_loss=112, lot_size=25, confidence=0.8, regime_multiplier=1.0, max_lots=1, available_margin=1000, margin_per_lot=15000).allowed
