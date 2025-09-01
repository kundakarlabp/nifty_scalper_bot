from types import SimpleNamespace

from risk import risk_gates


def test_rr_threshold_none_does_not_block() -> None:
    cfg = SimpleNamespace(risk=SimpleNamespace(per_trade_pct_max=1.0, max_consec_losses=3), rr_threshold=None)
    acct = risk_gates.AccountState(equity_rupees=100_000, dd_rupees=0.0, max_daily_loss=1_000_000.0, loss_streak=0)
    plan = {"rr": 0.5, "risk_rupees": 100.0}
    ok, reasons = risk_gates.evaluate(plan, acct, cfg)
    assert ok
    assert "rr_low" not in reasons
