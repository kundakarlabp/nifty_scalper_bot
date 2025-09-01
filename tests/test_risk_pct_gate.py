from types import SimpleNamespace

from risk import risk_gates


def test_risk_pct_blocks_when_risk_exceeds_pct() -> None:
    cfg = SimpleNamespace(risk=SimpleNamespace(per_trade_pct_max=0.5, max_consec_losses=3))
    acct = risk_gates.AccountState(equity_rupees=100_000, dd_rupees=0.0, max_daily_loss=1_000_000.0, loss_streak=0)
    plan = {"risk_rupees": 600.0}
    ok, reasons = risk_gates.evaluate(plan, acct, cfg)
    assert not ok
    assert "per_trade_pct" in reasons
    assert plan["per_trade_allowed_rupees"] == 500.0
    qty_lots = int(plan["per_trade_allowed_rupees"] // 600)
    assert qty_lots == 0
