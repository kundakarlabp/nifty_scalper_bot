from types import SimpleNamespace

from risk import risk_gates


def test_loss_streak_and_rr_low_block() -> None:
    cfg = SimpleNamespace(risk=SimpleNamespace(max_consec_losses=1), rr_threshold=2.0)
    acct = risk_gates.AccountState(
        equity_rupees=100_000,
        dd_rupees=0.0,
        max_daily_loss=1_000_000.0,
        loss_streak=1,
    )
    plan = {"rr": 1.0, "risk_rupees": 100.0}
    ok, reasons = risk_gates.evaluate(plan, acct, cfg)
    assert not ok
    assert "loss_streak" in reasons
    assert "rr_low" in reasons
