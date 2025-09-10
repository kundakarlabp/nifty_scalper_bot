from src.risk.risk_gates import AccountState, evaluate


class Cfg:
    rr_threshold = 2.0
    risk = type("R", (), {"max_consec_losses": 2})


def test_risk_gates_reasons() -> None:
    plan = {"rr": 1.0}
    acct = AccountState(
        equity_rupees=0.0,
        dd_rupees=1000.0,
        max_daily_loss=1000.0,
        loss_streak=3,
    )
    ok, reasons = evaluate(plan, acct, Cfg())
    assert not ok
    assert "loss_streak" in reasons
    assert "rr_low" in reasons
    assert "daily_dd" in reasons
