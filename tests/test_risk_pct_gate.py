from types import SimpleNamespace

from risk import risk_gates


def test_daily_dd_gate_blocks() -> None:
    cfg = SimpleNamespace(risk=SimpleNamespace(max_consec_losses=3))
    acct = risk_gates.AccountState(
        equity_rupees=100_000,
        dd_rupees=1_000_000.0,
        max_daily_loss=1_000_000.0,
        loss_streak=0,
    )
    plan = {}
    ok, reasons = risk_gates.evaluate(plan, acct, cfg)
    assert not ok
    assert "daily_dd" in reasons
