from dataclasses import dataclass

from src.risk.risk_gates import AccountState, evaluate


@dataclass
class _Cfg:
    class risk:
        max_consec_losses = 1

    rr_threshold = 2.0


def test_risk_gates_flags_conditions() -> None:
    acct = AccountState(equity_rupees=0, dd_rupees=5, max_daily_loss=1, loss_streak=2)
    ok, reasons = evaluate({"rr": 1.0}, acct, _Cfg())
    assert not ok
    assert {"daily_dd", "loss_streak", "rr_low"} == set(reasons)
