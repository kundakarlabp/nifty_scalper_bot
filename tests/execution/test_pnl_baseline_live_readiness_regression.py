from __future__ import annotations

from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def test_authoritative_empty_broker_snapshot_establishes_required_zero_baseline(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline()

    assert manager.current_pnl_reconciliation_blocker() == "pnl_baseline_uninitialized"

    manager.synchronize_with_broker([])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] == 0.0
    assert snapshot["pnl_trading_date"] == manager._trading_date_ist()
    assert snapshot["baseline_source"] == "validated_broker_empty_snapshot"
    assert manager.current_pnl_reconciliation_blocker() is None
    assert manager.get_realized_pnl() == 0.0


def test_empty_snapshot_does_not_erase_unverified_nonzero_local_pnl(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline()
    manager._local_realized_pnl = -125.0
    with manager._lock:
        manager._refresh_realized_pnl_locked()

    manager.synchronize_with_broker([])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] is None
    assert manager.get_realized_pnl() == -125.0
    assert manager.current_pnl_reconciliation_blocker() == "pnl_baseline_uninitialized"


def test_pnl_entry_gate_blockers_fail_closed_in_live_readiness() -> None:
    for blocker in (
        "pnl_baseline_uninitialized",
        "pnl_session_date_unverified",
        "pnl_reconciliation_mismatch",
    ):
        decision = normalize_readiness_blockers(
            [blocker],
            "OPEN",
            broker_state={"broker_balance_valid": True},
            live_mode=True,
            evaluation_ready=True,
            execution_ready=True,
        )

        assert decision.primary_blocker == blocker
        assert decision.live_orders_armed is False
        assert decision.execution_ready is False
