from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


SYMBOL = "NFO:NIFTY2691523250PE"


def test_authoritative_empty_broker_snapshot_establishes_required_zero_baseline(
    tmp_path,
) -> None:
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


def test_explicit_broker_realized_initializes_zero_local_baseline(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline()

    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "average_price": 101.0,
                "last_price": 101.0,
                "realised": -240.0,
            }
        ]
    )

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] == -240.0
    assert snapshot["baseline_source"] == "validated_broker_positions"
    assert manager.current_pnl_reconciliation_blocker() is None
    assert manager.get_realized_pnl() == 0.0


def test_empty_snapshot_preserves_unverified_nonzero_local_pnl_as_diagnostic(
    tmp_path,
) -> None:
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


def test_pnl_diagnostics_do_not_block_live_readiness() -> None:
    for diagnostic in (
        "pnl_baseline_uninitialized",
        "pnl_session_date_unverified",
        "pnl_reconciliation_mismatch",
    ):
        decision = normalize_readiness_blockers(
            [diagnostic],
            "OPEN",
            broker_state={"broker_balance_valid": True},
            live_mode=True,
            evaluation_ready=True,
            execution_ready=True,
        )

        assert decision.primary_blocker is None
        assert decision.blocker_list == []
        assert decision.live_orders_armed is True
        assert decision.execution_ready is True


def test_canonical_app_pnl_diagnostics_are_filtered_before_arming() -> None:
    app_source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    patch_source = Path(
        "src/nifty_scalper_bot/execution/pnl_nonblocking_patch.py"
    ).read_text(encoding="utf-8")

    assert "current_pnl_reconciliation_blocker" in app_source
    assert "missing.append(str(pnl_blocker))" in app_source
    assert "_normalize_readiness_without_pnl_blocking" in patch_source
    assert '"pnl_baseline_uninitialized"' in patch_source
    assert '"pnl_session_date_unverified"' in patch_source
    assert '"pnl_reconciliation_mismatch"' in patch_source


def test_canonical_app_requires_pnl_baseline_before_broker_hydration() -> None:
    source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    manager_index = source.index("position_manager = PositionManager")
    baseline_index = source.index(
        "position_manager.require_pnl_session_baseline()", manager_index
    )
    hydration_index = source.index("\n    _hydrate_positions(", manager_index)

    assert baseline_index < hydration_index
