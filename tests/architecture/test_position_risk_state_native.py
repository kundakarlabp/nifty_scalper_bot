from __future__ import annotations

import inspect
from pathlib import Path

from nifty_scalper_bot.execution.position_manager import PositionManager


def test_position_risk_state_is_native_not_package_patch() -> None:
    init_text = Path("src/nifty_scalper_bot/execution/__init__.py").read_text(
        encoding="utf-8"
    )

    assert "position_risk_state_patch" not in init_text
    assert not hasattr(PositionManager, "_position_risk_state_patch")
    assert PositionManager.get_risk_circuit_state.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert PositionManager.persist_risk_circuit_state.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert "_risk_runtime" in inspect.getsource(PositionManager.save_state)
    assert "_restore_risk_state" in inspect.getsource(PositionManager.load_state)


def test_position_manager_broker_sync_owns_risk_pnl_reconciliation() -> None:
    source = Path(
        "src/nifty_scalper_bot/execution/position_manager.py"
    ).read_text(encoding="utf-8")

    sync_start = source.index("    def synchronize_with_broker(")
    sync_end = source.index(
        "    def _synchronize_managed_positions_from_broker(",
        sync_start,
    )
    native_sync = source[sync_start:sync_end]
    assert "_maybe_seed_pnl_session_baseline" in native_sync
    assert "_reconcile_local_pnl_to_broker_snapshot" in native_sync
