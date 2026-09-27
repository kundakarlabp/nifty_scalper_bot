from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.execution.position_manager import PositionManager


def test_broker_order_ledger_patch_is_retired() -> None:
    assert not Path(
        "src/nifty_scalper_bot/execution/broker_order_ledger_patch.py"
    ).exists()
    assert PositionManager._broker_order_ledger_native is True
    assert PositionManager.apply_broker_order_update.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert PositionManager.reconcile_broker_orders.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )


def test_execution_package_does_not_apply_broker_order_ledger_patch() -> None:
    source = Path("src/nifty_scalper_bot/execution/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "broker_order_ledger_patch" not in source
