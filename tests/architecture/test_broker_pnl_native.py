from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.execution.position_manager import PositionManager


def test_broker_pnl_runtime_patches_are_retired() -> None:
    for path in (
        "src/nifty_scalper_bot/execution/broker_pnl_authority_patch.py",
        "src/nifty_scalper_bot/execution/pnl_session_rollover_patch.py",
    ):
        assert not Path(path).exists()


def test_broker_pnl_owners_are_native() -> None:
    assert PositionManager._broker_pnl_native is True
    assert PositionManager._pnl_session_rollover_native is True
    assert PositionManager.refresh_broker_pnl_diagnostic.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert PositionManager.get_broker_account_realized_pnl.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert ZerodhaKiteClient.get_pnl_snapshot.__module__ == (
        "nifty_scalper_bot.data.rest.zerodha_client"
    )


def test_execution_package_does_not_apply_pnl_patches() -> None:
    source = Path("src/nifty_scalper_bot/execution/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "broker_pnl_authority_patch" not in source
    assert "pnl_session_rollover_patch" not in source
