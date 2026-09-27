from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.execution.position_manager import PositionManager


def test_position_registry_state_patch_is_retired() -> None:
    assert not Path(
        "src/nifty_scalper_bot/execution/position_registry_state.py"
    ).exists()
    assert PositionManager._canonical_registry_state_native is True
    assert PositionManager.save_state.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
    assert PositionManager.load_state.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )


def test_execution_package_does_not_apply_registry_patch() -> None:
    source = Path("src/nifty_scalper_bot/execution/__init__.py").read_text(
        encoding="utf-8"
    )

    assert "position_registry_state" not in source
