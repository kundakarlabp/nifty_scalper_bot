from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.execution.position_manager import Position


def test_position_identity_is_native_not_import_time_overlay() -> None:
    execution_init = Path("src/nifty_scalper_bot/execution/__init__.py").read_text(
        encoding="utf-8"
    )
    manager_source = Path(
        "src/nifty_scalper_bot/execution/position_manager.py"
    ).read_text(encoding="utf-8")
    live_safety = Path(
        "src/nifty_scalper_bot/execution/live_safety_identity.py"
    ).read_text(encoding="utf-8")

    assert "position_identity_extension" not in execution_init
    assert "_apply_live_safety_identity_patches" not in execution_init
    assert "def _patch_position_manager" not in live_safety
    assert "_canonical_position_identity_native = True" in manager_source
    assert "_canonicalize_payload_symbol(dict(broker_payload))" in manager_source
    assert "BROKER_FLAT_CONFIRMED_FOR_UNKNOWN_ORDER" in manager_source
    assert "ENTRY_LIFECYCLE_BASIS_RESTORED" in manager_source


def test_position_bot_ownership_property_is_native() -> None:
    assert Position.strategy_name.fget is not None
    assert Position.strategy_name.fget.__module__ == (
        "nifty_scalper_bot.execution.position_manager"
    )
