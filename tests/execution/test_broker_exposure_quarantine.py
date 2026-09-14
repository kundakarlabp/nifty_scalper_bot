from __future__ import annotations

import json

import nifty_scalper_bot.execution  # noqa: F401 - applies runtime safety patches
from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY24JAN100CE"


def _unresolved_position() -> dict[str, object]:
    return {
        "tradingsymbol": "NIFTY24JAN100CE",
        "quantity": 65,
        "average_price": 0,
        "last_price": 88.55,
        "product": "MIS",
    }


def test_unresolved_broker_position_is_quarantined(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))

    manager.synchronize_with_broker([_unresolved_position()])

    exposures = manager.get_quarantined_broker_exposures()
    assert list(exposures) == [SYMBOL]
    exposure = exposures[SYMBOL]
    assert exposure["status"] == "BROKER_POSITION_QUARANTINED"
    assert exposure["reason"] == "cost_basis_unresolved"
    assert exposure["quantity"] == 65
    assert exposure["managed_position"] is False
    assert exposure["requires_history_recovery"] is True
    assert manager.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"
    assert manager.get_position(SYMBOL) is None


def test_cost_basis_quarantine_persists_when_managed_positions_remain_empty(tmp_path):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(str(state_file))

    manager.synchronize_with_broker([_unresolved_position()])

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["quarantined_broker_exposures"][SYMBOL]["reason"] == "cost_basis_unresolved"
    assert payload["cost_basis_unresolved_symbols"] == [SYMBOL]

    restored = PositionManager(str(state_file))
    exposure = restored.get_quarantined_broker_exposures()[SYMBOL]
    assert exposure["reason"] == "cost_basis_unresolved"
    assert restored.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"
    assert SYMBOL in restored._cost_basis_unresolved_symbols


def test_cost_basis_sync_preserves_stronger_external_quarantine(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))
    manager._quarantined_broker_exposures[SYMBOL] = {
        "symbol": SYMBOL,
        "tradingsymbol": SYMBOL,
        "quantity": 65,
        "status": "BROKER_STATE_UNVERIFIED",
        "reason": "broker_state_unverified",
        "source": "broker_order_ledger",
    }

    manager.synchronize_with_broker([_unresolved_position()])

    exposure = manager.get_quarantined_broker_exposures()[SYMBOL]
    assert exposure["reason"] == "broker_state_unverified"
    assert exposure["source"] == "broker_order_ledger"
    assert manager.current_entry_protection_blocker(SYMBOL) == "broker_state_unverified"


def test_masked_cost_basis_blocker_survives_clear_and_restart(tmp_path):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(str(state_file))
    manager._quarantined_broker_exposures[SYMBOL] = {
        "symbol": SYMBOL,
        "tradingsymbol": SYMBOL,
        "quantity": 65,
        "status": "BROKER_STATE_UNVERIFIED",
        "reason": "broker_state_unverified",
        "source": "broker_order_ledger",
    }

    manager.synchronize_with_broker([_unresolved_position()])
    assert SYMBOL in manager._cost_basis_unresolved_symbols
    assert manager.clear_quarantined_broker_exposure(SYMBOL) is True
    assert manager.get_quarantined_broker_exposures() == {}
    assert manager.current_entry_protection_blocker(SYMBOL) == "cost_basis_unresolved"

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["cost_basis_unresolved_symbols"] == [SYMBOL]

    restored = PositionManager(str(state_file))
    assert restored.get_quarantined_broker_exposures() == {}
    assert SYMBOL in restored._cost_basis_unresolved_symbols
    assert restored.current_entry_protection_blocker(SYMBOL) == "cost_basis_unresolved"


def test_cost_basis_sync_removes_only_stale_cost_basis_rows(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))
    other = "NFO:NIFTY24JAN200CE"
    manager._quarantined_broker_exposures = {
        SYMBOL: {
            "symbol": SYMBOL,
            "reason": "cost_basis_unresolved",
            "source": "broker_position_sync",
        },
        other: {
            "symbol": other,
            "reason": "broker_state_unverified",
            "source": "broker_order_ledger",
        },
    }

    manager.synchronize_with_broker([])

    exposures = manager.get_quarantined_broker_exposures()
    assert SYMBOL not in exposures
    assert exposures[other]["reason"] == "broker_state_unverified"


def test_registry_state_owner_serializes_after_runtime_patches() -> None:
    owner = "nifty_scalper_bot.execution.position_registry_state"
    assert PositionManager.save_state.__module__ == owner
    assert PositionManager.load_state.__module__ == owner
    assert PositionManager.synchronize_with_broker.__module__ == owner
    assert PositionManager._canonical_registry_state_owner is True
