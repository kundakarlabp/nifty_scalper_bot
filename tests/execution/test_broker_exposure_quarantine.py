from __future__ import annotations

from pathlib import Path

import nifty_scalper_bot.execution  # noqa: F401 - applies runtime safety patches
from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY24JAN100CE"


def _missing_basis_row(quantity: int = 65) -> dict[str, object]:
    return {
        "tradingsymbol": "NIFTY24JAN100CE",
        "quantity": quantity,
        "average_price": 0,
        "last_price": 88.55,
        "product": "MIS",
    }


def test_unresolved_broker_position_is_quarantined(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))

    manager.synchronize_with_broker([_missing_basis_row()])

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


def test_cost_basis_quarantine_survives_restart_without_managed_position_change(tmp_path):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(str(state_file))

    manager.synchronize_with_broker([_missing_basis_row()])

    restored = PositionManager(str(state_file))
    exposures = restored.get_quarantined_broker_exposures()
    assert exposures[SYMBOL]["reason"] == "cost_basis_unresolved"
    assert SYMBOL in restored._cost_basis_unresolved_symbols
    assert restored.get_position(SYMBOL) is None


def test_resolved_cost_basis_removes_persisted_quarantine_even_when_account_flat(tmp_path):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(str(state_file))
    manager.synchronize_with_broker([_missing_basis_row()])
    manager.save_state()

    manager.synchronize_with_broker([])

    assert manager.get_quarantined_broker_exposures() == {}
    restored = PositionManager(str(state_file))
    assert restored.get_quarantined_broker_exposures() == {}
    assert SYMBOL not in restored._cost_basis_unresolved_symbols


def test_position_sync_preserves_non_cost_basis_external_quarantine(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))
    manager._quarantined_broker_exposures = {
        SYMBOL: {
            "symbol": SYMBOL,
            "tradingsymbol": SYMBOL,
            "reason": "broker_state_unverified",
            "status": "BROKER_STATE_UNVERIFIED",
            "source": "broker_order_ledger",
            "order_id": "external-1",
        }
    }

    manager.synchronize_with_broker([_missing_basis_row()])

    exposure = manager.get_quarantined_broker_exposures()[SYMBOL]
    assert exposure["reason"] == "broker_state_unverified"
    assert exposure["order_id"] == "external-1"
    assert exposure["source"] == "broker_order_ledger"


def test_broker_order_ledger_patch_no_longer_owns_position_state_persistence() -> None:
    source = Path(
        "src/nifty_scalper_bot/execution/broker_order_ledger_patch.py"
    ).read_text(encoding="utf-8")

    assert "_persist_extra_state" not in source
    assert "cls.__init__ = __init__" not in source
    assert "cls.load_state = load_state" not in source
    assert "cls.save_state = save_state" not in source
    assert "def _read_state(" not in source
    assert "def _write_state(" not in source
