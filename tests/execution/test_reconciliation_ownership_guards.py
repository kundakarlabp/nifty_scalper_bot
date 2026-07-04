from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution import bracket_ownership_extension as bracket_owner
from nifty_scalper_bot.execution import position_identity_extension as position_owner
from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY2670724250PE"


def test_broker_sync_missing_average_price_is_unresolved_not_ltp_cost_basis():
    manager = SimpleNamespace(_positions={})
    prepared, unresolved = position_owner._prepare_broker_positions(
        manager,
        [
            {
                "tradingsymbol": "NIFTY2670724250PE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ],
    )

    assert unresolved == {SYMBOL}
    assert prepared[0]["symbol"] == SYMBOL
    assert prepared[0]["tradingsymbol"] == SYMBOL
    assert prepared[0]["average_price"] == 0


def test_broker_sync_uses_existing_local_entry_when_average_price_missing():
    manager = SimpleNamespace(
        _positions={
            SYMBOL: SimpleNamespace(entry_price=75.0),
        }
    )
    prepared, unresolved = position_owner._prepare_broker_positions(
        manager,
        [
            {
                "tradingsymbol": "NIFTY2670724250PE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ],
    )

    assert unresolved == set()
    assert prepared[0]["average_price"] == 75.0


def test_unresolved_broker_position_is_visible_as_quarantined_exposure(tmp_path):
    manager = PositionManager(str(tmp_path / "positions.json"))

    manager.synchronize_with_broker(
        [
            {
                "tradingsymbol": "NIFTY2670724250PE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ]
    )

    exposures = manager.get_quarantined_broker_exposures()
    assert list(exposures) == [SYMBOL]
    exposure = exposures[SYMBOL]
    assert exposure["status"] == "BROKER_POSITION_QUARANTINED"
    assert exposure["reason"] == "cost_basis_unresolved"
    assert exposure["symbol"] == SYMBOL
    assert exposure["quantity"] == 65
    assert exposure["signed_quantity"] == 65
    assert exposure["managed_position"] is False
    assert exposure["pnl_accounting_allowed"] is False
    assert exposure["requires_history_recovery"] is True
    assert manager.current_entry_protection_blocker(SYMBOL) == "cost_basis_unresolved"
    assert manager.get_position(SYMBOL) is None


def test_existing_nonterminal_bracket_owns_canonical_symbol():
    existing = SimpleNamespace(
        entry_order_id="ENTRY-1",
        symbol=SYMBOL,
        remaining_quantity=65,
        exit_executed=False,
        exit_state="OPEN_ACTIVE",
    )
    manager = SimpleNamespace(_brackets={"ENTRY-1": existing})

    assert bracket_owner._existing_owner(manager, "NIFTY2670724250PE", "ENTRY-2") is existing
    assert bracket_owner._existing_owner(manager, SYMBOL, "ENTRY-1") is None
