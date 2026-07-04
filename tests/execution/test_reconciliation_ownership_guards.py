from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution import bracket_ownership_extension as bracket_owner
from nifty_scalper_bot.execution import position_identity_extension as position_owner


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

    assert unresolved == {"NFO:NIFTY2670724250PE"}
    assert prepared[0]["symbol"] == "NFO:NIFTY2670724250PE"
    assert prepared[0]["tradingsymbol"] == "NFO:NIFTY2670724250PE"
    assert prepared[0]["average_price"] == 0


def test_broker_sync_uses_existing_local_entry_when_average_price_missing():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY2670724250PE": SimpleNamespace(entry_price=75.0),
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


def test_existing_nonterminal_bracket_owns_canonical_symbol():
    existing = SimpleNamespace(
        entry_order_id="ENTRY-1",
        symbol="NFO:NIFTY2670724250PE",
        remaining_quantity=65,
        exit_executed=False,
        exit_state="OPEN_ACTIVE",
    )
    manager = SimpleNamespace(_brackets={"ENTRY-1": existing})

    assert bracket_owner._existing_owner(manager, "NIFTY2670724250PE", "ENTRY-2") is existing
    assert bracket_owner._existing_owner(manager, "NFO:NIFTY2670724250PE", "ENTRY-1") is None
