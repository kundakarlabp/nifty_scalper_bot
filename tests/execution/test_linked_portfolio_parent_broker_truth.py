"""Canonical linked-parent durability and broker net-position reconciliation."""

import json

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager

PARENT = "p45-evidence-20261031"
CALL = "NFO:NIFTY26NOV24000CE"
PUT = "NFO:NIFTY26NOV24000PE"


class ReadOnlyBroker:
    def __init__(self):
        self.positions = {"net": []}
        self.orders = []

    def get_positions(self):
        return self.positions

    def get_orders(self):
        return self.orders


def legs():
    return [
        {
            "role": role,
            "symbol": symbol,
            "side": "BUY",
            "quantity": 65,
            "lot_size": 65,
            "instrument_token": token,
            "strike": 24000,
            "expiry": "2026-11-26",
        }
        for role, symbol, token in (
            ("long_call", CALL, 123456),
            ("long_put", PUT, 123457),
        )
    ]


def broker_row(order_id, symbol, *, status="COMPLETE", filled=65):
    return {
        "order_id": order_id,
        "tradingsymbol": symbol.split(":", 1)[1],
        "transaction_type": "BUY",
        "quantity": 65,
        "filled_quantity": filled,
        "average_price": 100.0,
        "status": status,
        "product": "MIS",
    }


def ready_manager(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    broker = ReadOnlyBroker()
    manager.set_broker_client(broker)
    manager.reserve_linked_portfolio(PARENT, "STRADDLE", legs())
    for role, order_id, symbol in (
        ("long_call", "call-1", CALL),
        ("long_put", "put-1", PUT),
    ):
        manager.add_pending_order(
            order_id=order_id,
            symbol=symbol,
            side="BUY",
            qty=65,
            price=100.0,
            order_type="LIMIT",
            bracket_id=PARENT,
            intent="ENTRY",
        )
        manager.bind_linked_portfolio_order(PARENT, role, order_id)
    return manager, broker


def fill_pair(manager, broker):
    broker.orders = [
        broker_row("call-1", CALL),
        broker_row("put-1", PUT),
    ]
    broker.positions = {
        "net": [
            {"tradingsymbol": CALL.split(":", 1)[1], "quantity": 65},
            {"tradingsymbol": PUT.split(":", 1)[1], "quantity": 65},
        ]
    }
    for row in broker.orders:
        manager.apply_broker_order_update(row["order_id"], row)


def test_write_ahead_parent_persists_and_idempotently_reserves(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    first = manager.reserve_linked_portfolio(PARENT, "STRADDLE", legs())
    assert first["live_entry_allowed"] is False
    assert first["state"] == "RESERVED_UNVERIFIED"
    assert manager.reserve_linked_portfolio(PARENT, "STRADDLE", legs()) == first
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    assert restored.get_linked_portfolio_reservation(PARENT) == first
    with pytest.raises(ValueError, match="linked_parent_symbol_reserved"):
        restored.reserve_linked_portfolio("another", "STRADDLE", legs())


def test_reused_parent_with_changed_leg_rejected(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.reserve_linked_portfolio(PARENT, "STRADDLE", legs())
    changed = legs()
    changed[0]["quantity"] = 130
    changed[1]["quantity"] = 130
    with pytest.raises(ValueError, match="linked_parent_id_conflict"):
        manager.reserve_linked_portfolio(PARENT, "STRADDLE", changed)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x[0].update(side="SELL"),
        lambda x: x[1].update(instrument_token=0),
        lambda x: x[0].update(quantity=60),
        lambda x: x[1].update(symbol=x[0]["symbol"]),
        lambda x: x[0].update(expiry="2026-10-01"),
        lambda x: x[1].update(strike=24050),
    ],
)
def test_invalid_leg_does_not_persist_reservation(tmp_path, mutate):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    given = legs()
    mutate(given)
    with pytest.raises(ValueError):
        manager.reserve_linked_portfolio(PARENT, "STRADDLE", given)
    assert manager.get_linked_portfolio_reservation(PARENT) is None


def test_child_binding_is_idempotent_and_restart_durable(tmp_path):
    manager, broker = ready_manager(tmp_path)
    parent = manager.get_linked_portfolio_reservation(PARENT)
    assert parent["state"] == "IN_PROGRESS_UNVERIFIED"
    assert parent["legs"]["long_call"]["order_id"] == "call-1"
    manager.bind_linked_portfolio_order(PARENT, "long_call", "call-1")
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    restored.set_broker_client(broker)
    assert restored.get_linked_portfolio_reservation(PARENT) == parent
    assert restored.linked_portfolio_broker_state(PARENT)["reason"] == (
        "linked_leg_broker_fill_unverified"
    )


def test_broker_position_snapshot_required_after_verified_fills(tmp_path):
    manager, broker = ready_manager(tmp_path)
    fill_pair(manager, broker)
    view = manager.linked_portfolio_broker_state(PARENT)
    assert view["broker_portfolio_confirmed"] is True
    assert view["reason"] == "linked_parent_broker_position_observed"
    assert view["live_entry_allowed"] is False
    assert view["group_exit_certified"] is False
    assert view["exposure"][CALL]["broker_net"] == 65
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    restored.set_broker_client(broker)
    assert restored.linked_portfolio_broker_state(PARENT)[
        "broker_portfolio_confirmed"
    ] is True


def test_broker_mismatch_and_open_orders_never_confirm(tmp_path):
    manager, broker = ready_manager(tmp_path)
    fill_pair(manager, broker)
    broker.positions = {"net": []}
    assert manager.linked_portfolio_broker_state(PARENT)["reason"] == (
        "linked_parent_broker_exposure_mismatch"
    )
    broker.positions = {
        "net": [
            {"tradingsymbol": CALL.split(":", 1)[1], "quantity": 65},
            {"tradingsymbol": PUT.split(":", 1)[1], "quantity": 65},
        ]
    }
    broker.orders[1]["status"] = "OPEN"
    assert manager.linked_portfolio_broker_state(PARENT)["reason"] == (
        "linked_parent_orderbook_fill_unverified"
    )


def test_parent_can_never_certify_margin_or_group_exit(tmp_path):
    manager, broker = ready_manager(tmp_path)
    fill_pair(manager, broker)
    assert manager.linked_portfolio_broker_state(PARENT)["group_exit_certified"] is False
    serialized = json.loads((tmp_path / "positions.json").read_text())
    assert serialized["linked_portfolio_reservations"][PARENT]["live_entry_allowed"] is False


def test_broker_unavailable_fails_closed_after_recovery(tmp_path):
    manager, broker = ready_manager(tmp_path)
    fill_pair(manager, broker)
    manager.set_broker_client(None)
    assert manager.linked_portfolio_broker_state(PARENT)["reason"] == (
        "linked_parent_broker_fetcher_missing"
    )


def test_parent_persistence_failure_does_not_certify(tmp_path, monkeypatch):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    monkeypatch.setattr(manager, "save_state", lambda: None)
    with pytest.raises(RuntimeError, match="linked_parent_persistence_unverified"):
        manager.reserve_linked_portfolio(PARENT, "STRADDLE", legs())
