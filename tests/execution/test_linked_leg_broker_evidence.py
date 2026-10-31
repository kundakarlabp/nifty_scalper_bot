"""Linked order evidence comes from broker reports, not local acknowledgements."""

import json

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager

_PARENT = "research-parent-001"
_CALL = "NFO:NIFTY26NOV24000CE"
_PUT = "NFO:NIFTY26NOV24000PE"


def manager_with_pair(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    for order_id, symbol in (("call-1", _CALL), ("put-1", _PUT)):
        manager.add_pending_order(
            order_id=order_id,
            symbol=symbol,
            side="BUY",
            qty=65,
            price=75.0,
            order_type="LIMIT",
            intent="ENTRY",
            bracket_id=_PARENT,
        )
    return manager


def broker_order(order_id, symbol, *, status="OPEN", filled=0):
    return {
        "order_id": order_id,
        "tradingsymbol": symbol.split(":", 1)[1],
        "transaction_type": "BUY",
        "quantity": 65,
        "filled_quantity": filled,
        "average_price": 75.0 if filled else 0.0,
        "status": status,
        "product": "MIS",
    }


def snapshot(manager):
    return manager.linked_leg_fill_snapshot(_PARENT, ["call-1", "put-1"])


def test_local_registration_never_counts_as_broker_fill(tmp_path):
    manager = manager_with_pair(tmp_path)
    view = snapshot(manager)
    assert view["all_legs_broker_filled"] is False
    assert view["reason"] == "linked_leg_broker_fill_unverified"
    assert view["live_entry_allowed"] is False
    assert view["portfolio_protection_certified"] is False
    assert (
        manager.get_broker_order_ledger()["call-1"].get("broker_fill_evidence", False)
        is False
    )


def test_explicit_open_and_partial_broker_updates_remain_unconfirmed(tmp_path):
    manager = manager_with_pair(tmp_path)
    manager.apply_broker_order_update(
        "call-1", broker_order("call-1", _CALL, status="OPEN")
    )
    manager.apply_broker_order_update(
        "put-1", broker_order("put-1", _PUT, status="OPEN")
    )
    assert snapshot(manager)["reason"] == "linked_leg_not_fully_filled"
    manager.apply_broker_order_update(
        "call-1", broker_order("call-1", _CALL, status="PARTIALLY FILLED", filled=15)
    )
    view = snapshot(manager)
    assert view["all_legs_broker_filled"] is False
    assert view["legs"]["call-1"]["broker_filled_quantity"] == 15


def test_both_broker_filled_are_observed_not_authorized(tmp_path):
    manager = manager_with_pair(tmp_path)
    for order_id, symbol in (("call-1", _CALL), ("put-1", _PUT)):
        manager.apply_broker_order_update(
            order_id, broker_order(order_id, symbol, status="COMPLETE", filled=65)
        )
    view = snapshot(manager)
    assert view["all_legs_broker_filled"] is True
    assert view["reason"] == "linked_leg_broker_fills_observed"
    assert view["live_entry_allowed"] is False
    assert view["portfolio_protection_certified"] is False

    state = json.loads((tmp_path / "positions.json").read_text(encoding="utf-8"))
    assert state["broker_order_ledger"]["call-1"]["broker_fill_evidence"] is True
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    assert snapshot(restored)["all_legs_broker_filled"] is True


def test_missing_fill_field_cannot_be_promoted_from_requested_quantity(tmp_path):
    manager = manager_with_pair(tmp_path)
    payload = broker_order("call-1", _CALL, status="COMPLETE", filled=65)
    del payload["filled_quantity"]
    manager.apply_broker_order_update("call-1", payload)
    assert snapshot(manager)["all_legs_broker_filled"] is False
    assert snapshot(manager)["legs"]["call-1"]["reason"] == (
        "linked_leg_broker_fill_unverified"
    )


def test_mismatched_broker_identity_cannot_confirm_leg(tmp_path):
    manager = manager_with_pair(tmp_path)
    payload = broker_order("call-1", _CALL, status="COMPLETE", filled=65)
    payload["quantity"] = 130
    manager.apply_broker_order_update("call-1", payload)
    assert snapshot(manager)["all_legs_broker_filled"] is False
    assert snapshot(manager)["legs"]["call-1"]["reason"] == (
        "linked_leg_broker_fill_unverified"
    )


def test_restored_inflight_order_remains_unconfirmed(tmp_path):
    manager = manager_with_pair(tmp_path)
    manager.apply_broker_order_update(
        "call-1", broker_order("call-1", _CALL, status="OPEN")
    )
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    assert snapshot(restored)["all_legs_broker_filled"] is False


@pytest.mark.parametrize(
    ("parent", "order_ids"),
    [
        ("", ["call-1", "put-1"]),
        (_PARENT, ["call-1"]),
        (_PARENT, ["call-1", "call-1"]),
        (_PARENT, "call-1,put-1"),
    ],
)
def test_invalid_parent_or_leg_identity_rejected(tmp_path, parent, order_ids):
    manager = manager_with_pair(tmp_path)
    with pytest.raises(ValueError, match="linked_leg_identity_invalid"):
        manager.linked_leg_fill_snapshot(parent, order_ids)


def test_wrong_parent_fails_closed(tmp_path):
    manager = manager_with_pair(tmp_path)
    view = manager.linked_leg_fill_snapshot("other-parent", ["call-1", "put-1"])
    assert view["all_legs_broker_filled"] is False
    assert view["reason"] == "linked_leg_local_identity_missing"
