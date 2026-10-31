"""Linked child order attempts remain durable across ambiguous broker results."""

import json

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager

PARENT = "condor-20261031"
SYMBOLS = {
    "long_put": ("NFO:NIFTY26NOV23700PE", "BUY", 23700, 101),
    "long_call": ("NFO:NIFTY26NOV24300CE", "BUY", 24300, 102),
    "short_put": ("NFO:NIFTY26NOV23800PE", "SELL", 23800, 103),
    "short_call": ("NFO:NIFTY26NOV24200CE", "SELL", 24200, 104),
}


class Broker:
    def get_positions(self):
        return {"net": []}

    def get_orders(self):
        return []


def create(tmp_path):
    state_path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_path))
    manager.set_broker_client(Broker())
    legs = [
        {
            "role": role,
            "symbol": symbol,
            "side": side,
            "strike": strike,
            "instrument_token": token,
            "lot_size": 65,
            "quantity": 65,
            "expiry": "2026-11-26",
        }
        for role, (symbol, side, strike, token) in SYMBOLS.items()
    ]
    manager.reserve_linked_portfolio(PARENT, "IRON_CONDOR", legs)
    return manager, state_path


def test_write_ahead_attempt_survives_restart_and_blocks_reorder(tmp_path):
    manager, path = create(tmp_path)
    attempt = manager.reserve_linked_child_attempt(PARENT, "long_put", "lp20261031a")
    assert attempt["state"] == "UNRESOLVED"
    assert attempt["broker_order_id"] is None
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (
        saved["linked_portfolio_reservations"][PARENT]["child_attempts"]["long_put"][
            "correlation_tag"
        ]
        == "lp20261031a"
    )
    restored = PositionManager(state_file=str(path))
    restored.set_broker_client(Broker())
    view = restored.linked_portfolio_recovery_plan(PARENT)
    assert view["next_roles"] == []
    assert view["reason"] == "linked_child_submission_ambiguous"
    with pytest.raises(ValueError, match="linked_child_wing_first_gate_blocked"):
        restored.reserve_linked_child_attempt(PARENT, "long_put", "lp20261031b")


def test_wrong_first_role_or_invalid_correlation_tag_does_not_write(tmp_path):
    manager, path = create(tmp_path)
    with pytest.raises(ValueError, match="linked_child_wing_first_gate_blocked"):
        manager.reserve_linked_child_attempt(PARENT, "short_put", "sp20261031a")
    with pytest.raises(ValueError, match="linked_child_correlation_invalid"):
        manager.reserve_linked_child_attempt(PARENT, "long_put", "not valid tag")
    assert (
        manager.get_linked_portfolio_reservation(PARENT).get("child_attempts") is None
    )


def test_broker_ack_bind_resolves_attempt_but_order_remains_unverified(tmp_path):
    manager, path = create(tmp_path)
    manager.reserve_linked_child_attempt(PARENT, "long_put", "lp20261031a")
    manager.add_pending_order(
        order_id="order-1",
        symbol=SYMBOLS["long_put"][0],
        side="BUY",
        qty=65,
        price=100.0,
        order_type="LIMIT",
        intent="ENTRY",
        bracket_id=PARENT,
    )
    parent = manager.bind_linked_portfolio_order(PARENT, "long_put", "order-1")
    assert parent["child_attempts"]["long_put"]["state"] == "BOUND"
    assert parent["legs"]["long_put"]["order_id"] == "order-1"
    restored = PositionManager(state_file=str(path))
    restored.set_broker_client(Broker())
    assert restored.get_linked_portfolio_reservation(PARENT) == parent
    assert restored.linked_portfolio_recovery_plan(PARENT)["next_roles"] == []


def test_abort_is_persistent_and_not_cleared_by_parent_reuse(tmp_path):
    manager, path = create(tmp_path)
    held = manager.abort_linked_parent(PARENT, "partial_wing_cancel")
    assert held["state"] == "ABORT_RECOVERY_REQUIRED"
    restored = PositionManager(state_file=str(path))
    restored.set_broker_client(Broker())
    view = restored.linked_portfolio_recovery_plan(PARENT)
    assert view["next_roles"] == []
    assert view["reason"] == "linked_parent_abort_hold"
    with pytest.raises(ValueError, match="linked_child_wing_first_gate_blocked"):
        restored.reserve_linked_child_attempt(PARENT, "long_put", "lp20261031a")


def test_write_failure_is_never_claimed_durable(tmp_path, monkeypatch):
    manager, path = create(tmp_path)
    monkeypatch.setattr(manager, "save_state", lambda: None)
    with pytest.raises(
        RuntimeError, match="linked_child_attempt_persistence_unverified"
    ):
        manager.reserve_linked_child_attempt(PARENT, "long_put", "lp20261031a")
