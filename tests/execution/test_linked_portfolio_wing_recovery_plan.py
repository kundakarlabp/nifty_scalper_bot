"""Broker-backed recovery advice never places a multi-leg order."""

from nifty_scalper_bot.execution.position_manager import PositionManager

PARENT = "condor-parent-001"
ROLES = (
    ("long_put", 23700, "PE", "BUY", 1),
    ("long_call", 24300, "CE", "BUY", 2),
    ("short_put", 23800, "PE", "SELL", 3),
    ("short_call", 24200, "CE", "SELL", 4),
)


class Broker:
    def __init__(self):
        self.positions = {"net": []}
        self.orders = []
        self.placed = 0
        self.cancelled = 0

    def get_positions(self):
        return self.positions

    def get_orders(self):
        return list(self.orders)

    def place_order(self, **kwargs):
        self.placed += 1
        raise AssertionError("recovery observer must never place")

    def cancel_order(self, *args, **kwargs):
        self.cancelled += 1
        raise AssertionError("recovery observer must never cancel")


def new_parent(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    broker = Broker()
    manager.set_broker_client(broker)
    legs = [
        {
            "role": role,
            "symbol": f"NFO:NIFTY26NOV{strike}{option}",
            "side": side,
            "quantity": 65,
            "lot_size": 65,
            "instrument_token": 9000 + token,
            "strike": strike,
            "expiry": "2026-11-26",
        }
        for role, strike, option, side, token in ROLES
    ]
    parent = manager.reserve_linked_portfolio(PARENT, "IRON_CONDOR", legs)
    return manager, broker, parent


def add_child(manager, broker, parent, role, *, status="COMPLETE", filled=65):
    leg = parent["legs"][role]
    order_id = f"order-{role}"
    manager.add_pending_order(
        order_id=order_id,
        symbol=leg["symbol"],
        side=leg["side"],
        qty=leg["quantity"],
        price=45.0,
        order_type="LIMIT",
        intent="ENTRY",
        bracket_id=PARENT,
    )
    manager.bind_linked_portfolio_order(PARENT, role, order_id)
    broker.orders.append(
        {
            "order_id": order_id,
            "tradingsymbol": leg["symbol"].split(":", 1)[1],
            "transaction_type": leg["side"],
            "quantity": leg["quantity"],
            "filled_quantity": filled,
            "average_price": 45.0 if filled else 0.0,
            "status": status,
            "product": "MIS",
        }
    )
    if filled:
        rows = broker.positions["net"]
        rows.append(
            {
                "tradingsymbol": leg["symbol"].split(":", 1)[1],
                "quantity": filled if leg["side"] == "BUY" else -filled,
            }
        )


def plan(manager, broker):
    view = manager.linked_portfolio_recovery_plan(PARENT)
    assert view["live_entry_allowed"] is False
    assert view["broker_execution_allowed"] is False
    assert broker.placed == 0
    assert broker.cancelled == 0
    return view


def test_entry_planning_requires_both_wings_before_any_short(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    assert plan(manager, broker)["next_roles"] == ["long_put"]
    add_child(manager, broker, parent, "long_put")
    assert plan(manager, broker)["next_roles"] == ["long_call"]
    add_child(manager, broker, parent, "long_call")
    view = plan(manager, broker)
    assert view["next_roles"] == ["short_put"]
    assert view["state"] == "RESEARCH_NEXT_LEG_ONLY"
    add_child(manager, broker, parent, "short_put")
    assert plan(manager, broker)["next_roles"] == ["short_call"]


def test_partial_open_order_cannot_advance_and_requires_cancel_reconcile(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    add_child(
        manager, broker, parent, "long_put",
        status="PARTIALLY FILLED", filled=25
    )
    view = plan(manager, broker)
    assert view["state"] == "CANCEL_THEN_RECONCILE"
    assert view["next_roles"] == []
    assert view["cancel_order_ids"] == ["order-long_put"]
    assert view["group_exit_intents"] == []


def test_terminal_partial_fill_must_unwind_before_any_new_leg(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    add_child(manager, broker, parent, "long_put", status="CANCELLED", filled=25)
    view = plan(manager, broker)
    assert view["state"] == "PARTIAL_EXPOSURE_RECOVERY_REQUIRED"
    assert view["next_roles"] == []
    assert view["group_exit_intents"] == [
        {
            "role": "long_put",
            "symbol": parent["legs"]["long_put"]["symbol"],
            "side": "SELL",
            "quantity": 25,
            "broker_submission_allowed": False,
        }
    ]


def test_group_exit_is_short_first_then_protective_longs(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    for role in ("long_put", "long_call", "short_put", "short_call"):
        add_child(manager, broker, parent, role)
    view = plan(manager, broker)
    assert view["state"] == "BROKER_PORTFOLIO_OBSERVED"
    assert view["next_roles"] == []
    assert [intent["role"] for intent in view["group_exit_intents"]] == [
        "short_put", "short_call", "long_put", "long_call"
    ]
    assert [intent["side"] for intent in view["group_exit_intents"]] == [
        "BUY", "BUY", "SELL", "SELL"
    ]


def test_mismatch_or_external_order_blocks_group_actions(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    add_child(manager, broker, parent, "long_put")
    broker.positions["net"][0]["quantity"] = -65
    view = plan(manager, broker)
    assert view["state"] == "RECONCILIATION_REQUIRED"
    assert view["reason"] == "broker_exposure_outside_parent_limits"
    assert view["group_exit_intents"] == []

    broker.positions["net"][0]["quantity"] = 65
    broker.orders.append(
        {
            "order_id": "manual-2",
            "tradingsymbol": parent["legs"]["long_put"]["symbol"].split(":", 1)[1],
            "transaction_type": "BUY",
            "quantity": 65,
            "filled_quantity": 65,
            "status": "COMPLETE",
        }
    )
    view = plan(manager, broker)
    assert view["reason"] == "external_order_on_parent_symbol"
    assert view["group_exit_intents"] == []


def test_unknown_broker_snapshot_does_not_infer_flat_or_exit(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    add_child(manager, broker, parent, "long_put")
    broker.positions = None
    view = plan(manager, broker)
    assert view["reason"] == "broker_snapshot_fetch_failed"
    assert view["group_exit_intents"] == []
    assert view["next_roles"] == []


def test_recovery_policy_survives_restart_without_submitting(tmp_path):
    manager, broker, parent = new_parent(tmp_path)
    add_child(manager, broker, parent, "long_put")
    restored = PositionManager(state_file=str(tmp_path / "positions.json"))
    restored.set_broker_client(broker)
    assert plan(restored, broker)["next_roles"] == ["long_call"]
