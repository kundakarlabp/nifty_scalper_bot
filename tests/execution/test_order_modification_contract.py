from __future__ import annotations

import inspect
import logging
import threading
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


@pytest.fixture
def modification_manager(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    manager = OrderManager.__new__(OrderManager)
    manager._logger = logging.getLogger(__name__)
    manager._lock = threading.RLock()
    manager._orders = {
        "order-1": SimpleNamespace(order_type=OrderType.LIMIT, side="SELL")
    }
    calls = []

    def modify_order(order_id, **changes):
        calls.append((order_id, changes))
        return {"order_id": order_id}

    manager._broker = SimpleNamespace(modify_order=modify_order)
    return manager, calls


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"price": 101.23}, {"price": 101.25}),
        ({"quantity": 65}, {"quantity": 65}),
        ({"trigger_price": 99.48}, {"trigger_price": 99.5}),
    ],
)
def test_modification_changes_only_requested_fields(
    modification_manager, changes, expected
):
    manager, calls = modification_manager

    assert manager.modify_order("order-1", **changes) is True
    assert calls == [("order-1", {"variety": "regular", **expected})]


def test_stop_trigger_modification_keeps_buffer_and_tick_grid(modification_manager):
    manager, calls = modification_manager
    manager._orders["order-1"].order_type = OrderType.STOP_LOSS

    assert manager.modify_order("order-1", trigger_price=99.45) is True
    assert calls == [
        ("order-1", {"variety": "regular", "price": 94.5, "trigger_price": 99.45})
    ]


def test_unknown_order_and_empty_modification_never_call_broker(modification_manager):
    manager, calls = modification_manager

    assert manager.modify_order("missing", price=100.0) is False
    assert manager.modify_order("order-1") is False
    assert calls == []


def test_broker_modification_failure_is_not_success(modification_manager):
    manager, calls = modification_manager

    def fail(*args, **kwargs):
        raise RuntimeError("broker rejected modification")

    manager._broker.modify_order = fail
    assert manager.modify_order("order-1", quantity=65) is False
    assert calls == []


def test_stop_resize_uses_existing_order_without_cancel_replacement(
    modification_manager,
):
    manager, calls = modification_manager
    state = SimpleNamespace(stop_order_id="order-1")

    manager._resize_stop_order(state, 65)

    assert calls == [("order-1", {"variety": "regular", "quantity": 65})]
    assert manager._orders["order-1"].quantity == 65


@pytest.mark.parametrize("target", ["primary", "secondary"])
def test_partial_target_resize_sends_total_quantity_to_broker(
    modification_manager, target
):
    manager, calls = modification_manager
    state = SimpleNamespace(
        tp_primary_id="order-1",
        tp_secondary_id="order-1",
        tp_primary_filled=10,
        tp_secondary_filled=10,
        tp_primary_qty=100,
        tp_secondary_qty=100,
    )

    manager._resize_target_order(state, target=target, new_outstanding=55)

    assert calls == [("order-1", {"variety": "regular", "quantity": 65})]
    assert manager._orders["order-1"].quantity == 65
    assert getattr(state, f"tp_{target}_qty") == 65


def test_repricing_and_resize_callers_match_public_signature():
    import ast
    from pathlib import Path

    source = Path("src/nifty_scalper_bot/execution/order_manager_core.py").read_text()
    signature = inspect.signature(OrderManager.modify_order)
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "modify_order"
    ]
    assert len(calls) >= 4
    for call in calls:
        signature.bind(
            None, *[None for _ in call.args], **{kw.arg: None for kw in call.keywords}
        )
