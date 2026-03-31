from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager


class _StubRateLimiter:
    def acquire(self, *_args: object, **_kwargs: object) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {}


class _StubBroker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.cancelled: list[str] = []

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        order_id = f"ord-{len(self.orders) + 1}"
        self.orders[order_id] = {"payload": payload, "status": "OPEN"}
        return {"order_id": order_id, "status": "OPEN"}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.cancelled.append(order_id)
        self.orders.setdefault(order_id, {})["status"] = "CANCELLED"
        return {"status": "CANCELLED"}

    def get_positions(self) -> list[dict[str, Any]]:
        return []


def _make_order_manager(tmp_path: Any) -> tuple[OrderManager, _StubBroker]:
    broker = _StubBroker()
    limiter = _StubRateLimiter()
    position_manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager = OrderManager(
        broker,
        position_manager,
        limiter,
        history_path=tmp_path / "orders.json",
    )
    manager._ensure_trading_allowed = lambda **_kwargs: True  # type: ignore[attr-defined]
    return manager, broker


def _mark_filled(
    manager: OrderManager,
    order_id: str,
    *,
    price: float,
) -> None:
    order = manager._orders[order_id]  # type: ignore[attr-defined]
    payload = {
        "status": "complete",
        "filled_quantity": order.quantity,
        "average_price": price,
    }
    manager._update_from_response(order, payload)


@pytest.mark.parametrize("fill_first", ["stop", "target"])
def test_guard_pair_cancels_sibling_on_fill(tmp_path, fill_first: str) -> None:
    manager, broker = _make_order_manager(tmp_path)

    manager.guard_existing_position(
        symbol="NSE:ABC",
        side="LONG",
        quantity=3,
        average_price=105.0,
        last_price=104.5,
        product="MIS",
    )

    assert manager.has_guard_pair("NSE:ABC")
    orders = list(broker.orders.keys())
    assert len(orders) == 2
    stop_id, target_id = orders

    first_id = stop_id if fill_first == "stop" else target_id
    second_id = target_id if fill_first == "stop" else stop_id

    _mark_filled(manager, first_id, price=103.0)

    assert second_id in broker.cancelled
    assert not manager.has_guard_pair("NSE:ABC")


def test_guard_pair_removed_when_position_flat(tmp_path) -> None:
    manager, broker = _make_order_manager(tmp_path)

    manager.guard_existing_position(
        symbol="NSE:XYZ",
        side="LONG",
        quantity=2,
        average_price=120.0,
        last_price=121.0,
        product="MIS",
    )

    assert manager.has_guard_pair("NSE:XYZ")
    stop_id, target_id = list(broker.orders.keys())

    manager.clear_guard_pair("NSE:XYZ")

    assert stop_id in broker.cancelled
    assert target_id in broker.cancelled
    assert not manager.has_guard_pair("NSE:XYZ")
