from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pytest

from nifty_scalper_bot.data.persistent_state import PersistentStateManager
from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    OrderStatus,
    OrderType,
)
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


class _DummyBroker:
    """Minimal broker stub returning configured open orders."""

    def __init__(self, open_orders: Iterable[Mapping[str, Any]]) -> None:
        self._open_orders = list(open_orders)

    def get_open_orders(self) -> list[Mapping[str, Any]]:
        """Return configured open orders."""

        return list(self._open_orders)


@pytest.mark.integration
def test_persistent_state_recovery(tmp_path) -> None:
    """Ensure orders and brackets recover cleanly after a simulated crash."""

    base_path = tmp_path / "store"
    manager = PersistentStateManager(base_path=base_path)
    now = datetime.now(timezone.utc).isoformat()

    position_symbol = "NIFTY24OCT17500CE"
    manager.save_position(
        {"symbol": position_symbol, "quantity": 2, "average_price": 105.5}
    )

    manager.save_order(
        {
            "order_id": "OPEN-1",
            "symbol": position_symbol,
            "side": "BUY",
            "order_type": OrderType.MARKET.value,
            "quantity": 2,
            "price": 100.0,
            "status": OrderStatus.SUBMITTED.value,
            "timestamp": now,
            "filled_quantity": 0,
            "child_order_ids": [],
        }
    )
    manager.save_order(
        {
            "order_id": "FILLED-1",
            "symbol": position_symbol,
            "side": "SELL",
            "order_type": OrderType.MARKET.value,
            "quantity": 2,
            "price": 100.0,
            "status": OrderStatus.SUBMITTED.value,
            "timestamp": now,
            "filled_quantity": 0,
            "child_order_ids": [],
        }
    )

    manager.save_bracket(
        {
            "entry_id": "OPEN-1",
            "symbol": position_symbol,
            "side": "BUY",
            "exit_side": "SELL",
            "total_quantity": 2,
            "entry_price": 100.0,
            "product": "NRML",
            "tag": None,
            "stop_order_id": "STOP-1",
            "stop_price": 95.0,
            "stop_order_type": OrderType.STOP_LOSS.value,
            "stop_filled": 0,
            "tp_primary_id": None,
            "tp_primary_price": None,
            "tp_primary_qty": 0,
            "tp_primary_filled": 0,
            "tp_secondary_id": None,
            "tp_secondary_price": None,
            "tp_secondary_qty": 0,
            "tp_secondary_filled": 0,
            "partial_fraction": 0.0,
            "second_target_price": None,
        }
    )
    manager.flush()
    manager.close()

    restored_manager = PersistentStateManager(base_path=base_path)

    pos_manager = PositionManager(state_file=str(tmp_path / "positions_state.json"))
    pos_manager.attach_persistent_state(restored_manager)
    pos_manager.restore_positions(restored_manager.load_positions())
    assert any(
        pos.symbol == position_symbol for pos in pos_manager.get_open_positions()
    )

    broker = _DummyBroker(
        [
            {
                "order_id": "OPEN-1",
                "symbol": position_symbol,
                "side": "BUY",
                "status": "open",
                "quantity": 2,
                "price": 100.0,
            }
        ]
    )
    rate_limiter = RateLimiter()
    order_manager = OrderManager(
        broker_client=broker,
        position_manager=pos_manager,
        rate_limiter=rate_limiter,
    )
    order_manager.attach_persistent_state(restored_manager)

    history_before = {order.order_id for order in order_manager.get_order_history()}
    assert {"OPEN-1", "FILLED-1"}.issubset(history_before)

    order_manager.reconcile_open_orders_with_broker()

    open_orders = restored_manager.load_open_orders()
    assert [order["order_id"] for order in open_orders] == ["OPEN-1"]

    reconciled = {order.order_id: order for order in order_manager.get_order_history()}
    assert reconciled["FILLED-1"].status == OrderStatus.FILLED
    assert reconciled["OPEN-1"].status not in OrderManager.FINAL_STATUSES
    assert "OPEN-1" in order_manager._brackets  # noqa: SLF001 - integration visibility

    restored_manager.close()
