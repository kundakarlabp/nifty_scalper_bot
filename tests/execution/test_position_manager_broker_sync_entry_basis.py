from __future__ import annotations

from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY2681824400PE"


def _broker_row(*, qty: int, average_price: float, last_price: float) -> dict[str, object]:
    return {
        "tradingsymbol": SYMBOL,
        "exchange": "NFO",
        "product": "MIS",
        "quantity": qty,
        "average_price": average_price,
        "last_price": last_price,
        "realised": 0.0,
    }


def test_flat_to_same_symbol_reentry_fill_repairs_day_aggregate_broker_basis(tmp_path) -> None:
    """A broker-sync race may reflect entry quantity before the order fill callback."""
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order_id = "2087403339074953216"

    manager.add_pending_order(order_id, SYMBOL, "BUY", 65, intent="ENTRY")
    assert manager._orders[order_id].pre_order_quantity == 0

    # Reproduce 12-Aug live race: periodic broker sync lands after the new buy
    # but before its order-fill callback. 113.90 is the day aggregate; the
    # authoritative new order fill is 124.45.
    manager.synchronize_with_broker(
        [_broker_row(qty=65, average_price=113.90, last_price=124.30)]
    )
    assert manager._positions[SYMBOL].quantity == 65
    assert manager._positions[SYMBOL].entry_price == 113.90

    manager.apply_broker_order_update(
        order_id,
        {
            "order_id": order_id,
            "tradingsymbol": SYMBOL,
            "status": "COMPLETE",
            "filled_quantity": 65,
            "average_price": 124.45,
        },
    )

    position = manager._positions[SYMBOL]
    assert position.quantity == 65
    assert position.entry_price == 124.45
    assert position.order_id == order_id


def test_scale_in_broker_sync_race_keeps_broker_weighted_basis(tmp_path) -> None:
    """Fresh-entry repair must not reinterpret a real scale-in as a new lot."""
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.open_position(SYMBOL, "LONG", 65, 100.0, order_id="entry-old")
    order_id = "scale-1"
    manager.add_pending_order(order_id, SYMBOL, "BUY", 65, intent="SCALE_IN")
    assert manager._orders[order_id].pre_order_quantity == 65

    manager.synchronize_with_broker(
        [_broker_row(qty=130, average_price=110.0, last_price=120.0)]
    )
    manager.apply_broker_order_update(
        order_id,
        {
            "order_id": order_id,
            "tradingsymbol": SYMBOL,
            "status": "COMPLETE",
            "filled_quantity": 65,
            "average_price": 120.0,
        },
    )

    position = manager._positions[SYMBOL]
    assert position.quantity == 130
    assert position.entry_price == 110.0
