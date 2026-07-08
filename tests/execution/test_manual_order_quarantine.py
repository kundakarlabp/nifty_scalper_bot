from __future__ import annotations

from nifty_scalper_bot.execution import position_identity_extension as identity_ext
from nifty_scalper_bot.execution.position_manager import Order, PositionManager


SYMBOL = "NFO:NIFTY2670724250PE"


def test_unknown_buy_fill_does_not_scale_existing_long_position(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.open_position(SYMBOL, "LONG", 65, 75.0, order_id="entry-1")
    order = Order(
        order_id="manual-buy",
        symbol=SYMBOL,
        side="BUY",
        order_type="MARKET",
        quantity=65,
        price=80.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=80.0,
        intent="UNKNOWN",
    )
    manager._orders[order.order_id] = order

    manager.update_order_status(order.order_id, "COMPLETE", fill_price=80.0)

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65
    assert position.entry_price == 75.0
    terminal = manager._terminal_orders[order.order_id]
    assert terminal.intent == "UNKNOWN"
    assert terminal.lifecycle_applied is False
    assert terminal.lifecycle_resolved is False
    exposures = manager.get_quarantined_broker_exposures()
    assert exposures[SYMBOL]["status"] == "MANUAL_ORDER_QUARANTINED"
    assert exposures[SYMBOL]["order_id"] == "manual-buy"
    assert exposures[SYMBOL]["reason"] == "manual_order_quarantined"


def test_unknown_sell_fill_is_recognized_as_manual_exit_for_existing_long_position(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.open_position(SYMBOL, "LONG", 65, 75.0, order_id="entry-1")
    order = Order(
        order_id="manual-sell",
        symbol=SYMBOL,
        side="SELL",
        order_type="MARKET",
        quantity=65,
        price=80.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=80.0,
        intent="UNKNOWN",
    )
    manager._orders[order.order_id] = order

    manager.update_order_status(order.order_id, "COMPLETE", fill_price=80.0)

    assert manager.get_position(SYMBOL) is None
    assert manager.get_realized_pnl() == 325.0
    terminal = manager._terminal_orders[order.order_id]
    assert terminal.intent == "REDUCE"
    assert terminal.lifecycle_applied is True
    assert terminal.pnl_applied is True
    assert terminal.lifecycle_resolved is True
    assert manager.get_quarantined_broker_exposures() == {}


def test_unresolved_broker_cost_basis_blocks_entries_without_sync_exception(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))

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

    assert manager.get_position(SYMBOL) is None
    assert manager.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"
    exposures = manager.get_quarantined_broker_exposures()
    assert exposures[SYMBOL]["reason"] == "cost_basis_unresolved"
    assert exposures[SYMBOL]["status"] == "BROKER_POSITION_QUARANTINED"
    prepared, unresolved = identity_ext._prepare_broker_positions(
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
    assert prepared[0]["symbol"] == SYMBOL
    assert unresolved == {SYMBOL}
