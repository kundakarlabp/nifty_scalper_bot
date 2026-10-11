from .simulated_broker import SimulatedBroker
from .virtual_clock import VirtualClock


def test_simulated_broker_fill_cancel_modify_duplicate_and_positions():
    broker = SimulatedBroker(VirtualClock())
    updates = []
    broker.register_callback(updates.append)
    entry_response = broker.place_order(
        symbol="NFO:CE", side="BUY", quantity=100, order_type="LIMIT", price=100
    )
    entry = entry_response["order_id"]
    broker.on_quote("NFO:CE", bid=99.8, ask=100, ltp=100)
    assert broker.query_order(entry).status == "COMPLETE"
    assert broker.query_positions()["NFO:CE"] == 100

    stop_response = broker.place_order(
        symbol="NFO:CE",
        side="SELL",
        quantity=40,
        order_type="SL",
        price=95,
        trigger_price=95,
    )
    stop = stop_response["order_id"]
    broker.modify_order(stop, quantity=50, price=95.5, trigger_price=96)
    broker.on_quote("NFO:CE", bid=96.5, ask=97, ltp=96.5)
    assert broker.query_order(stop).status == "TRIGGER_PENDING"
    broker.on_quote("NFO:CE", bid=95.9, ask=96, ltp=95.9)
    assert broker.query_positions()["NFO:CE"] == 50

    target_response = broker.place_order(
        symbol="NFO:CE", side="SELL", quantity=50, order_type="LIMIT", price=110
    )
    target = target_response["order_id"]
    broker.fill(target, 25, 110)
    broker.fill(target, 25, 110, duplicate=True)
    broker.fill(target, 25, 110)
    assert broker.query_positions()["NFO:CE"] == 0
    cancelled_response = broker.place_order(
        symbol="NFO:X", side="BUY", quantity=1, order_type="LIMIT", price=1
    )
    cancelled = cancelled_response["order_id"]
    broker.cancel_order(cancelled)
    assert broker.query_order(cancelled).status == "CANCELLED"
    assert updates


def test_sell_stop_limit_gap_through_rests_until_executable_bid_returns():
    broker = SimulatedBroker(VirtualClock())
    broker.positions["NFO:PE"] = 65
    placed = broker.place_order(
        symbol="NFO:PE",
        side="SELL",
        quantity=65,
        order_type="SL",
        price=95.0,
        trigger_price=100.0,
    )
    order = broker.query_order(placed["order_id"])
    assert order.status == "TRIGGER_PENDING"

    broker.on_quote("NFO:PE", bid=92.0, ask=93.0, ltp=92.0)
    assert order.stop_triggered
    assert order.status == "OPEN"
    assert order.filled_quantity == 0
    assert broker.query_positions()["NFO:PE"] == 65
    assert broker.query_trades() == []

    broker.on_quote("NFO:PE", bid=94.0, ask=94.5, ltp=94.2)
    assert order.filled_quantity == 0
    broker.on_quote("NFO:PE", bid=96.0, ask=96.5, ltp=96.2)
    assert order.status == "COMPLETE"
    assert order.average_price == 96.0
    assert broker.query_positions()["NFO:PE"] == 0


def test_stop_market_gap_does_not_reuse_stale_trigger_as_fill_price():
    broker = SimulatedBroker(VirtualClock())
    broker.positions["NFO:PE"] = 65
    placed = broker.place_order(
        symbol="NFO:PE",
        side="SELL",
        quantity=65,
        order_type="SL-M",
        price=95.0,
        trigger_price=100.0,
    )
    broker.on_quote("NFO:PE", bid=89.0, ask=90.0, ltp=89.5)
    order = broker.query_order(placed["order_id"])
    assert order.status == "COMPLETE"
    assert order.average_price == 89.0
    assert broker.query_positions()["NFO:PE"] == 0


def test_cancelled_triggered_unfilled_stop_cannot_sell_later():
    broker = SimulatedBroker(VirtualClock())
    broker.positions["NFO:PE"] = 65
    placed = broker.place_order(
        symbol="NFO:PE",
        side="SELL",
        quantity=65,
        order_type="SL",
        price=95.0,
        trigger_price=100.0,
    )
    order_id = placed["order_id"]
    broker.on_quote("NFO:PE", bid=90.0, ask=91.0, ltp=90.5)
    assert broker.query_order(order_id).status == "OPEN"
    broker.cancel_order(order_id)
    broker.on_quote("NFO:PE", bid=99.0, ask=99.5, ltp=99.1)
    assert broker.query_order(order_id).status == "CANCELLED"
    assert broker.query_positions()["NFO:PE"] == 65
    assert broker.query_trades() == []
