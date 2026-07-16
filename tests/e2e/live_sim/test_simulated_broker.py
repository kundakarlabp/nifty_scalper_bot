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
    broker.modify_order(stop, quantity=50, price=96, trigger_price=96)
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
