from __future__ import annotations

from src.execution.broker_executor import BrokerOrderExecutor
from src.brokers.mock import MockBroker
from src.broker.interface import OrderRequest, Side
from src.broker.instruments import Instrument, InstrumentStore


def test_executor_accepts_order_request() -> None:
    broker = MockBroker()
    broker.connect()
    executor = BrokerOrderExecutor(broker)
    oid = executor.place_order(OrderRequest(instrument_id=1, side=Side.BUY, qty=10))
    assert oid.startswith("MOCK-")


def test_executor_accepts_dict_with_mapper() -> None:
    broker = MockBroker()
    store = InstrumentStore([Instrument(token=123, symbol="ABC")])
    mapper = lambda sym: store.by_symbol(sym).token  # noqa: E731 - simple lambda
    executor = BrokerOrderExecutor(broker, instrument_id_mapper=mapper)
    broker.connect()
    oid = executor.place_order({"symbol": "ABC", "side": "BUY", "qty": 5})
    assert oid.startswith("MOCK-")


def test_executor_buy_sell_helpers() -> None:
    broker = MockBroker()
    store = InstrumentStore([Instrument(token=200, symbol="XYZ")])
    mapper = lambda sym: store.by_symbol(sym).token  # noqa: E731 - simple lambda
    executor = BrokerOrderExecutor(broker, instrument_id_mapper=mapper)
    broker.connect()
    buy_id = executor.buy("XYZ", 1)
    sell_id = executor.sell("XYZ", 1)
    assert buy_id != sell_id
