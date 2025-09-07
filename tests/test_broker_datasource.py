from __future__ import annotations

from decimal import Decimal

from src.data.broker_source import BrokerDataSource
from src.brokers.mock import MockBroker


def test_broker_datasource_tick_flow() -> None:
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    received = []

    def _cb(tick):
        received.append(tick)

    ds.set_tick_callback(_cb)
    ds.subscribe([101])
    ds.start()
    broker.push_tick(101, Decimal("100.5"))
    assert len(received) == 1
    assert received[0].ltp == Decimal("100.5")
