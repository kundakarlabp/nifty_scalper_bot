from __future__ import annotations

from decimal import Decimal

from src.broker.interface import OrderRequest, OrderType, Side, OrderStatus, Tick
from src.brokers.mock import MockBroker


def test_mock_connect_and_ltp() -> None:
    broker = MockBroker()
    broker.connect()
    assert broker.is_connected()
    assert broker.ltp(123) == Decimal("100.00")


def test_mock_place_and_get_order() -> None:
    broker = MockBroker()
    broker.connect()
    req = OrderRequest(instrument_id=123, side=Side.BUY, qty=10, order_type=OrderType.MARKET)
    oid = broker.place_order(req)
    order = broker.get_order(oid)
    assert order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
    assert order.order_id == oid


def test_mock_ticks_subscription() -> None:
    broker = MockBroker()
    received: list[Tick] = []

    def on_tick(t: Tick) -> None:
        received.append(t)

    broker.subscribe_ticks([123], on_tick)
    broker.push_tick(123, Decimal("101.25"))
    assert received and received[0].instrument_id == 123 and received[0].ltp == Decimal("101.25")
