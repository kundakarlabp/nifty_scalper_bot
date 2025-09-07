from __future__ import annotations

import os

import pytest

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


def test_kill_switch_env_blocks_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBroker()
    broker.connect()
    executor = BrokerOrderExecutor(broker)
    monkeypatch.setenv("KILL_SWITCH", "1")
    with pytest.raises(RuntimeError):
        executor.place_order(OrderRequest(instrument_id=1, side=Side.BUY, qty=1))


def test_kill_switch_file_blocks_orders(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBroker()
    broker.connect()
    executor = BrokerOrderExecutor(broker)
    flag = tmp_path / "ks"
    flag.write_text("1")
    monkeypatch.setenv("KILL_SWITCH_FILE", str(flag))
    with pytest.raises(RuntimeError):
        executor.place_order(OrderRequest(instrument_id=1, side=Side.BUY, qty=1))
    monkeypatch.delenv("KILL_SWITCH_FILE")
    os.remove(flag)


def test_client_id_generated_and_deduped(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBroker()
    broker.connect()
    executor = BrokerOrderExecutor(broker)

    seen: list[str] = []

    orig = broker.place_order

    def spy(req: OrderRequest) -> str:
        assert req.client_order_id is not None
        seen.append(req.client_order_id)
        return orig(req)

    monkeypatch.setattr(broker, "place_order", spy)
    req = OrderRequest(instrument_id=1, side=Side.BUY, qty=1, client_order_id="X1")
    oid1 = executor.place_order(req)
    oid2 = executor.place_order(req)
    assert oid1 == oid2
    assert seen == ["X1"]


def test_auto_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = MockBroker()
    broker.connect()
    executor = BrokerOrderExecutor(broker)
    captured: list[str] = []

    orig = broker.place_order

    def spy(req: OrderRequest) -> str:
        captured.append(req.client_order_id or "")
        return orig(req)

    monkeypatch.setattr(broker, "place_order", spy)
    executor.place_order({"instrument_id": 1, "side": "BUY", "qty": 1})
    assert captured and captured[0]
