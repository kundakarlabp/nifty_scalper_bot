from __future__ import annotations

"""Tests for data source utilities."""

from collections import deque
from dataclasses import dataclass

import pytest

from src.data.base_source import BaseDataSource
from src.data.broker_source import BrokerDataSource


class DummySource(BaseDataSource):
    _last_tick_ts = None
    _last_bar_open_ts = None
    _tf_seconds = 90


def test_base_data_source_defaults() -> None:
    source = BaseDataSource()

    assert source.last_tick_dt() is None
    assert source.last_bar_open_ts() is None
    assert source.timeframe_seconds == 60


def test_base_data_source_with_overrides() -> None:
    dummy = DummySource()

    dummy._last_tick_ts = 123
    dummy._last_bar_open_ts = 456

    assert dummy.last_tick_dt() == 123
    assert dummy.last_bar_open_ts() == 456
    assert dummy.timeframe_seconds == 90


@dataclass
class FakeBroker:
    connected: bool = False

    def __post_init__(self) -> None:
        self.subscriptions = deque()
        self.reconnect_handlers = deque()
        self.disconnect_handlers = deque()

    def subscribe_ticks(self, instruments, handler):
        self.subscriptions.append((tuple(instruments), handler))

    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def on_reconnect(self, cb):
        self.reconnect_handlers.append(cb)

    def on_disconnect(self, cb):
        self.disconnect_handlers.append(cb)


def test_broker_data_source_requires_callback() -> None:
    ds = BrokerDataSource(FakeBroker())

    with pytest.raises(RuntimeError):
        ds.subscribe([1, 2])


def test_broker_data_source_subscribe_and_start() -> None:
    broker = FakeBroker()
    ds = BrokerDataSource(broker)
    received = []

    ds.set_tick_callback(received.append)
    ds.subscribe([101])
    ds.start()

    assert broker.connected
    # Validate that the handler echoes ticks to the callback.
    _, handler = broker.subscriptions.pop()
    handler({"instrument_token": 101, "last_price": 10})
    assert received == [{"instrument_token": 101, "last_price": 10}]
    # Hooks should be registered for reconnection handling.
    assert broker.reconnect_handlers and broker.disconnect_handlers
    # Invoke the stored reconnect callback to ensure it resubscribes.
    broker.reconnect_handlers.pop()()
    assert broker.subscriptions[-1][0] == (101,)


def test_broker_data_source_handles_callback_errors() -> None:
    broker = FakeBroker(connected=True)
    ds = BrokerDataSource(broker)
    ds.set_tick_callback(lambda tick: (_ for _ in ()).throw(RuntimeError("boom")))
    ds.subscribe([42])

    # Handler should swallow the exception and allow execution to continue.
    instruments, handler = broker.subscriptions.pop()
    assert instruments == (42,)
    handler({})


def test_broker_data_source_stop_and_no_callback() -> None:
    broker = FakeBroker(connected=True)
    ds = BrokerDataSource(broker)

    ds.stop()
    assert not broker.connected

    # With no callback set, handler should immediately return.
    ds._handle_tick({"token": 1})
