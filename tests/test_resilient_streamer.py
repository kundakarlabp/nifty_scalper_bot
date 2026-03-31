from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.streaming import ResilientStreamer


class DummySettings:
    def __init__(self, queue_size: int = 10) -> None:
        self.streamer = SimpleNamespace(
            gap_seconds=1.0,
            max_backfill_candles=2,
            backfill_interval="minute",
            queue_size=queue_size,
        )


class FakeWS:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.reconnect_called = False
        self.on_tick = lambda tick: None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def force_reconnect(self) -> None:
        self.reconnect_called = True

    def is_connected(self) -> bool:
        return True


class FakeRest:
    def get_ohlc(
        self, symbol: str, interval: str, from_date: str, to_date: str
    ) -> list[list[object]]:
        return [["2024-01-01T00:00:00", 1, 1, 1, 1, 1]]


@pytest.mark.asyncio
async def test_streamer_simulate_disconnect_triggers_reconnect() -> None:
    ws = FakeWS()
    rest = FakeRest()
    streamer = ResilientStreamer(ws, rest, DummySettings())
    streamer.simulate_disconnect()
    assert ws.reconnect_called


@pytest.mark.asyncio
async def test_streamer_backlog_size_grows_on_tick() -> None:
    ws = FakeWS()
    rest = FakeRest()
    streamer = ResilientStreamer(ws, rest, DummySettings())
    streamer._queue.put_nowait({})  # type: ignore[attr-defined]
    assert streamer.backlog_size() == 1


def test_handle_tick_enqueues_when_queue_not_full() -> None:
    ws = FakeWS()
    rest = FakeRest()
    streamer = ResilientStreamer(ws, rest, DummySettings())

    streamer._handle_tick({"symbol": "NIFTY", "ts_ms": 1_700_000_000_000, "ltp": 10})

    assert streamer.backlog_size() == 1


def test_handle_tick_replaces_oldest_when_queue_full() -> None:
    ws = FakeWS()
    rest = FakeRest()
    streamer = ResilientStreamer(ws, rest, DummySettings(queue_size=1))
    streamer._queue.put_nowait({"symbol": "OLD"})  # type: ignore[attr-defined]

    streamer._handle_tick({"symbol": "NEW", "ts_ms": 1_700_000_000_500, "ltp": 12})

    assert streamer.backlog_size() == 1
    queued = streamer._queue.get_nowait()  # type: ignore[attr-defined]
    assert queued["symbol"] == "NEW"
