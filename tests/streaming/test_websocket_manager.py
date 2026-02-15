"""Tests for async KiteTicker websocket manager."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from nifty_scalper_bot.streaming import websocket_manager as ws_module


class FakeKiteTicker:
    """Args: api_key/access_token; Returns: fake; Raises: none."""

    MODE_FULL = "full"

    def __init__(self, api_key: str, access_token: str) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.on_connect: Callable[..., None] | None = None
        self.on_ticks: Callable[..., None] | None = None
        self.on_error: Callable[..., None] | None = None
        self.on_close: Callable[..., None] | None = None
        self.connect_calls = 0
        self.subscribed: list[int] = []
        self.mode_set: list[int] = []
        self.closed = False

    def connect(self, threaded: bool = True) -> None:
        self.connect_calls += 1
        if self.on_connect:
            self.on_connect(self, {"status": "ok"})

    def subscribe(self, tokens: list[int]) -> None:
        self.subscribed.extend(tokens)

    def set_mode(self, _mode: str, tokens: list[int]) -> None:
        self.mode_set.extend(tokens)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_uses_single_ticker_and_subscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager("k", "t", [111, 222])

    first_ticker = manager.ticker
    await manager.connect()

    assert manager.ticker is first_ticker
    assert first_ticker.connect_calls == 1
    assert set(first_ticker.subscribed) == {111, 222}


@pytest.mark.asyncio
async def test_ticks_dispatch_to_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    received: list[dict[str, Any]] = []

    async def _on_tick(tick: dict[str, Any]) -> None:
        received.append(tick)

    manager = ws_module.WebSocketManager("k", "t", on_tick_callback=_on_tick)
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    assert fake_ticker.on_ticks is not None
    fake_ticker.on_ticks(fake_ticker, [{"instrument_token": 99, "last_price": 12.4}])
    await asyncio.sleep(0)

    assert received == [{"instrument_token": 99, "last_price": 12.4}]


@pytest.mark.asyncio
async def test_reconnect_scheduled_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager("k", "t")
    manager._calculate_backoff = lambda _attempt: 0.01
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    fake_ticker.on_error(fake_ticker, 1006, "drop")
    fake_ticker.on_close(fake_ticker, 1006, "drop")
    await asyncio.sleep(0.08)

    assert fake_ticker.connect_calls >= 2

    await manager.disconnect()
    calls_before = fake_ticker.connect_calls
    fake_ticker.on_close(fake_ticker, 1000, "manual")
    await asyncio.sleep(0.05)

    assert fake_ticker.connect_calls == calls_before
