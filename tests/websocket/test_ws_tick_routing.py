"""Tests covering WebSocketManager tick callback routing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from nifty_scalper_bot.streaming import websocket_manager as ws_module


class FakeKiteTicker:
    MODE_FULL = 'full'

    def __init__(self, api_key: str, access_token: str, reconnect: bool = True) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.reconnect = reconnect
        self.on_connect: Callable[..., None] | None = None
        self.on_ticks: Callable[..., None] | None = None
        self.on_error: Callable[..., None] | None = None
        self.on_close: Callable[..., None] | None = None

    def connect(self, threaded: bool = True) -> None:
        del threaded
        if self.on_connect:
            self.on_connect(self, {'ok': True})

    def close(self) -> None:
        return None

    def subscribe(self, _tokens: list[int]) -> None:
        return None

    def set_mode(self, _mode: str, _tokens: list[int]) -> None:
        return None


@pytest.mark.asyncio
async def test_manager_routes_ticks_to_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    async def _callback(tick: dict[str, Any]) -> None:
        received.append(tick)

    monkeypatch.setattr(ws_module, 'KiteTicker', FakeKiteTicker)
    manager = ws_module.WebSocketManager('key', 'token', on_tick=_callback)
    await manager.connect()

    assert manager.ticker.on_ticks is not None
    manager.ticker.on_ticks(manager.ticker, [{'instrument_token': 101}])
    await asyncio.sleep(0)

    assert received == [{'instrument_token': 101}]
