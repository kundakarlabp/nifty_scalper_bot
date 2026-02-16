from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from nifty_scalper_bot.streaming import websocket_manager as ws_module


class FakeKiteTicker:
    MODE_FULL = "full"

    def __init__(self, api_key: str, access_token: str, reconnect: bool = True) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.reconnect = reconnect
        self.on_connect: Callable[..., None] | None = None
        self.on_ticks: Callable[..., None] | None = None
        self.on_error: Callable[..., None] | None = None
        self.on_close: Callable[..., None] | None = None
        self.connect_calls = 0

    def connect(self, threaded: bool = True) -> None:
        del threaded
        self.connect_calls += 1
        if self.on_connect:
            self.on_connect(self, {"ok": True})

    def close(self) -> None:
        return None

    def subscribe(self, _tokens: list[int]) -> None:
        return None

    def set_mode(self, _mode: str, _tokens: list[int]) -> None:
        return None


@pytest.mark.asyncio
async def test_manager_disables_internal_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager("k", "t")
    ticker = manager._build_ticker()
    assert ticker.reconnect is False


@pytest.mark.asyncio
async def test_single_reconnect_task_is_scheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager("k", "t", trading_window_enabled=False)
    manager._next_backoff_delay = lambda: 0.01
    await manager.connect()

    try:
        manager._on_error(manager.ticker, 1006, "drop")
        manager._on_close(manager.ticker, 1006, "drop")

        await asyncio.sleep(0.05)
        assert manager._ticker is not None
    finally:
        await manager.disconnect()
