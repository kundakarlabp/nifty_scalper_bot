"""Tests covering WebSocketManager callback wiring for non-Kite clients."""

from __future__ import annotations

from typing import Any, Callable

from nifty_scalper_bot.data.websocket.manager import WebSocketManager


class TickOnlyClient:
    """Websocket client exposing only an ``on_tick`` callback setter."""

    def __init__(self) -> None:
        self._callback: Callable[[dict[str, Any]], None] = lambda _tick: None
        # Attributes consumed by WebSocketManager during configuration
        self.on_error = None
        self.on_close = None
        self.on_reconnect = None
        self.on_open = None
        self.on_connect = None

    @property
    def on_tick(self) -> Callable[[dict[str, Any]], None]:
        return self._callback

    @on_tick.setter
    def on_tick(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._callback = callback or (lambda _tick: None)

    def connect(
        self, threaded: bool = True
    ) -> None:  # noqa: ARG002 - parity with real client
        return None

    def close(self) -> None:
        return None

    def trigger(self, payload: dict[str, Any]) -> None:
        self._callback(payload)


def test_manager_supports_on_tick_only_clients() -> None:
    """Manager should attach to ``on_tick`` when ``on_ticks`` is unavailable."""

    received: list[dict[str, Any]] = []

    client = TickOnlyClient()
    _ = WebSocketManager(client, received.append)

    client.trigger({"instrument_token": 101, "last_price": 123.45})

    assert received == [{"instrument_token": 101, "last_price": 123.45}]
