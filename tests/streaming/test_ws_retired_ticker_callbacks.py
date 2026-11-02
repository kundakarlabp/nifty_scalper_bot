"""Retired KiteTicker callbacks must not corrupt the active transport generation."""

from __future__ import annotations

from nifty_scalper_bot.streaming.websocket_manager import (
    ConnectionState,
    WebSocketManager,
)


def test_retired_ticker_callbacks_cannot_disconnect_or_feed_active_session() -> None:
    delivered = []
    manager = WebSocketManager(
        api_key="dummy",
        access_token="dummy",
        on_tick=lambda tick: delivered.append(tick),
    )
    active = object()
    retired = object()
    reconnects = []
    manager._ticker = active
    manager._connected.set()
    manager._state = ConnectionState.CONNECTED
    manager._stream_health = "healthy"
    manager._last_pong_mono = 123.0
    manager._schedule_reconnect = lambda reason: reconnects.append(reason)

    manager._on_ticks(retired, [{"instrument_token": 123}])
    manager._on_pong(retired)
    manager._on_error(retired, 1006, "late old socket")
    manager._on_close(retired, 1006, "late old socket")
    manager._on_connect(retired, {})

    assert delivered == []
    assert reconnects == []
    assert manager._connected.is_set()
    assert manager._state is ConnectionState.CONNECTED
    assert manager._stream_health == "healthy"
    assert manager._last_pong_mono == 123.0


def test_current_ticker_close_still_disarms_and_schedules_reconnect() -> None:
    manager = WebSocketManager(api_key="dummy", access_token="dummy")
    active = object()
    manager._ticker = active
    manager._connected.set()
    manager._state = ConnectionState.CONNECTED
    manager._is_within_trading_window = lambda: True
    reconnects = []
    manager._schedule_reconnect = lambda reason: reconnects.append(reason)

    manager._on_close(active, 1006, "closing_handshake_timeout")

    assert not manager._connected.is_set()
    assert manager._state is ConnectionState.DISCONNECTED
    assert manager._stream_health == "degraded"
    assert reconnects == ["close:1006"]
