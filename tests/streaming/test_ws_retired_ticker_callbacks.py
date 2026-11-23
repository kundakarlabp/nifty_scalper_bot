"""Retired KiteTicker callbacks must not corrupt the active transport generation."""

from __future__ import annotations

import threading

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


def test_expected_shutdown_1006_does_not_raise_alarm_or_reconnect() -> None:
    """A systemd-controlled stop is not a live-feed outage."""
    manager = WebSocketManager(api_key="dummy", access_token="dummy")
    active = object()
    manager._ticker = active
    manager._connected.set()
    manager._state = ConnectionState.CONNECTED
    manager._manual_disconnect = True
    manager._shutdown = True
    reconnects = []
    errors = []
    manager._schedule_reconnect = lambda reason: reconnects.append(reason)
    manager._on_error_callback = lambda error: errors.append(error)

    manager._on_error(active, 1006, "closing_handshake_timeout")
    manager._on_close(active, 1006, "closing_handshake_timeout")

    assert not manager._connected.is_set()
    assert manager._state is ConnectionState.DISCONNECTED
    assert manager._stream_health == "idle"
    assert reconnects == []
    assert errors == []


def test_live_1006_still_notifies_error_and_reconnects() -> None:
    """Never suppress unexpected faults outside deliberate shutdown."""
    manager = WebSocketManager(api_key="dummy", access_token="dummy")
    active = object()
    manager._ticker = active
    manager._connected.set()
    manager._state = ConnectionState.CONNECTED
    manager._shutdown = False
    manager._manual_disconnect = False
    reconnects = []
    errors = []
    manager._schedule_reconnect = lambda reason: reconnects.append(reason)
    manager._on_error_callback = lambda error: errors.append(str(error))

    manager._on_error(active, 1006, "unexpected_disconnect")

    assert manager._stream_health == "degraded"
    assert reconnects == ["error:1006"]
    assert len(errors) == 1
    assert "unexpected_disconnect" in errors[0]


def test_kite_order_updates_are_serial_and_do_not_block_tick_thread() -> None:
    manager = WebSocketManager(api_key="dummy", access_token="dummy")
    active = type("Ticker", (), {})()
    manager._ticker = active
    received: list[str] = []
    completed = threading.Event()

    def on_order(update: dict[str, str]) -> None:
        received.append(update["order_id"])
        if len(received) == 2:
            completed.set()

    manager.set_order_update_callback(on_order)
    manager._bind_handlers(active)
    assert active.on_order_update is not None

    active.on_order_update(active, {"order_id": "broker-1", "status": "OPEN"})
    active.on_order_update(active, {"order_id": "broker-2", "status": "COMPLETE"})
    assert completed.wait(timeout=2)
    assert received == ["broker-1", "broker-2"]

    executor = manager._order_update_executor
    assert executor is not None
    executor.shutdown(wait=True)


def test_kite_order_updates_from_retired_session_and_invalid_ids_are_ignored() -> None:
    manager = WebSocketManager(api_key="dummy", access_token="dummy")
    active = object()
    manager._ticker = active
    received: list[dict[str, str]] = []
    manager.set_order_update_callback(received.append)

    manager._on_order_update(object(), {"order_id": "old", "status": "COMPLETE"})
    manager._on_order_update(active, {"status": "COMPLETE"})
    manager._on_order_update(active, "invalid")
    assert received == []
    assert manager._order_update_executor is None
