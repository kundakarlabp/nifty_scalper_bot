from __future__ import annotations

import types
from typing import Any

import pytest

from nifty_scalper_bot.data.websocket.manager import (
    ConnectionState,
    HandshakeTimeoutError,
    WebSocketManager,
)


class FakeWS:
    """Minimal broker websocket client supporting connect/disconnect hooks."""

    def __init__(self) -> None:
        self.on_ticks = None
        self.on_error = None
        self.connected = False
        self._fail_code: int | None = None
        self.refreshed = False

    def connect(self, threaded: bool = True) -> None:  # noqa: ARG002 - interface parity
        if self._fail_code:
            err = types.SimpleNamespace(code=self._fail_code)
            raise RuntimeError("boom", err)
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def refresh_session(self) -> None:
        self.refreshed = True


def test_exponential_backoff_and_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWS()
    ticks: list[dict[str, Any]] = []
    mgr = WebSocketManager(
        ws,
        on_tick=lambda d: ticks.append(d),
        heartbeat_sec=0.05,
        stale_sec=0.1,
        backoff_min_sec=0.01,
        backoff_max_sec=0.2,
        backoff_factor=2.0,
        backoff_jitter=0.0,
        breaker_threshold=2,
        breaker_open_sec=0.3,
    )

    now = [0.0]
    monkeypatch.setattr(
        "nifty_scalper_bot.data.websocket.manager.time.monotonic", lambda: now[0]
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.data.websocket.manager.time.sleep", lambda s: None
    )

    mgr.start()
    assert mgr.connection_state() in {
        ConnectionState.CONNECTING,
        ConnectionState.CONNECTED,
    }

    ws._fail_code = 1006
    mgr._handle_error(types.SimpleNamespace(code=1006))
    assert mgr.connection_state() in {
        ConnectionState.ERROR,
        ConnectionState.CONNECTING,
    }
    assert mgr._reconnect_failures == 1

    now[0] += 0.02
    with pytest.raises(Exception):  # noqa: B017 - broad for compatibility
        mgr._reconnect()
    assert mgr._reconnect_failures == 2

    now[0] += 0.03
    mgr._reconnect()
    assert mgr._reconnect_failures == 2
    assert mgr._breaker_open_until > now[0]

    before = mgr._breaker_open_until - now[0]
    mgr._reconnect()
    assert mgr._breaker_open_until - now[0] <= before

    now[0] = mgr._breaker_open_until + 0.01
    ws._fail_code = None
    mgr._reconnect()
    assert mgr._reconnect_failures == 0
    assert mgr.connection_state() in {
        ConnectionState.CONNECTING,
        ConnectionState.CONNECTED,
    }


def test_stuck_connecting_triggers_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeWS()
    mgr = WebSocketManager(
        ws,
        on_tick=lambda _: None,
        heartbeat_sec=0.05,
        stale_sec=0.1,
        backoff_min_sec=0.01,
        backoff_max_sec=0.02,
    )
    mgr._handshake_timeout = 0.5
    mgr._handshake_hb_timeout = 0.5
    mgr._transition_state(ConnectionState.CONNECTING)
    mgr._connecting_since = 1.0
    mgr._last_heartbeat = 0.0

    captured: dict[str, object] = {}

    def _fake_force_reconnect(*, reason: str | None = None) -> None:
        captured["reason"] = reason

    class _Counter:
        def __init__(self) -> None:
            self.count = 0

        def inc(self) -> None:
            self.count += 1

    counter = _Counter()
    mgr._m_handshake_failures = counter  # type: ignore[attr-defined]
    monkeypatch.setattr(mgr, "_force_reconnect", _fake_force_reconnect)
    mgr._detect_stuck_connecting(5.0)

    assert captured.get("reason") == "handshake_timeout"
    last_error = mgr.last_error()
    assert isinstance(last_error, HandshakeTimeoutError)
    assert mgr.connection_state() == ConnectionState.ERROR
    assert mgr._handshake_failures == 1
    assert ws.refreshed
    assert counter.count == 1
