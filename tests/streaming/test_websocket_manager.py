"""Tests for async KiteTicker websocket manager."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from nifty_scalper_bot.core import app as core_app
from nifty_scalper_bot.streaming import websocket_manager as ws_module


class FakeKiteTicker:
    """Args: api_key/access_token; Returns: fake; Raises: none."""

    MODE_FULL = "full"

    def __init__(self, api_key: str, access_token: str, reconnect: bool = True) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.on_connect: Callable[..., None] | None = None
        self.on_ticks: Callable[..., None] | None = None
        self.on_error: Callable[..., None] | None = None
        self.on_close: Callable[..., None] | None = None
        self.on_pong: Callable[..., None] | None = None
        self.reconnect = reconnect
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
async def test_connect_subscribes_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager(
        "k", "t", [111, 222], trading_window_enabled=False
    )

    await manager.connect()

    first_ticker = manager.ticker
    assert first_ticker.connect_calls == 1
    assert set(first_ticker.subscribed) == {111, 222}
    await manager.disconnect()


@pytest.mark.asyncio
async def test_ticks_dispatch_to_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    received: list[dict[str, Any]] = []

    async def _on_tick(tick: dict[str, Any]) -> None:
        received.append(tick)

    manager = ws_module.WebSocketManager(
        "k", "t", on_tick_callback=_on_tick, trading_window_enabled=False
    )
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    assert fake_ticker.on_ticks is not None
    fake_ticker.on_ticks(fake_ticker, [{"instrument_token": 99, "last_price": 12.4}])
    await asyncio.sleep(0)

    assert received == [{"instrument_token": 99, "last_price": 12.4}]
    await manager.disconnect()


@pytest.mark.asyncio
async def test_reconnect_scheduled_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager("k", "t", trading_window_enabled=False)
    manager._next_backoff_delay = lambda: 0.01
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    fake_ticker.on_error(fake_ticker, 1006, "drop")
    fake_ticker.on_close(fake_ticker, 1006, "drop")
    await asyncio.sleep(5.2)

    assert manager.ticker is not fake_ticker
    assert manager.ticker.connect_calls >= 1

    await manager.disconnect()
    latest_ticker = manager.ticker if manager._ticker is not None else None
    if latest_ticker is not None:
        calls_before = latest_ticker.connect_calls
        latest_ticker.on_close(latest_ticker, 1000, "manual")
        await asyncio.sleep(0.05)
        assert latest_ticker.connect_calls == calls_before


@pytest.mark.asyncio
async def test_watchdog_schedules_reconnect_for_missing_pong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager(
        "k",
        "t",
        heartbeat_interval_seconds=0.01,
        heartbeat_timeout_seconds=0.03,
        trading_window_enabled=False,
    )
    manager._next_backoff_delay = lambda: 0.01
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    await asyncio.sleep(5.2)

    assert manager.ticker is not fake_ticker
    await manager.disconnect()


@pytest.mark.asyncio
async def test_watchdog_suppresses_pong_reconnect_outside_trading_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager(
        "k",
        "t",
        heartbeat_interval_seconds=0.01,
        heartbeat_timeout_seconds=0.03,
    )
    manager._is_within_trading_window = lambda: False  # type: ignore[method-assign]
    await manager.connect()

    fake_ticker = manager.ticker
    assert isinstance(fake_ticker, FakeKiteTicker)
    await asyncio.sleep(0.12)

    assert manager.ticker is fake_ticker
    assert fake_ticker.connect_calls == 1
    await manager.disconnect()


@pytest.mark.asyncio
async def test_outside_trading_hours_log_is_mode_agnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    caplog.set_level("WARNING")

    manager = ws_module.WebSocketManager("k", "t")
    manager._is_within_trading_window = lambda: False  # type: ignore[method-assign]
    await manager.connect()
    await manager.disconnect()

    messages = [record.message for record in caplog.records]
    assert any("outside trading hours" in message for message in messages)
    assert all("paper-mode only" not in message for message in messages)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_repeated_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)

    manager = ws_module.WebSocketManager(
        "k",
        "t",
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown_seconds=0.05,
        trading_window_enabled=False,
    )

    manager._record_failure()
    manager._record_failure()

    snapshot = manager.health_snapshot()
    assert snapshot["circuit_open"] is True


def test_app_uses_module_level_time_module_only() -> None:
    source = inspect.getsource(core_app)
    assert "import time as time_module" in source
    assert "        import time as time_module" not in source


def test_polling_degrade_ignores_quote_age_when_ws_is_healthy() -> None:
    assert (
        core_app._polling_fallback_degraded(  # noqa: SLF001 - behavior contract test
            ws_ok=True,
            stale=False,
            lagging=False,
            quote_age_degraded=True,
        )
        is False
    )
    assert (
        core_app._polling_fallback_degraded(  # noqa: SLF001 - behavior contract test
            ws_ok=False,
            stale=False,
            lagging=False,
            quote_age_degraded=False,
        )
        is True
    )


def test_set_tokens_reconciles_added_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager("k", "t", [101], trading_window_enabled=False)
    manager._ticker = FakeKiteTicker("k", "t")
    manager._connected.set()

    changed = manager.set_tokens([101, 202, 303])

    assert changed is True
    assert manager._tokens == {101, 202, 303}  # noqa: SLF001
    assert set(manager._ticker.subscribed) == {202, 303}


def test_set_tokens_noop_for_same_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager("k", "t", [101, 202], trading_window_enabled=False)

    changed = manager.set_tokens([202, 101])

    assert changed is False
    assert manager._tokens == {101, 202}  # noqa: SLF001


def test_on_connect_replays_current_tokens_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager("k", "t", [11, 22], trading_window_enabled=False)
    ticker = FakeKiteTicker("k", "t")

    manager._on_connect(ticker, {"status": "ok"})  # noqa: SLF001

    assert set(ticker.subscribed) == {11, 22}
    assert set(manager._tokens) == {11, 22}  # noqa: SLF001


def test_subscribe_tokens_wrapper_unions_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "KiteTicker", FakeKiteTicker)
    manager = ws_module.WebSocketManager("k", "t", [10], trading_window_enabled=False)

    changed = manager.add_tokens([20])
    assert changed is True
    manager.subscribe_tokens([20, 30])

    assert manager._tokens == {10, 20, 30}  # noqa: SLF001
