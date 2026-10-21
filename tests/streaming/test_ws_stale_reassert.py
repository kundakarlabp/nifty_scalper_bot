"""Regression coverage for single-symbol WS recovery and basket preservation."""

from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


class _RecordingTicker:
    MODE_FULL = "full"

    def __init__(self) -> None:
        self.subscribed: list[list[int]] = []
        self.unsubscribed: list[list[int]] = []
        self.modes: list[tuple[str, list[int]]] = []

    def subscribe(self, tokens: list[int]) -> None:
        self.subscribed.append(list(tokens))

    def unsubscribe(self, tokens: list[int]) -> None:
        self.unsubscribed.append(list(tokens))

    def set_mode(self, mode: str, tokens: list[int]) -> None:
        self.modes.append((mode, list(tokens)))


def _connected_ws() -> tuple[WebSocketManager, _RecordingTicker]:
    ws = WebSocketManager(
        "api_key", "access_token", tokens=[101, 202, 303], trading_window_enabled=False
    )
    ticker = _RecordingTicker()
    ws._ticker = ticker
    ws._connected.set()
    ws._schedule_blocking = lambda action: action()
    return ws, ticker


def test_reassert_one_stale_token_keeps_entire_active_basket() -> None:
    ws, ticker = _connected_ws()

    ws.resubscribe([202])

    assert ws.tokens_snapshot() == [101, 202, 303]
    assert ticker.subscribed == [[202]]
    assert ticker.modes == [("full", [202])]
    assert ticker.unsubscribed == []


def test_reassert_new_token_adds_without_removing_existing_symbols() -> None:
    ws, ticker = _connected_ws()

    ws.resubscribe([202, 404])

    assert ws.tokens_snapshot() == [101, 202, 303, 404]
    assert ticker.unsubscribed == []
    assert ticker.subscribed == [[404], [202]]
    assert ticker.modes == [("full", [404]), ("full", [202])]


def test_reassert_disconnected_token_replays_only_on_connect() -> None:
    ws, ticker = _connected_ws()
    ws._connected.clear()

    ws.resubscribe([202])

    assert ws.tokens_snapshot() == [101, 202, 303]
    assert ticker.subscribed == []
    assert ticker.unsubscribed == []


def test_stale_required_option_reasserts_ws_and_keeps_rest_recovery(monkeypatch) -> None:
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    mdm._last_valid_live_tick_mono[symbol] = now - 120.0
    mdm._required_symbol_since_mono[symbol] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    mdm._last_raw_ws_receive_mono = now
    mdm._last_ws_tick_mono = now
    recovered: list[tuple[str, str]] = []
    refreshed: list[tuple[str, str]] = []

    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_required_live_symbols", lambda: {symbol})
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_connected", lambda: True)
    monkeypatch.setattr(
        mdm,
        "_attempt_symbol_subscription_recovery",
        lambda sym, *, reason, now_mono: (
            recovered.append((sym, reason)) or {"attempted": True}
        ),
    )
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda sym, *, reason: refreshed.append((sym, reason)) or True,
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("isolated stale option must not restart healthy WS"),
    )

    mdm._check_zombie_ticks()

    assert recovered == [(symbol, "required_option_ws_tick_stale")]
    assert refreshed == [(symbol, "ws_symbol_stale_recovery")]
