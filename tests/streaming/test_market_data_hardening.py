from __future__ import annotations

from datetime import datetime, time as dtime
import logging
from zoneinfo import ZoneInfo

from nifty_scalper_bot.streaming.market_data_hardening import (
    install_websocket_market_data_hardening,
)
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


class _BatchMdm:
    def __init__(self) -> None:
        self.processed: list[list[dict]] = []
        self.authoritative: list[list[dict]] = []

    def process_ticks(self, ticks: list[dict]) -> None:
        self.processed.append(list(ticks))

    def update_authoritative_ticks(self, ticks: list[dict]) -> None:
        self.authoritative.append(list(ticks))


def test_ws_batch_ingress_suppresses_legacy_callback_when_mdm_present() -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    callback_ticks: list[dict] = []
    manager = WebSocketManager(
        "api_key",
        "access_token",
        on_tick=lambda tick: callback_ticks.append(dict(tick)),
        trading_window_enabled=False,
    )
    mdm = _BatchMdm()
    manager._market_data_manager = mdm

    tick = {"instrument_token": 101, "last_price": 123.45}
    manager._on_ticks(object(), [tick])

    assert mdm.processed == [[tick]]
    assert mdm.authoritative == [[tick]]
    assert callback_ticks == []


def test_ws_legacy_callback_still_runs_without_mdm_batch_ingress() -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    callback_ticks: list[dict] = []
    manager = WebSocketManager(
        "api_key",
        "access_token",
        on_tick=lambda tick: callback_ticks.append(dict(tick)),
        trading_window_enabled=False,
    )

    tick = {"instrument_token": 202, "last_price": 234.56}
    manager._on_ticks(object(), [tick])

    assert callback_ticks == [tick]


def test_ticker_close_error_is_suppressed() -> None:
    class BadTicker:
        def close(self) -> None:
            raise RuntimeError("close failed")

    class DummyManager:
        _logger = logging.getLogger("test.websocket_hardening")

        def _build_ticker(self) -> BadTicker:
            return BadTicker()

    install_websocket_market_data_hardening(DummyManager)
    ticker = DummyManager()._build_ticker()

    assert ticker.close() is None


def test_trading_window_uses_configured_timezone_object() -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    manager = WebSocketManager(
        "api_key",
        "access_token",
        trading_window_enabled=True,
        trading_window_tz="Asia/Kolkata",
        trading_start=dtime(0, 0),
        trading_end=dtime(23, 59),
    )

    assert str(manager._trading_tz) == str(ZoneInfo("Asia/Kolkata"))
    if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() < 5:
        assert manager._is_within_trading_window() is True
