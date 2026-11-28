from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.data.time_contract import normalize_market_tick_timestamp
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


class _BadCloseTicker:
    def close(self) -> None:
        raise RuntimeError("close failed")


class _TradingDayDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 15, 11, 0, tzinfo=tz)


class _GaneshChaturthiDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 14, 11, 0, tzinfo=tz)


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


def test_kite_sdk_naive_exchange_datetime_preserves_host_local_instant() -> None:
    manager = WebSocketManager("api_key", "access_token", trading_window_enabled=False)
    mdm = _BatchMdm()
    manager._market_data_manager = mdm

    # The SDK uses datetime.fromtimestamp(epoch), producing host-local naive
    # exchange timestamps. UTC hosts must NOT reinterpret them as IST.
    epoch = datetime(2026, 11, 20, 8, 10, tzinfo=timezone.utc).timestamp()
    naive = datetime.fromtimestamp(epoch)
    raw = {
        "instrument_token": 101,
        "last_price": 123.45,
        "exchange_timestamp": naive,
        "last_trade_time": naive,
    }
    manager._on_ticks(object(), [raw])

    delivered = mdm.processed[0][0]
    assert delivered is not raw
    assert raw["exchange_timestamp"].tzinfo is None
    assert delivered["exchange_timestamp"].tzinfo is not None
    assert delivered["exchange_timestamp"].timestamp() == epoch
    assert delivered["last_trade_time"].timestamp() == epoch
    assert normalize_market_tick_timestamp(delivered).timestamp.timestamp() == epoch
    assert mdm.authoritative[0][0] is delivered


def test_kite_sdk_already_aware_and_numeric_times_remain_unchanged() -> None:
    manager = WebSocketManager("api_key", "access_token", trading_window_enabled=False)
    mdm = _BatchMdm()
    manager._market_data_manager = mdm
    aware = datetime(2026, 11, 20, 13, 40, tzinfo=ZoneInfo("Asia/Kolkata"))
    aware_tick = {
        "instrument_token": 101,
        "last_price": 123.45,
        "exchange_timestamp": aware,
    }
    epoch_tick = {
        "instrument_token": 102,
        "last_price": 124.45,
        "exchange_timestamp": aware.timestamp(),
    }
    manager._on_ticks(object(), [aware_tick, epoch_tick])

    assert mdm.processed[0] == [aware_tick, epoch_tick]
    assert mdm.processed[0][0] is aware_tick
    assert mdm.processed[0][1] is epoch_tick


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


def test_native_ticker_builder_makes_close_best_effort() -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    manager = WebSocketManager(
        "api_key",
        "access_token",
        ticker_factory=_BadCloseTicker,
        trading_window_enabled=False,
    )

    ticker = manager._build_ticker()

    assert ticker.close() is None


@pytest.mark.asyncio
async def test_disconnect_suppresses_existing_ticker_close_error() -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    manager = WebSocketManager(
        "api_key",
        "access_token",
        trading_window_enabled=False,
    )
    manager._ticker = _BadCloseTicker()

    await manager.disconnect()

    assert manager._ticker is None


@pytest.mark.asyncio
async def test_reconnect_cleanup_continues_after_existing_ticker_close_error(monkeypatch) -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    manager = WebSocketManager(
        "api_key",
        "access_token",
        trading_window_enabled=False,
    )
    manager._ticker = _BadCloseTicker()
    manager._connected.clear()
    calls: list[str] = []

    async def no_sleep(_: float) -> None:
        return None

    async def fake_connect_once(reason: str) -> None:
        calls.append(reason)
        manager._connected.set()

    monkeypatch.setattr("nifty_scalper_bot.streaming.websocket_manager.asyncio.sleep", no_sleep)
    manager._connect_once = fake_connect_once

    await manager._reconnect_loop("unit_test")

    assert calls == ["reconnect"]


def test_trading_window_uses_configured_timezone_object(monkeypatch) -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    monkeypatch.setattr(
        "nifty_scalper_bot.streaming.websocket_manager.datetime",
        _TradingDayDateTime,
    )
    manager = WebSocketManager(
        "api_key",
        "access_token",
        trading_window_enabled=True,
        trading_window_tz="Asia/Kolkata",
        trading_start=dtime(0, 0),
        trading_end=dtime(23, 59),
    )

    assert str(manager._trading_tz) == str(ZoneInfo("Asia/Kolkata"))
    assert manager._is_within_trading_window() is True


def test_market_data_hardening_preserves_nse_holiday_guard(monkeypatch) -> None:
    install_websocket_market_data_hardening(WebSocketManager)
    monkeypatch.setattr(
        "nifty_scalper_bot.streaming.websocket_manager.datetime",
        _GaneshChaturthiDateTime,
    )
    manager = WebSocketManager(
        "api_key",
        "access_token",
        trading_window_enabled=True,
        trading_window_tz="Asia/Kolkata",
        trading_start=dtime(0, 0),
        trading_end=dtime(23, 59),
    )

    assert manager._is_within_trading_window() is False


def test_websocket_hardening_installer_is_verification_only() -> None:
    before = WebSocketManager._on_ticks

    install_websocket_market_data_hardening(WebSocketManager)

    assert WebSocketManager._on_ticks is before
    assert WebSocketManager._market_data_hardening_installed is True
