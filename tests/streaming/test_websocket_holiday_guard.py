from __future__ import annotations

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager
from nifty_scalper_bot.utils import runtime_session_guards


class _FixedHolidayClock(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 6, 26, 11, 52)
        return value.replace(tzinfo=tz) if tz is not None else value


class _FixedOpenClock(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 6, 29, 11, 52)
        return value.replace(tzinfo=tz) if tz is not None else value


def _manager(*, window_enabled: bool = True) -> WebSocketManager:
    manager = object.__new__(WebSocketManager)
    manager._trading_window_enabled = window_enabled
    manager._trading_tz = ZoneInfo("Asia/Kolkata")
    manager._trading_start = dtime(9, 15)
    manager._trading_end = dtime(15, 30)
    manager._logger = __import__("logging").getLogger(__name__)
    return manager


def test_websocket_window_is_closed_on_nse_holiday(monkeypatch) -> None:
    monkeypatch.setattr(runtime_session_guards, "datetime", _FixedHolidayClock)
    assert _manager()._is_within_trading_window() is False


def test_websocket_window_remains_open_on_regular_weekday(monkeypatch) -> None:
    monkeypatch.setattr(runtime_session_guards, "datetime", _FixedOpenClock)
    assert _manager()._is_within_trading_window() is True


def test_disabled_transport_window_preserves_existing_override(monkeypatch) -> None:
    monkeypatch.setattr(runtime_session_guards, "datetime", _FixedHolidayClock)
    assert _manager(window_enabled=False)._is_within_trading_window() is True
