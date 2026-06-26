from __future__ import annotations

from datetime import datetime

from nifty_scalper_bot.utils import market_hours
from nifty_scalper_bot.utils.nse_calendar import holiday_name, is_holiday
from nifty_scalper_bot.utils.smart_symbol import is_nse_trading_day


def test_muharram_2026_is_closed_during_normal_market_clock(monkeypatch) -> None:
    now = datetime(2026, 6, 26, 11, 52, tzinfo=market_hours.IST)
    monkeypatch.setattr(market_hours, "_now_ist", lambda: now)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")

    assert holiday_name(now.date()) == "Muharram"
    assert is_holiday(now.date()) is True
    assert is_nse_trading_day(now.date()) is False
    assert market_hours.get_market_state() is market_hours.MarketState.CLOSED
    assert market_hours.get_runtime_market_mode() == "HOLIDAY"
    assert market_hours.get_market_session_state(now) == "closed"
    assert market_hours.is_market_open_session() is False
    assert market_hours.is_safe_entry_window() is False
    allowed, reason = market_hours.get_time_status()
    assert allowed is False
    assert reason == "Exchange holiday (Muharram)"


def test_regular_weekday_remains_open(monkeypatch) -> None:
    now = datetime(2026, 6, 29, 11, 52, tzinfo=market_hours.IST)
    monkeypatch.setattr(market_hours, "_now_ist", lambda: now)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")

    assert is_nse_trading_day(now.date()) is True
    assert market_hours.get_market_state() is market_hours.MarketState.OPEN
    assert market_hours.get_runtime_market_mode() == "OPEN"
    assert market_hours.is_market_open_session() is True
    assert market_hours.is_safe_entry_window() is True


def test_market_hours_cache_key_changes_across_calendar_days() -> None:
    first = datetime(2026, 6, 26, 11, 52, tzinfo=market_hours.IST)
    second = datetime(2026, 6, 29, 11, 52, tzinfo=market_hours.IST)

    assert market_hours._cache_minute_key(first) != market_hours._cache_minute_key(second)
