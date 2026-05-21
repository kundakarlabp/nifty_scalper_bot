from __future__ import annotations

from datetime import datetime
import importlib

from nifty_scalper_bot.utils import market_hours


class _FixedClock(datetime):
    _fixed: datetime

    @classmethod
    def now(cls, tz=None):
        return cls._fixed.astimezone(tz) if tz else cls._fixed


def _set_now(monkeypatch, dt: datetime) -> None:
    _FixedClock._fixed = dt
    monkeypatch.setattr(market_hours, "datetime", _FixedClock)


def _clear_caches() -> None:
    market_hours._cached_market_hours_check.cache_clear()
    market_hours._cached_time_status_check.cache_clear()


def test_safe_window_1517_true(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 17, tzinfo=market_hours.IST))
    _clear_caches()
    assert market_hours.is_safe_entry_window() is True


def test_safe_window_1526_false(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 26, tzinfo=market_hours.IST))
    assert market_hours.is_safe_entry_window() is False


def test_market_open_session_boundaries(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 17, tzinfo=market_hours.IST))
    assert market_hours.is_market_open_session() is True
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 31, tzinfo=market_hours.IST))
    assert market_hours.is_market_open_session() is False


def test_open_but_not_safe_at_0916(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 9, 16, tzinfo=market_hours.IST))
    assert market_hours.is_market_open_session() is True
    assert market_hours.is_safe_entry_window() is False


def test_safe_at_0921(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 9, 21, tzinfo=market_hours.IST))
    assert market_hours.is_safe_entry_window() is True


def test_time_status_messages(monkeypatch):
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 17, tzinfo=market_hours.IST))
    allowed, reason = market_hours.get_time_status()
    assert allowed is True
    assert reason == "Within safe entry window"
    _set_now(monkeypatch, datetime(2026, 5, 21, 15, 26, tzinfo=market_hours.IST))
    allowed, reason = market_hours.get_time_status()
    assert allowed is False
    assert "EOD safety cutoff" in reason


def test_safe_end_env(monkeypatch):
    monkeypatch.setenv("SAFE_END_TIME", "15:20")
    reloaded = importlib.reload(market_hours)
    monkeypatch.setattr(reloaded, "_now_ist", lambda: datetime(2026, 5, 21, 15, 21, tzinfo=reloaded.IST))
    assert reloaded.is_safe_entry_window() is False

    monkeypatch.setenv("SAFE_END_TIME", "15:25")
    reloaded = importlib.reload(reloaded)
    monkeypatch.setattr(reloaded, "_now_ist", lambda: datetime(2026, 5, 21, 15, 21, tzinfo=reloaded.IST))
    assert reloaded.is_safe_entry_window() is True


def test_live_mode_disables_offhours_override(monkeypatch):
    monkeypatch.setenv("ALLOW_OFFHOURS_TESTING", "true")
    monkeypatch.setenv("SESSION_ALLOW_OUT_OF_HOURS", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    assert market_hours.allow_offhours_testing_safe() is False
