from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from nifty_scalper_bot.core.app import _next_eod_flatten_time_ist

IST = ZoneInfo("Asia/Kolkata")


def test_eod_before_close_same_trading_day():
    now = datetime(2026, 7, 10, 14, 0)
    target = _next_eod_flatten_time_ist(now)
    assert target == datetime(2026, 7, 10, 15, 24, tzinfo=IST)
    assert target > now.replace(tzinfo=IST)


def test_eod_after_close_moves_to_next_trading_day():
    now = datetime(2026, 7, 9, 15, 25, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 10, 15, 24, tzinfo=IST)


def test_eod_utc_aware_converts_to_ist():
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)  # 13:30 IST
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 10, 15, 24, tzinfo=IST)


def test_eod_friday_after_close_skips_weekend():
    now = datetime(2026, 7, 10, 15, 25, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 13, 15, 24, tzinfo=IST)


def test_eod_saturday_skips_to_monday():
    now = datetime(2026, 7, 11, 10, 0, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 13, 15, 24, tzinfo=IST)


def test_eod_sunday_skips_to_monday():
    now = datetime(2026, 7, 12, 10, 0, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 13, 15, 24, tzinfo=IST)


def test_eod_nse_holiday_skips_to_next_valid_day(monkeypatch):
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app.is_nse_trading_day",
        lambda day: day.isoformat() != "2026-01-26" and day.weekday() < 5,
    )
    now = datetime(2026, 1, 26, 10, 0, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 1, 27, 15, 24, tzinfo=IST)


def test_eod_next_valid_trading_day_after_holiday_and_weekend(monkeypatch):
    closed = {"2026-07-10"}
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app.is_nse_trading_day",
        lambda day: day.isoformat() not in closed and day.weekday() < 5,
    )
    now = datetime(2026, 7, 10, 16, 0, tzinfo=IST)
    assert _next_eod_flatten_time_ist(now) == datetime(2026, 7, 13, 15, 24, tzinfo=IST)


def test_eod_timezone_conversion_correctness_no_past():
    now = datetime(2026, 7, 10, 9, 54, 1, tzinfo=timezone.utc)  # 15:24:01 IST
    target = _next_eod_flatten_time_ist(now)
    assert target == datetime(2026, 7, 13, 15, 24, tzinfo=IST)
    assert target > now.astimezone(IST)
