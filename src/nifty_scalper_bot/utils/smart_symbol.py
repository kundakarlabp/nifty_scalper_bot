# src/nifty_scalper_bot/utils/smart_symbol.py
#
# Legacy CE/PE symbol-building functions were removed. This module retains only
# pure date/calendar helpers used by risk, expiry and orchestration code.

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.nse_calendar import NSE_FO_HOLIDAYS_BY_YEAR, is_holiday

IST = ZoneInfo("Asia/Kolkata")
WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (0=Mon, 1=Tue, ...)

# Backward-compatible set used by existing imports and diagnostics.
NSE_HOLIDAYS = {
    holiday
    for holidays in NSE_FO_HOLIDAYS_BY_YEAR.values()
    for holiday in holidays
}


def is_nse_trading_day(day: date) -> bool:
    """Return whether ``day`` is a weekday and not an NSE F&O holiday."""
    return day.weekday() < 5 and not is_holiday(day)


def next_nse_trading_day(start_day: date) -> date:
    """Return the first NSE trading day on or after ``start_day``."""
    day = start_day
    for _ in range(14):
        if is_nse_trading_day(day):
            return day
        day += timedelta(days=1)
    return day


def now_ist() -> datetime:
    """Return the current datetime in IST."""
    return datetime.now(IST)


def get_actual_expiry_date(start_date: date, target_weekday: int) -> date:
    """Return the next target weekday, moved backward over closed sessions."""
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    expiry = start_date + timedelta(days=days_ahead)
    while not is_nse_trading_day(expiry):
        expiry -= timedelta(days=1)
    return expiry


def next_weekday(start_date: date, target_weekday: int) -> date:
    """Return the holiday-adjusted target weekday on or after ``start_date``."""
    return get_actual_expiry_date(start_date, target_weekday)


__all__ = [
    "IST",
    "NSE_HOLIDAYS",
    "WEEKLY_EXPIRY_WEEKDAY",
    "get_actual_expiry_date",
    "is_nse_trading_day",
    "next_nse_trading_day",
    "next_weekday",
    "now_ist",
]
