"""Canonical NSE equity-derivatives holiday calendar.

The built-in 2026 dates follow the NSE F&O trading-holiday circular. Runtime
add/remove overrides support exceptional exchange notices without weakening the
normal fail-closed session gate.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from typing import Mapping


NSE_FO_HOLIDAYS_BY_YEAR: Mapping[int, Mapping[date, str]] = {
    2026: {
        date(2026, 1, 26): "Republic Day",
        date(2026, 3, 3): "Holi",
        date(2026, 3, 26): "Shri Ram Navami",
        date(2026, 3, 31): "Shri Mahavir Jayanti",
        date(2026, 4, 3): "Good Friday",
        date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2026, 5, 1): "Maharashtra Day",
        date(2026, 5, 28): "Bakri Id",
        date(2026, 6, 26): "Muharram",
        date(2026, 9, 14): "Ganesh Chaturthi",
        date(2026, 10, 2): "Mahatma Gandhi Jayanti",
        date(2026, 10, 20): "Dussehra",
        date(2026, 11, 10): "Diwali-Balipratipada",
        date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
        date(2026, 12, 25): "Christmas",
    }
}


def _parse_dates(raw: str | None) -> frozenset[date]:
    parsed: set[date] = set()
    for item in str(raw or "").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            parsed.add(date.fromisoformat(value))
        except ValueError:
            continue
    return frozenset(parsed)


@lru_cache(maxsize=32)
def _runtime_overrides(
    additional_raw: str,
    removed_raw: str,
) -> tuple[frozenset[date], frozenset[date]]:
    return _parse_dates(additional_raw), _parse_dates(removed_raw)


def holiday_name(day: date) -> str | None:
    """Return the NSE F&O holiday name for ``day`` or ``None``."""
    additional, removed = _runtime_overrides(
        os.getenv("NSE_ADDITIONAL_HOLIDAYS", ""),
        os.getenv("NSE_REMOVED_HOLIDAYS", ""),
    )
    if day in removed:
        return None
    if day in additional:
        return "Exchange holiday (runtime override)"
    return NSE_FO_HOLIDAYS_BY_YEAR.get(day.year, {}).get(day)


def is_holiday(day: date) -> bool:
    """Return whether ``day`` is an NSE equity-derivatives holiday."""
    return holiday_name(day) is not None


def calendar_available(year: int) -> bool:
    """Return whether a built-in holiday calendar exists for ``year``."""
    return int(year) in NSE_FO_HOLIDAYS_BY_YEAR


__all__ = [
    "NSE_FO_HOLIDAYS_BY_YEAR",
    "calendar_available",
    "holiday_name",
    "is_holiday",
]
