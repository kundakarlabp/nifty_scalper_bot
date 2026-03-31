"""Tests for TimeBasedSizer expiry-day detection."""

from __future__ import annotations

import datetime as datetime_module

import pytest

from nifty_scalper_bot.risk import time_based_sizer as sizer_module
from nifty_scalper_bot.risk.time_based_sizer import TimeBasedSizer
from nifty_scalper_bot.utils.smart_symbol import WEEKLY_EXPIRY_WEEKDAY


class _FixedDatetime(datetime_module.datetime):
    """Datetime stub for deterministic weekday testing."""

    _fixed_now: datetime_module.datetime = datetime_module.datetime(2025, 1, 7, 10, 0)

    @classmethod
    def now(cls, tz: datetime_module.tzinfo | None = None) -> datetime_module.datetime:
        """Args: tz. Returns: datetime. Raises: None."""
        if tz is None:
            return cls._fixed_now
        return cls._fixed_now.replace(tzinfo=tz)


@pytest.mark.parametrize(
    ("weekday", "expected"),
    [
        (WEEKLY_EXPIRY_WEEKDAY, True),
        ((WEEKLY_EXPIRY_WEEKDAY + 1) % 7, False),
    ],
)
def test_time_based_sizer_expiry_day_detection(
    monkeypatch: pytest.MonkeyPatch,
    weekday: int,
    expected: bool,
) -> None:
    """Args: monkeypatch, weekday, expected. Returns: None. Raises: AssertionError."""
    anchor = datetime_module.datetime(2025, 1, 6 + weekday, 10, 0)
    _FixedDatetime._fixed_now = anchor
    monkeypatch.setattr(sizer_module, "datetime", _FixedDatetime)

    assert TimeBasedSizer()._is_expiry_day() is expected
