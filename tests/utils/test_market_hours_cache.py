"""Tests for market hours cache behavior with override changes."""

from __future__ import annotations

import os
from datetime import datetime

from nifty_scalper_bot.utils import market_hours


def test_market_hours_cached_respects_runtime_override(monkeypatch) -> None:
    """Args: monkeypatch fixture. Returns: None. Raises: AssertionError."""

    def _fake_is_market_hours(allow_override: bool = True) -> bool:
        return os.getenv('SESSION_ALLOW_OUT_OF_HOURS', '').lower() == 'true'

    monkeypatch.setattr(market_hours, 'is_market_hours', _fake_is_market_hours)
    market_hours._cached_market_hours_check.cache_clear()

    monkeypatch.setenv('SESSION_ALLOW_OUT_OF_HOURS', 'false')
    assert market_hours.is_market_hours_cached() is False

    monkeypatch.setenv('SESSION_ALLOW_OUT_OF_HOURS', 'true')
    assert market_hours.is_market_hours_cached() is True

    market_hours._cached_market_hours_check.cache_clear()


def test_market_hours_blocks_weekend_even_inside_safe_clock(
    monkeypatch,
) -> None:
    class _SundayClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 19, 11, 53, tzinfo=tz)

    monkeypatch.setattr(market_hours, "datetime", _SundayClock)
    market_hours._cached_market_hours_check.cache_clear()

    assert market_hours.is_market_hours() is False
    assert market_hours.is_market_hours_cached() is False
