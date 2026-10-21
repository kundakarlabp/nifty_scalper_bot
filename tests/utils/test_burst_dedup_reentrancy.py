"""Burst dedup logging must not deadlock on reentrant asyncio error records."""

from __future__ import annotations

import logging

from nifty_scalper_bot.utils.logging import _BurstDedupFilter


def test_burst_filter_allows_reentrant_exception_record() -> None:
    """A nested record can enter while the outer filter owns its cache lock."""
    dedup = _BurstDedupFilter(window_sec=10.0)
    original = dedup._last_seen
    nested = logging.LogRecord(
        "asyncio", logging.ERROR, __file__, 1, "nested exception", (), None
    )
    outer = logging.LogRecord(
        "nifty", logging.ERROR, __file__, 1, "outer exception", (), None
    )

    class ReentrantCache(dict[str, float]):
        def __init__(self) -> None:
            super().__init__()
            self.entered = False

        def get(self, key: str, default: float = 0.0) -> float:
            if not self.entered:
                self.entered = True
                assert dedup.filter(nested) is True
            return super().get(key, default)

    cache = ReentrantCache()
    dedup._last_seen = cache
    try:
        assert dedup.filter(outer) is True
        assert cache.entered
        assert dedup.filter(nested) is False
        assert dedup.filter(outer) is False
    finally:
        dedup._last_seen = original
