from __future__ import annotations

"""Common helpers for data sources."""

from datetime import datetime


class BaseDataSource:
    """Mixin providing safe accessors for live data timestamps."""

    def last_tick_dt(self) -> datetime | None:
        """UTC datetime of last tick; ``None`` if unknown."""
        return getattr(self, "_last_tick_ts", None)

    def last_bar_open_ts(self) -> datetime | None:
        """UTC datetime of last completed bar OPEN; ``None`` if unknown."""
        return getattr(self, "_last_bar_open_ts", None)

    @property
    def timeframe_seconds(self) -> int:
        """Seconds per bar for the most recent fetch (default 60)."""
        return int(getattr(self, "_tf_seconds", 60))
