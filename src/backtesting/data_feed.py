from __future__ import annotations

"""Minimal spot data feed for backtests."""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime

import pandas as pd


@dataclass
class SpotFeed:
    """In-memory 1-minute OHLC feed."""

    df: pd.DataFrame
    tz: ZoneInfo

    @classmethod
    def from_csv(
        cls,
        path: str,
        tz: str = "Asia/Kolkata",
        parse_dates: list[str] | None = None,
    ) -> "SpotFeed":
        """Load a CSV of OHLCV data.

        The CSV must contain a ``timestamp`` column along with ``open``, ``high``,
        ``low``, ``close`` and ``volume`` columns. The timestamp is interpreted
        as naive local time (no timezone).
        """

        parse_dates = parse_dates or ["timestamp"]
        df = pd.read_csv(path)
        ts = pd.to_datetime(df[parse_dates[0]], utc=False, infer_datetime_format=True)
        df.index = ts.dt.tz_localize(None)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].sort_index()
        return cls(df=df, tz=ZoneInfo(tz))

    def window(self, start: Optional[str], end: Optional[str]) -> "SpotFeed":
        """Return a new feed limited to the provided window."""

        df = self.df
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return SpotFeed(df=df, tz=self.tz)

    def iter_bars(self) -> Iterator[Tuple[datetime, float, float, float, float, float]]:
        """Yield each bar as ``(ts, open, high, low, close, volume)``."""

        for ts, row in self.df.iterrows():
            yield ts, float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume or 0.0)
