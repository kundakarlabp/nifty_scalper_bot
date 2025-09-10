from __future__ import annotations

"""Minimal spot data feed for backtests."""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Tuple
from zoneinfo import ZoneInfo

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

        A timestamp column is detected by looking for case-insensitive headers
        among ``timestamp``, ``date``, ``time`` or ``datetime`` (additional names
        may be provided via ``parse_dates``). OHLC headers are normalized
        case-insensitively with aliases such as ``Adj Close`` mapped to ``close``.
        If a ``volume`` column is missing, it is added and filled with zeros.
        The timestamp is interpreted as naive local time (no timezone).
        """

        df = pd.read_csv(path)

        lower_cols = {c.lower(): c for c in df.columns}
        candidates = [
            *(parse_dates or []),
            "timestamp",
            "date",
            "time",
            "datetime",
        ]
        ts_name: Optional[str] = None
        for name in (c.lower() for c in candidates):
            if name in lower_cols:
                ts_name = lower_cols[name]
                break
        if ts_name is None:
            raise ValueError("no timestamp column found")

        ts_col = df[ts_name]
        if pd.api.types.is_numeric_dtype(ts_col):
            ts = pd.to_datetime(ts_col, unit="s", utc=False)
        else:
            ts = pd.to_datetime(ts_col, utc=False, infer_datetime_format=True)
        df.index = ts.dt.tz_localize(None)

        df = df.rename(columns=str.lower)
        df = df.rename(columns={"adj close": "close", "adj_close": "close"})
        if "volume" not in df.columns:
            df["volume"] = 0
        df = df[["open", "high", "low", "close", "volume"]].sort_index()

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
            yield ts, float(row.open), float(row.high), float(row.low), float(
                row.close
            ), float(row.volume or 0.0)
