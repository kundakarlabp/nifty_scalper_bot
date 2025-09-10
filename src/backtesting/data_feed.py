from __future__ import annotations

"""Minimal spot data feed for backtests."""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Tuple
from zoneinfo import ZoneInfo

import logging
import pandas as pd

logger = logging.getLogger(__name__)

_TS_ALIASES = ["timestamp", "datetime", "date", "time"]

_OHLC_MAP = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj close", "adj_close", "adjclose"],
    "volume": ["volume", "vol", "qty"],
}


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp/OHLC columns from various aliases."""

    def _clean(s: str) -> str:
        return s.lower().replace(" ", "").replace("/", "").replace("-", "").replace("_", "")

    norm = {c: _clean(c) for c in df.columns}

    ts_col: str | None = None
    for c in df.columns:
        key = norm[c]
        if key in {"timestamp", "datetime"}:
            ts_col = c
            break

    if ts_col is None and {"date", "time"}.issubset(set(norm.values())):
        date_col = next(cc for cc in df.columns if norm[cc] == "date")
        time_col = next(cc for cc in df.columns if norm[cc] == "time")
        df["timestamp"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str)
        )
        ts_col = "timestamp"

    if ts_col is None:
        for c in df.columns:
            if norm[c] in {"date", "time"}:
                ts_col = c
                break

    if ts_col is None:
        raise KeyError(f"No timestamp-like column found. Available: {list(df.columns)}")

    col = df[ts_col]
    if pd.api.types.is_numeric_dtype(col):
        df["timestamp"] = pd.to_datetime(col, unit="s", errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(col, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    rename: dict[str, str] = {}
    for target, aliases in _OHLC_MAP.items():
        alias_keys = {_clean(a) for a in aliases}
        for c in df.columns:
            if c == "timestamp" or c in rename:
                continue
            if _clean(c) in alias_keys:
                rename[c] = target
                break
    df = df.rename(columns=rename)

    missing = [k for k in ["open", "high", "low", "close"] if k not in df.columns]
    if missing:
        logger.error("Missing OHLC columns after normalization: %s", missing)
        raise KeyError(f"Missing OHLC columns after normalization: {missing}. Have: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
    return df


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

        The CSV must contain a timestamp column along with open, high, low,
        close and volume columns. Timestamp and OHLC columns are detected
        case-insensitively and may appear under common aliases such as
        ``Date``/``Time`` or ``Adj Close``.
        """

        del parse_dates  # unused, retained for compatibility
        df = pd.read_csv(path)
        df = _normalize_cols(df)
        df.index = df["timestamp"].dt.tz_localize(None)
        df = df.drop(columns=["timestamp"])
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
