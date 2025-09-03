# src/backtesting/data_source.py
"""
CSV-backed DataSource for backtesting.

- Loads historical OHLC[V] data from a CSV.
- Standardizes common column names and enforces a DatetimeIndex.
- Advances bar-by-bar with `tick()` without look-ahead.
- Exposes point-in-time OHLC history via `fetch_ohlc()` (up to current bar).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import settings
from src.data.source import DataSource
from src.utils.logging_tools import RateLimitFilter
from src.backtesting.synth import make_synth_1m

logger = logging.getLogger(__name__)
logging.getLogger().addFilter(RateLimitFilter(interval=120.0))
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


_PathLike = Union[str, Path]

# Common header mappings we normalize to: datetime, open, high, low, close, volume
COL_MAP = {
    "date": "datetime",
    "Date": "datetime",
    "Datetime": "datetime",
    "timestamp": "datetime",
    "Timestamp": "datetime",
    "open": "open",
    "Open": "open",
    "high": "high",
    "High": "high",
    "low": "low",
    "Low": "low",
    "close": "close",
    "Close": "close",
    "adj_close": "close",
    "Adj Close": "close",
    "volume": "volume",
    "Volume": "volume",
    "vol": "volume",
}


def load_and_prepare_data(csv_path: _PathLike) -> pd.DataFrame:
    """Return a prepared DataFrame, synthesizing data if needed."""
    p = Path(csv_path)
    needs_synth = (not p.exists()) or (p.stat().st_size < 200)
    if needs_synth:
        df = make_synth_1m(
            start=(
                datetime.now(ZoneInfo("Asia/Kolkata"))
                .replace(second=0, microsecond=0)
                .replace(tzinfo=None)
                - timedelta(minutes=600)
            ),
            minutes=600,
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p)
    else:
        try:
            df = pd.read_csv(p)
        except EmptyDataError:
            df = make_synth_1m(
                start=(
                    datetime.now(ZoneInfo("Asia/Kolkata"))
                    .replace(second=0, microsecond=0)
                    .replace(tzinfo=None)
                    - timedelta(minutes=600)
                ),
                minutes=600,
            )
            df.to_csv(p)
        else:
            min_rows = int(getattr(settings.strategy, "min_bars_for_signal", 20))
            if df.empty or len(df.columns) < 4 or len(df) < min_rows:
                df = make_synth_1m(
                    start=(
                        datetime.now(ZoneInfo("Asia/Kolkata"))
                        .replace(second=0, microsecond=0)
                        .replace(tzinfo=None)
                        - timedelta(minutes=600)
                    ),
                    minutes=600,
                )
                df.to_csv(p)

    rename_map = {col: COL_MAP.get(col, col) for col in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

    if df.index.tz is None:
        df = df.tz_localize("Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT")

    cols = ["open", "high", "low", "close", "volume"]
    return df[cols]


class BacktestCsvSource(DataSource):
    """
    A data source for backtesting that simulates market data flow from a CSV file.

    The CSV is read into memory once. Each `tick()` advances the internal cursor by 1 bar.
    All reads (spot/ohlc) are limited to the bars up to the current cursor to avoid
    look-ahead bias.
    """

    _COL_MAP = COL_MAP

    def __init__(self, csv_filepath: _PathLike, symbol: str) -> None:
        self.symbol = str(symbol).strip()
        self._df = load_and_prepare_data(csv_filepath)
        if self._df.empty:
            raise ValueError(f"No rows found in {csv_filepath}")

        self._current_index = 0
        self._max_index = len(self._df) - 1
        logger.info(
            "BacktestCsvSource ready: %s rows, symbol=%s", len(self._df), self.symbol
        )

    # --------------------------------------------------------------------- #
    # DataSource interface
    # --------------------------------------------------------------------- #
    def connect(self) -> None:
        """No remote connection needed for a CSV-based data source."""
        logger.debug("Backtest data source initialized. No connection required.")

    def tick(self) -> bool:
        """
        Advances the simulation by one time step (bar).
        Returns False if there is no more data to advance.
        """
        if self._current_index < self._max_index:
            self._current_index += 1
            return True
        return False

    @property
    def current_datetime(self) -> datetime:
        """Returns the datetime of the current bar."""
        return self._df.index[self._current_index].to_pydatetime()

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Returns the closing price of the current bar as the spot price.
        If the requested symbol differs from this source's symbol, returns None.
        """
        if str(symbol).strip() != self.symbol:
            logger.warning(
                "Requested symbol %s does not match data source symbol %s",
                symbol, self.symbol,
            )
            return None
        return float(self._df["close"].iloc[self._current_index])

    def fetch_ohlc(
        self,
        token: int,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Return historical bars up to the current simulation time.

        Parameters are accepted for interface compatibility with other
        ``DataSource`` implementations.  The CSV-backed source ignores them and
        simply returns all rows up to the current cursor position.

        No look-ahead is performed; the last row corresponds to the current bar.
        """
        _ = token, start, end, timeframe  # Unused by CSV source
        end_idx = self._current_index + 1
        return self._df.iloc[:end_idx].copy()

    # --------------------------------------------------------------------- #
    # Convenience helpers (useful in tests/backtester)
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        """Reset the cursor to the first bar."""
        self._current_index = 0

    def is_finished(self) -> bool:
        """True if no further tick is possible."""
        return self._current_index >= self._max_index

    def current_bar(self) -> pd.Series:
        """Return the current row (bar)."""
        return self._df.iloc[self._current_index].copy()
