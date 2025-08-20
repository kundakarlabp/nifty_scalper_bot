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
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import settings
from src.data.source import DataSource

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


_PathLike = Union[str, Path]


class BacktestCsvSource(DataSource):
    """
    A data source for backtesting that simulates market data flow from a CSV file.

    The CSV is read into memory once. Each `tick()` advances the internal cursor by 1 bar.
    All reads (spot/ohlc) are limited to the bars up to the current cursor to avoid
    look-ahead bias.
    """

    # Common header mappings we normalize to: datetime, open, high, low, close, volume
    _COL_MAP = {
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

    def __init__(self, csv_filepath: _PathLike, symbol: str) -> None:
        p = Path(csv_filepath)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found at: {p}")

        self.symbol = str(symbol).strip()
        self._df = self._load_and_prepare_data(p)
        if self._df.empty:
            raise ValueError(f"No rows found in {p}")

        self._current_index = 0
        self._max_index = len(self._df) - 1
        logger.info("BacktestCsvSource ready: %s rows, symbol=%s", len(self._df), self.symbol)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_and_prepare_data(self, csv_filepath: Path) -> pd.DataFrame:
        """Loads and prepares the CSV data."""
        logger.info("Loading backtest data from %s ...", csv_filepath)
        df = pd.read_csv(csv_filepath)

        # Standardize column names
        rename_map = {col: self._COL_MAP.get(col, col) for col in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # Require a datetime column
        if "datetime" not in df.columns:
            raise ValueError(
                "CSV must contain a 'datetime' column (or one of: date, Date, timestamp)."
            )

        # Parse datetime and set index
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
        df = df.dropna(subset=["datetime"]).copy()
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # Ensure OHLC numeric columns exist; coerce to numeric if present
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop any rows missing core OHLC; keep volume optional
        df = df.dropna(subset=["open", "high", "low", "close"]).copy()

        logger.info("Loaded %d rows for backtesting (index %s → %s).", len(df), df.index.min(), df.index.max())
        return df

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
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """
        Returns a slice of the historical data up to the current simulation time.

        Note:
        - Arguments are accepted for interface compatibility; this source uses
          its pre-loaded CSV data and simply returns all bars up to the cursor.
        - No look-ahead: the last row returned is the current bar.
        """
        _ = instrument_token, from_date, to_date, interval  # Unused by CSV source
        end = self._current_index + 1
        return self._df.iloc[:end].copy()

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
