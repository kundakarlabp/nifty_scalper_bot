# src/backtesting/data_source.py
"""
A DataSource implementation for backtesting that reads from a CSV file.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.source import DataSource

logger = logging.getLogger(__name__)


class BacktestCsvSource(DataSource):
    """
    A data source for backtesting that simulates market data flow from a CSV file.

    This class provides historical OHLC data bar-by-bar, ensuring that the
    backtest does not suffer from look-ahead bias.
    """

    def __init__(self, csv_filepath: Path, symbol: str):
        if not csv_filepath.exists():
            raise FileNotFoundError(f"CSV file not found at: {csv_filepath}")

        self.symbol = symbol
        self._df = self._load_and_prepare_data(csv_filepath)
        self._current_index = 0
        self._max_index = len(self._df) - 1

    def _load_and_prepare_data(self, csv_filepath: Path) -> pd.DataFrame:
        """Loads and prepares the CSV data."""
        logger.info(f"Loading backtest data from {csv_filepath}...")
        df = pd.read_csv(csv_filepath)
        # Standardize column names
        df.rename(columns={
            "date": "datetime",
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }, inplace=True)

        if "datetime" not in df.columns:
            raise ValueError("CSV must contain a 'datetime' column.")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Loaded {len(df)} data points for backtesting.")
        return df

    def connect(self) -> None:
        """No connection needed for a CSV-based data source."""
        logger.info("Backtest data source initialized. No connection required.")

    def tick(self) -> bool:
        """
        Advances the simulation by one time step (bar).
        Returns False if there is no more data.
        """
        if self._current_index < self._max_index:
            self._current_index += 1
            return True
        return False

    @property
    def current_datetime(self) -> datetime:
        """Returns the datetime of the current bar."""
        return self._df.index[self._current_index]

    def get_spot_price(self, symbol: str) -> float | None:
        """
        Returns the closing price of the current bar as the spot price.
        """
        if symbol != self.symbol:
            logger.warning(f"Requested symbol {symbol} does not match data source symbol {self.symbol}")
            return None
        return self._df["close"].iloc[self._current_index]

    def fetch_ohlc(
        self, instrument_token: int, from_date: datetime, to_date: datetime, interval: str
    ) -> pd.DataFrame:
        """
        Returns a slice of the historical data up to the current simulation time.
        This simulates a point-in-time data request and prevents look-ahead bias.

        Note: The arguments (token, dates, interval) are ignored as the data is
        pre-loaded from the CSV. The method signature is for interface compatibility.
        """
        _ = instrument_token, from_date, to_date, interval  # Mark as unused

        # Return data up to and including the current bar
        end_index = self._current_index + 1
        return self._df.iloc[:end_index]
