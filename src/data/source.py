# src/data/source.py
"""
Defines the data source abstraction for fetching market data.
This allows the application to seamlessly switch between live data from an API
and historical data from a file for backtesting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from kiteconnect import KiteConnect

from src.config import settings

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract Base Class for all market data sources."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source, if necessary."""
        raise NotImplementedError

    @abstractmethod
    def get_spot_price(self, symbol: str) -> float | None:
        """Fetch the last traded price for a given symbol."""
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlc(
        self, instrument_token: int, from_date: datetime, to_date: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch historical OHLC data for a given instrument token."""
        raise NotImplementedError


class LiveKiteSource(DataSource):
    """
    A data source that fetches live and historical data from the Zerodha Kite API.
    """

    def __init__(self, kite: KiteConnect):
        if not isinstance(kite, KiteConnect):
            raise TypeError("A valid KiteConnect instance is required.")
        self.kite = kite
        self.is_connected = False

    def connect(self) -> None:
        """
        Confirms the connection by fetching the user profile.
        The actual WebSocket connection is managed by the order executor/ticker.
        """
        try:
            self.kite.profile()
            self.is_connected = True
            logger.info("Successfully connected to Kite API.")
        except Exception as e:
            logger.error(f"Failed to connect to Kite API: {e}", exc_info=True)
            self.is_connected = False
            raise ConnectionError("Could not connect to Kite API.") from e

    def get_spot_price(self, symbol: str) -> float | None:
        """Fetches the last traded price for the NIFTY 50 spot index."""
        if not self.is_connected:
            logger.warning("Not connected to Kite. Cannot fetch spot price.")
            return None
        try:
            # The symbol should be in the format 'EXCHANGE:TRADINGSYMBOL'
            response = self.kite.ltp([symbol])
            if symbol in response and response[symbol]["last_price"]:
                return float(response[symbol]["last_price"])
            logger.warning(f"Could not get LTP for {symbol} from response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error fetching spot price for {symbol}: {e}", exc_info=True)
            return None

    def fetch_ohlc(
        self, instrument_token: int, from_date: datetime, to_date: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetches historical OHLC data from the Kite API."""
        if not self.is_connected:
            logger.warning("Not connected to Kite. Cannot fetch OHLC data.")
            return pd.DataFrame()
        try:
            records = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
                oi=False,
            )
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.rename(columns={"date": "datetime"}, inplace=True)
            # Ensure datetime is timezone-naive for consistency
            if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            df.set_index("datetime", inplace=True)

            required_cols = ["open", "high", "low", "close", "volume"]
            df = df[required_cols]
            df = df.astype(float)

            return df
        except Exception as e:
            logger.error(
                f"Error fetching historical data for token {instrument_token}: {e}",
                exc_info=True,
            )
            return pd.DataFrame()
