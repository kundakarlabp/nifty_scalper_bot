# src/data/source.py
"""
Defines the data source abstraction for fetching market data.
This allows the application to seamlessly switch between live data from an API
and historical data from a file for backtesting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

# Optional import to avoid hard crash when kiteconnect is absent (e.g., CI/backtests)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # ImportError or anything else
    KiteConnect = None  # type: ignore

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for market data sources."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_spot_price(self, symbol: str) -> Optional[float]: ...

    @abstractmethod
    def get_historical_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "minute",
    ) -> pd.DataFrame: ...


class LiveKiteSource(DataSource):
    """Live data source using Zerodha KiteConnect."""

    def __init__(self, kite: object):
        if KiteConnect is None:
            raise ImportError("kiteconnect is not installed. Install with: pip install kiteconnect")
        if not isinstance(kite, KiteConnect):
            raise TypeError("LiveKiteSource expects a KiteConnect instance.")
        self.kite: KiteConnect = kite
        self.is_connected = False

    def connect(self) -> None:
        """Confirm connectivity by fetching profile."""
        try:
            self.kite.profile()
            self.is_connected = True
            logger.info("Successfully connected to Kite API.")
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to Kite API: {e}", exc_info=True)
            raise ConnectionError("Could not connect to Kite API.") from e

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Fetch last traded price for a symbol 'EXCHANGE:TRADINGSYMBOL'."""
        if not self.is_connected:
            logger.warning("Not connected to Kite. Cannot fetch spot price.")
            return None
        try:
            response = self.kite.ltp([symbol])
            price = response.get(symbol, {}).get("last_price")
            return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Error fetching spot price for {symbol}: {e}", exc_info=True)
            return None

    def get_historical_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "minute",
    ) -> pd.DataFrame:
        """Fetch OHLCV via KiteConnect.historical_data."""
        try:
            records = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                oi=False,
            )
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.rename(columns={"date": "datetime"}, inplace=True)
            # Ensure timezone-naive timestamps for consistency
            if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            df.set_index("datetime", inplace=True)

            required = ["open", "high", "low", "close", "volume"]
            return df[required]
        except Exception as e:
            logger.error("Error fetching historical data: %s", e, exc_info=True)
            return pd.DataFrame()
