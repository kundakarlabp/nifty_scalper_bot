"""
Defines the market data source abstraction and concrete implementations.

- DataSource: abstract interface used across the app/backtests.
- LiveKiteSource: Zerodha Kite-backed source for spot LTP and OHLC history.

Notes:
- Logging uses a safe default (INFO). If settings exposes a log level, it is
  picked up *without* assuming the attribute exists.
- Ensures returned OHLC DataFrames are indexed by naive datetimes (no tz),
  sorted ascending, with columns: open, high, low, close, volume (floats).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# Import settings late and guard attribute access
try:
    from src.config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # type: ignore[var-annotated]

# ---------------------------------------------------------------------
# Logging: pick a sane default; only use settings.* if present
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    level = logging.INFO
    try:
        # Accept log_level or LOG_LEVEL (string or int)
        lvl = None
        if settings is not None:
            lvl = getattr(settings, "log_level", None) or getattr(settings, "LOG_LEVEL", None)
        if isinstance(lvl, str):
            level = getattr(logging, lvl.upper(), logging.INFO)
        elif isinstance(lvl, int):
            level = lvl
    except Exception:
        # keep default INFO
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# -----------------------------------------------------------------------------
# Abstract interface
# -----------------------------------------------------------------------------
class DataSource(ABC):
    """Abstract Base Class for all market data sources."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source, if necessary."""
        raise NotImplementedError

    @abstractmethod
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Fetch the last traded price for a given symbol (e.g., 'NSE:NIFTY 50')."""
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLC data for a given instrument token."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Live Zerodha Kite implementation
# -----------------------------------------------------------------------------
class LiveKiteSource(DataSource):
    """
    Data source that fetches live and historical data from the Zerodha Kite API.

    LTP:
      - Call `get_spot_price('NSE:NIFTY 50')` for NIFTY 50 spot index.
    OHLC:
      - `fetch_ohlc(token, from_dt, to_dt, interval)`; interval should be one of:
        {'minute','3minute','5minute','10minute','15minute','30minute','60minute','day','week','month'}

    This class is intentionally defensive. It:
      - Verifies connection via `kite.profile()`
      - Normalizes datetime outputs to naive (tz removed) and sorted
      - Returns an empty DataFrame on recoverable errors (with logging)
    """

    _VALID_INTERVALS: set[str] = {
        "minute", "3minute", "5minute", "10minute", "15minute",
        "30minute", "60minute", "day", "week", "month",
    }

    def __init__(self, kite: Optional[KiteConnect]) -> None:
        if KiteConnect is not None and kite is not None and not isinstance(kite, KiteConnect):  # type: ignore[arg-type]
            raise TypeError("LiveKiteSource requires a KiteConnect instance (or None if kiteconnect not installed).")
        self.kite = kite
        self.is_connected = False

    # ------------------------ Connection ------------------------
    def connect(self) -> None:
        """
        Confirms the connection by fetching the user profile.
        The actual WebSocket connection (for ticks) is handled elsewhere.
        """
        if self.kite is None:
            self.is_connected = False
            logger.warning("LiveKiteSource.connect(): no Kite client; offline mode.")
            return

        try:
            # Lightweight auth check
            try:
                self.kite.margins()  # cheaper than profile in some cases
            except Exception:
                self.kite.profile()
            self.is_connected = True
            logger.info("LiveKiteSource: connected to Kite API.")
        except Exception as e:
            self.is_connected = False
            logger.error("LiveKiteSource: failed to connect to Kite API: %s", e, exc_info=True)
            raise ConnectionError("Could not connect to Kite API.") from e

    # ------------------------ LTP / Spot ------------------------
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the last traded price for `symbol`, e.g., 'NSE:NIFTY 50'.

        Returns `None` if not connected or unable to fetch.
        """
        if not self.is_connected or self.kite is None:
            logger.warning("LiveKiteSource: not connected; cannot fetch spot price.")
            return None

        try:
            # Kite expects a list of symbols; returns a dict keyed by symbol
            response = self.kite.ltp([symbol])
            data = response.get(symbol)
            if not data:
                logger.warning("LiveKiteSource: no LTP entry for %s. Raw response: %s", symbol, response)
                return None
            last_price = data.get("last_price")
            return float(last_price) if last_price is not None else None
        except Exception as e:
            logger.error("LiveKiteSource: error fetching spot price for %s: %s", symbol, e, exc_info=True)
            return None

    # ------------------------ OHLC History ------------------------
    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data from the Kite API.

        Returns a DataFrame with index=datetime (naive), columns:
          ['open','high','low','close','volume'] — float dtype
        On error, returns an empty DataFrame.
        """
        if not self.is_connected or self.kite is None:
            logger.warning("LiveKiteSource: not connected; cannot fetch OHLC data.")
            return pd.DataFrame()

        interval = str(interval).lower()
        if interval not in self._VALID_INTERVALS:
            logger.warning("LiveKiteSource: unsupported interval '%s'. Using 'minute' as fallback.", interval)
            interval = "minute"

        # Guard dates
        if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
            logger.error("LiveKiteSource: from_date and to_date must be datetime objects.")
            return pd.DataFrame()
        if from_date >= to_date:
            logger.warning("LiveKiteSource: from_date >= to_date; returning empty frame.")
            return pd.DataFrame()

        try:
            # Zerodha SDK returns list[dict]: date, open, high, low, close, volume, etc.
            records: Iterable[dict] = self.kite.historical_data(
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

            # Standardize datetime index
            col_dt = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
            if col_dt is None:
                logger.error("LiveKiteSource: historical response missing 'date'/'datetime' field.")
                return pd.DataFrame()

            df.rename(columns={col_dt: "datetime"}, inplace=True)

            if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                # Normalize to naive timestamps
                try:
                    df["datetime"] = df["datetime"].dt.tz_localize(None)
                except Exception:
                    pass
            else:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

            df = df.dropna(subset=["datetime"]).copy()
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

            # Ensure required numeric columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = float("nan")

            df = df[required_cols].astype(float)
            return df

        except Exception as e:
            logger.error(
                "LiveKiteSource: error fetching historical data for token %s: %s",
                instrument_token, e, exc_info=True
            )
            return pd.DataFrame()