"""
Defines the market data source abstraction and concrete implementations.

- DataSource: abstract interface used across the app/backtests.
- LiveKiteSource: Zerodha Kite-backed source for spot LTP and OHLC history.

Conventions:
- Returned OHLC DataFrames have a naive datetime index (no tz), ascending,
  columns: ['open', 'high', 'low', 'close', 'volume'] as floats.
- Optional dependencies (kiteconnect) are guarded so imports remain safe.
- All broker calls are wrapped with a lightweight retry/backoff.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional, Type, TypeVar

import pandas as pd

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore

# Centralized settings (optional import guard)
try:
    from src.config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # type: ignore[var-annotated]


logger = logging.getLogger(__name__)
if not logger.handlers:
    level = logging.INFO
    try:
        if settings is not None:
            lvl = getattr(settings, "log_level", None) or getattr(settings, "LOG_LEVEL", None)
            if isinstance(lvl, str):
                level = getattr(logging, lvl.upper(), logging.INFO)
            elif isinstance(lvl, int):
                level = lvl
    except Exception:
        pass
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------
# Retry decorator (lightweight exponential backoff)
# ---------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., object])

def _retry(
    exceptions: tuple[Type[BaseException], ...] = (NetworkException, TokenException, InputException),
    tries: int = 3,
    base_delay: float = 0.3,
    max_delay: float = 2.0,
):
    def deco(fn: F) -> F:
        def wrapped(*args, **kwargs):  # type: ignore[misc]
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:  # type: ignore[misc]
                    attempt += 1
                    if attempt >= tries:
                        logger.error("Retry exhausted for %s: %s", fn.__name__, e)
                        raise
                    logger.warning("Transient error in %s (attempt %d/%d): %s", fn.__name__, attempt, tries, e)
                    time.sleep(min(delay, max_delay))
                    delay *= 2.0
        return wrapped  # type: ignore[return-value]
    return deco


# ---------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------
class DataSource(ABC):
    """Abstract interface for all data sources (live or backtest)."""

    @abstractmethod
    def connect(self) -> None:
        """Optional connectivity probe; should be a no-op for offline sources."""
        ...

    @abstractmethod
    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data.
        Returns DataFrame with index=datetime (naive), cols: open, high, low, close, volume (float).
        """
        ...

    @abstractmethod
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Fetch the last traded price (LTP) for a given instrument symbol."""
        ...


# ---------------------------------------------------------------------
# Live Kite Source
# ---------------------------------------------------------------------
class LiveKiteSource(DataSource):
    """
    Live data source using Zerodha KiteConnect API.

    - `connect()` probes session via margins() then profile()
    - `get_last_price(symbol)` uses `ltp([symbol])`
    - `fetch_ohlc(token, from, to, interval)` normalizes to standard OHLCV frame
    """

    def __init__(self, kite: KiteConnect | None) -> None:
        if KiteConnect is None:
            raise RuntimeError("kiteconnect is not installed.")
        if kite is None:
            raise RuntimeError("KiteConnect instance is required.")
        self._kite: KiteConnect = kite

    # --- helpers ---
    @staticmethod
    def _normalize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
        # standardize datetime column name
        col_dt = "date" if "date" in df.columns else ("datetime" if "datetime" in df.columns else None)
        if col_dt is None:
            logger.error("Historical response missing 'date'/'datetime' field.")
            return pd.DataFrame()

        df = df.rename(columns={col_dt: "datetime"}).copy()

        # ensure datetime dtype, drop tz info
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except Exception:
            pass  # already naive

        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

        # enforce required columns and dtypes
        required = ["open", "high", "low", "close", "volume"]
        for c in required:
            if c not in df.columns:
                df[c] = math.nan
        return df[required].astype(float)

    # --- API probes ---
    @_retry()
    def connect(self) -> None:
        """Probe the session; try margins() then profile()."""
        try:
            self._kite.margins()
        except Exception:
            self._kite.profile()

    # --- LTP ---
    @_retry()
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        LTP via Kite. NOTE: Kite expects a list of instruments.
        Returns None on error.
        """
        try:
            data = self._kite.ltp([symbol])
            if symbol in data:
                return float(data[symbol]["last_price"])
            # fallback: pick any value
            for _k, v in data.items():
                return float(v["last_price"])
            return None
        except Exception as e:
            logger.error("LTP fetch failed for %s: %s", symbol, e, exc_info=False)
            return None

    # --- Historical OHLC ---
    @_retry()
    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLC from Kite and normalize."""
        try:
            # Validate interval against typical Kite values we use
            if interval not in {"minute", "5minute"}:
                logger.warning("Unsupported interval '%s'; defaulting to 'minute'.", interval)
                interval = "minute"

            records = self._kite.historical_data(instrument_token, from_date, to_date, interval)
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            return self._normalize_ohlc_df(df)

        except Exception as e:
            logger.error(
                "Historical fetch failed for token %s: %s",
                instrument_token,
                e,
                exc_info=False,
            )
            return pd.DataFrame()