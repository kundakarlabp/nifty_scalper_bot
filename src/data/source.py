# src/data/source.py
"""
Live market data source abstraction.

- DataSource: ABC used across the app.
- LiveKiteSource: Zerodha Kite-backed source for LTP and OHLC.

Design:
- Logging falls back to INFO; honors settings.log_level if present.
- OHLC DataFrame: index=naive datetime (ascending); columns: open, high, low, close, volume (float).
- Narrow exception handling for Kite errors; graceful fallbacks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

# Optional imports to keep module import-safe
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    class NetworkException(Exception): ...  # type: ignore
    class TokenException(Exception): ...    # type: ignore
    class InputException(Exception): ...    # type: ignore

from src.config import settings

# ── logging ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    level = getattr(logging, str(getattr(settings, "log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ── interface ───────────────────────────────────────────────────────
class DataSource(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def get_spot_price(self, symbol: str) -> Optional[float]: ...
    @abstractmethod
    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame: ...


# ── Zerodha live implementation ─────────────────────────────────────
class LiveKiteSource(DataSource):
    _VALID_INTERVALS: set[str] = {
        "minute", "3minute", "5minute", "10minute", "15minute",
        "30minute", "60minute", "day", "week", "month",
    }

    def __init__(self, kite: Optional[KiteConnect]) -> None:  # type: ignore[name-defined]
        if KiteConnect is not None and kite is not None and not isinstance(kite, KiteConnect):
            raise TypeError("LiveKiteSource requires a KiteConnect instance (or None).")
        self.kite = kite
        self.is_connected = False

    def connect(self) -> None:
        """Verify session by calling a lightweight endpoint."""
        if self.kite is None:
            self.is_connected = False
            logger.warning("LiveKiteSource.connect(): no Kite client; offline mode.")
            return
        try:
            try:
                self.kite.margins()
            except Exception:
                self.kite.profile()
            self.is_connected = True
            logger.info("LiveKiteSource: connected to Kite API.")
        except (NetworkException, TokenException, InputException) as e:
            self.is_connected = False
            logger.error("LiveKiteSource: connect failed: %s", e)
            raise ConnectionError("Could not connect to Kite API.") from e

    def get_spot_price(self, symbol: str) -> Optional[float]:
        if not self.is_connected or self.kite is None:
            logger.warning("LiveKiteSource: not connected; cannot fetch spot price.")
            return None
        try:
            resp = self.kite.ltp([symbol])
            data = resp.get(symbol)
            if not data:
                logger.warning("LiveKiteSource: no LTP entry for %s. Raw: %s", symbol, resp)
                return None
            last_price = data.get("last_price")
            return float(last_price) if last_price is not None else None
        except (NetworkException, TokenException, InputException) as e:
            logger.error("LiveKiteSource: LTP failed for %s: %s", symbol, e)
            return None

    def fetch_ohlc(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        if not self.is_connected or self.kite is None:
            logger.warning("LiveKiteSource: not connected; cannot fetch OHLC.")
            return pd.DataFrame()

        interval = (interval or "minute").lower()
        if interval not in self._VALID_INTERVALS:
            logger.warning("LiveKiteSource: unsupported interval '%s' → 'minute'.", interval)
            interval = "minute"

        if not isinstance(from_date, datetime) or not isinstance(to_date, datetime) or from_date >= to_date:
            logger.warning("LiveKiteSource: invalid date range; returning empty frame.")
            return pd.DataFrame()

        try:
            records: Iterable[dict] = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
                oi=False,
            )
        except (NetworkException, TokenException, InputException) as e:
            logger.error("LiveKiteSource: historical_data failed (%s): %s", instrument_token, e)
            return pd.DataFrame()

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Standardize datetime index
        dt_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
        if dt_col is None:
            logger.error("LiveKiteSource: historical response missing datetime column.")
            return pd.DataFrame()
        df.rename(columns={dt_col: "datetime"}, inplace=True)

        if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            try:
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            except Exception:
                pass
        else:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

        # Ensure required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = float("nan")
        return df[["open", "high", "low", "close", "volume"]].astype(float)
