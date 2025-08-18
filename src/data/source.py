# src/data/source.py
"""
DataSource abstraction + LiveKiteSource implementation.

- get_historical_ohlc(...) returns a pandas DataFrame indexed by 'datetime'
  with columns: open, high, low, close, volume
- Lightweight rate-limiter + single retry on Kite rate errors
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore


class DataSource:
    """Interface for live/backfill data sources."""
    def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_historical_ohlc(
        self,
        instrument_token: int,
        start: datetime,
        end: datetime,
        interval: str = "minute",
    ) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


class LiveKiteSource(DataSource):
    """
    Zerodha Kite-backed source.

    Notes:
    - interval accepts: "minute", "3minute", "5minute", "15minute", "day" etc.
    - DataFrame is guaranteed to have ['open','high','low','close','volume'].
    """

    def __init__(self, kite: Optional[Any]):
        self.kite: Optional[Any] = kite
        self._min_interval_sec: float = 0.40  # throttle API a bit
        self._last_call_ts: float = 0.0

    # ---------- plumbing ----------

    def connect(self) -> None:
        if KiteConnect is None:
            raise RuntimeError("kiteconnect not installed. pip install kiteconnect")
        if self.kite is None:
            raise RuntimeError("Kite client is None. Provide an authenticated KiteConnect instance.")
        logger.info("LiveKiteSource connected.")

    def _rate_limited(self, func, *args, **kwargs):
        now = time.time()
        since = now - self._last_call_ts
        if since < self._min_interval_sec:
            time.sleep(self._min_interval_sec - since)

        try:
            out = func(*args, **kwargs)
            self._last_call_ts = time.time()
            return out
        except Exception as e:
            msg = str(e).lower()
            if "too many" in msg or "rate" in msg:
                logger.warning("Rate limit hit; retrying in 1.5s...")
                time.sleep(1.5)
                out = func(*args, **kwargs)
                self._last_call_ts = time.time()
                return out
            raise

    # ---------- public ----------

    def get_historical_ohlc(
        self,
        instrument_token: int,
        start: datetime,
        end: datetime,
        interval: str = "minute",
    ) -> pd.DataFrame:
        if self.kite is None:
            logger.error("get_historical_ohlc: no Kite client")
            return pd.DataFrame()

        # Zerodha expects specific interval strings; pass through if already valid
        valid = {
            "minute", "3minute", "5minute", "10minute", "15minute",
            "30minute", "60minute", "day"
        }
        api_interval = interval if interval in valid else "minute"

        try:
            raw = self._rate_limited(
                self.kite.historical_data,  # type: ignore[attr-defined]
                instrument_token,
                start,
                end,
                api_interval,
            )
        except Exception as e:
            logger.error("historical_data failed: %s", e, exc_info=True)
            return pd.DataFrame()

        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        # normalise datetime index
        dt_col = "date" if "date" in df.columns else ("datetime" if "datetime" in df.columns else None)
        if dt_col is None:
            logger.error("historical_data returned payload without datetime field")
            return pd.DataFrame()

        df.rename(columns={dt_col: "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False).dt.tz_localize(None)
        df.set_index("datetime", inplace=True)

        # keep standard OHLCV; fill missing volume if absent
        out_cols = []
        for c in ("open", "high", "low", "close"):
            if c not in df.columns:
                logger.error("historical_data missing column: %s", c)
                return pd.DataFrame()
            out_cols.append(c)

        if "volume" in df.columns:
            out_cols.append("volume")
        else:
            df["volume"] = 0
            out_cols.append("volume")

        return df[out_cols].sort_index()

    # Convenience wrapper if you need raw LTP (not used in runner)
    def ltp(self, symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
        if self.kite is None:
            return {}
        try:
            return self._rate_limited(self.kite.ltp, list(symbols))  # type: ignore[attr-defined]
        except Exception as e:
            logger.error("ltp() failed: %s", e)
            return {}
