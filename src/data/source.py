# src/data/source.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore


# ------------------------------- Base Interface --------------------------------

class DataSource:
    """Minimal interface used by the runner/executor."""

    def connect(self) -> None:
        return

    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """Return OHLC DataFrame with columns: open, high, low, close, volume."""
        raise NotImplementedError

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        """Return LTP for a trading symbol (e.g., 'NSE:NIFTY 50') or token int."""
        raise NotImplementedError


# ------------------------------- Helpers --------------------------------

def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


_INTERVAL_MAP = {
    "minute": "minute", "1minute": "minute", "1m": "minute",
    "5minute": "5minute", "5m": "5minute",
}

@dataclass
class _CacheEntry:
    df: pd.DataFrame
    ts: float


class _TTLCache:
    def __init__(self, ttl_sec: float = 5.0) -> None:
        self._ttl = ttl_sec
        self._data: Dict[Tuple[int, str], _CacheEntry] = {}

    def get(self, token: int, interval: str) -> Optional[pd.DataFrame]:
        key = (int(token), interval)
        ent = self._data.get(key)
        if not ent:
            return None
        if time.time() - ent.ts > self._ttl:
            self._data.pop(key, None)
            return None
        return ent.df

    def set(self, token: int, interval: str, df: pd.DataFrame) -> None:
        key = (int(token), interval)
        self._data[key] = _CacheEntry(df=df, ts=time.time())


def _safe_dataframe(rows: Any) -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows or [])
        if df.empty:
            return pd.DataFrame()
        # Normalize column names
        rename_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename_map)
        # Kite historical returns 'date' timestamps; ensure timezone-naive
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.tz_localize(None)
            df = df.set_index("date")
        # Ensure all needed columns exist
        need = {"open", "high", "low", "close"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        # Volume may be missing for some instruments → default 0
        if "volume" not in df.columns:
            df["volume"] = 0
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        log.warning("Failed to normalize OHLC frame: %s", e)
        return pd.DataFrame()


def _retry(fn, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    delay = base_delay
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException) as e:
            last = e
            if i == tries - 1:
                raise
            time.sleep(delay); delay *= 2.0
        except Exception as e:  # other unexpected
            last = e
            if i == tries - 1:
                raise
            time.sleep(delay); delay *= 2.0
    if last:
        raise last


# ------------------------------- LiveKiteSource --------------------------------

class LiveKiteSource(DataSource):
    """
    Reads candles via Kite's historical API + LTP for quick checks.
    Adds a tiny TTL cache to stay under rate limits during frequent ticks/diags.
    """

    def __init__(self, kite: Optional["KiteConnect"]) -> None:
        self.kite = kite
        self._cache = _TTLCache(ttl_sec=4.0)  # tiny per-token cache

    def connect(self) -> None:
        if not self.kite:
            log.info("LiveKiteSource: kite is None (shadow mode).")
        else:
            log.info("LiveKiteSource: connected to Kite.")

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        if not self.kite:
            return None
        try:
            if isinstance(symbol_or_token, int):
                data = _retry(self.kite.ltp, [symbol_or_token], tries=2)
                for _, v in (data or {}).items():
                    return float(v.get("last_price"))
                return None
            else:
                sym = str(symbol_or_token)
                data = _retry(self.kite.ltp, [sym], tries=2)
                if sym in data:
                    return float(data[sym].get("last_price"))
                for _, v in (data or {}).items():
                    return float(v.get("last_price"))
                return None
        except Exception as e:
            log.debug("get_last_price failed for %s: %s", symbol_or_token, e)
            return None

    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """
        Primary path: Kite historical_data
        Fallback: if empty, synthesize one 'bar' from current LTP to avoid a hard stop.
        """
        if not self.kite:
            log.warning("LiveKiteSource.fetch_ohlc: kite is None.")
            return pd.DataFrame()

        interval = _INTERVAL_MAP.get(str(timeframe).lower())
        if not interval:
            log.warning("Unsupported timeframe '%s' → using 'minute'.", timeframe)
            interval = "minute"

        # Cache window key: (token, interval).
        cached = self._cache.get(token, interval)
        if cached is not None and not cached.empty:
            try:
                return cached.loc[(cached.index >= start) & (cached.index <= end)]
            except Exception:
                pass

        frm = pd.to_datetime(start).to_pydatetime()
        to = pd.to_datetime(end).to_pydatetime()

        try:
            rows = _retry(
                self.kite.historical_data,
                token, frm, to, interval, continuous=False, oi=False, tries=2,
            )
            df = _safe_dataframe(rows)
            if df.empty:
                log.warning(
                    "historical_data returned empty for token=%s interval=%s window=%s→%s",
                    token, interval, start, end,
                )
                ltp = self.get_last_price(token)
                if ltp is not None:
                    ts = _now_ist_naive().replace(second=0, microsecond=0)
                    df = pd.DataFrame(
                        {"open": [ltp], "high": [ltp], "low": [ltp], "close": [ltp], "volume": [0]},
                        index=[ts],
                    )
            if df is not None:
                self._cache.set(token, interval, df)
            try:
                df = df.loc[(df.index >= start) & (df.index <= end)]
            except Exception:
                pass
            need = {"open", "high", "low", "close"}
            if df.empty or not need.issubset(df.columns):
                return pd.DataFrame()
            return df
        except Exception as e:
            log.error("fetch_ohlc failed token=%s interval=%s: %s", token, interval, e)
            return pd.DataFrame()