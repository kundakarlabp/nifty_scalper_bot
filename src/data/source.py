# src/data/source.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Callable, List

import pandas as pd

log = logging.getLogger(__name__)

# Optional broker SDK (keep imports tolerant so paper mode works)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (  # type: ignore
        NetworkException,
        TokenException,
        InputException,
        DataException,
        GeneralException,
    )
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    # Collapse to base Exception so retry wrapper still works in paper mode
    NetworkException = TokenException = InputException = DataException = GeneralException = Exception  # type: ignore


# --------------------------------------------------------------------------------------
# Base Interface
# --------------------------------------------------------------------------------------
class DataSource:
    """Minimal interface used by StrategyRunner/OrderExecutor."""

    def connect(self) -> None:
        """Connect or noop."""
        return

    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """
        Return OHLC DataFrame with columns: open, high, low, close, volume.
        Index should be pandas Timestamps (timezone‑naive IST is fine).
        """
        raise NotImplementedError

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        """Return LTP for a trading symbol (e.g., 'NSE:NIFTY 50') or instrument token."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _now_ist_naive() -> datetime:
    """Current time in IST, timezone‑naive (to match most data frames we use)."""
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# Accept a few common aliases; default to 'minute'
_INTERVAL_MAP: Dict[str, str] = {
    "minute": "minute",
    "1minute": "minute",
    "1m": "minute",
    "3minute": "3minute",
    "3m": "3minute",
    "5minute": "5minute",
    "5m": "5minute",
    "10minute": "10minute",
    "10m": "10minute",
    "15minute": "15minute",
    "15m": "15minute",
}

@dataclass
class _CacheEntry:
    df: pd.DataFrame
    ts: float               # insertion timestamp (epoch seconds)
    window: Tuple[datetime, datetime]  # (start, end) that the DF covers


class _TTLCache:
    """
    Extremely simple per‑(token,interval) cache to soften historical_data pressure
    during frequent ticks or Telegram diagnostics. Keeps a tiny TTL.
    """
    def __init__(self, ttl_sec: float = 4.0) -> None:
        self._ttl = float(ttl_sec)
        self._data: Dict[Tuple[int, str], _CacheEntry] = {}

    def get(self, token: int, interval: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        key = (int(token), interval)
        ent = self._data.get(key)
        if not ent:
            return None
        if time.time() - ent.ts > self._ttl:
            self._data.pop(key, None)
            return None
        # If our cached window fully contains the requested window, subset it
        s0, e0 = ent.window
        if s0 <= start and e0 >= end:
            try:
                return ent.df.loc[(ent.df.index >= start) & (ent.df.index <= end)].copy()
            except Exception:
                # fall through to miss
                return None
        return None

    def set(self, token: int, interval: str, df: pd.DataFrame, start: datetime, end: datetime) -> None:
        key = (int(token), interval)
        self._data[key] = _CacheEntry(df=df.copy(), ts=time.time(), window=(start, end))


def _safe_dataframe(rows: Any) -> pd.DataFrame:
    """
    Normalize Kite historical rows to a canonical OHLCV frame.
    - lowercases columns
    - parses 'date' to naive datetime index
    - guarantees presence/order of: open, high, low, close, volume
    """
    try:
        df = pd.DataFrame(rows or [])
        if df.empty:
            return pd.DataFrame()

        # Normalize column names
        df = df.rename(columns={c: str(c).lower() for c in df.columns})

        # historical_data returns 'date' column; ensure timezone‑naive Timestamp index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.tz_localize(None)
            df = df.set_index("date")

        # Required columns
        need = {"open", "high", "low", "close"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        # Volume may not be present; default to 0 (int)
        if "volume" not in df.columns:
            df["volume"] = 0

        # Ensure column order and numeric types
        out = df[["open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna()
        return out

    except Exception as e:
        log.warning("Failed to normalize OHLC frame: %s", e)
        return pd.DataFrame()


def _retry(fn: Callable, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    """
    Simple retry with exponential backoff. Retries on common Kite exceptions and
    any unexpected exception once or twice, then surfaces the error.
    """
    delay = float(base_delay)
    last: Optional[BaseException] = None
    for i in range(max(1, int(tries))):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException, DataException, GeneralException) as e:  # type: ignore
            last = e
        except Exception as e:  # other unexpected (keep very short retries)
            last = e
        if i == tries - 1:
            break
        time.sleep(delay)
        delay = min(8.0, delay * 2.0)
    if last:
        raise last


def _coerce_interval(s: str) -> str:
    s = (s or "").strip().lower()
    interval = _INTERVAL_MAP.get(s)
    if interval:
        return interval
    log.warning("Unsupported timeframe '%s' — falling back to 'minute'.", s)
    return "minute"


def _clip_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        return df.loc[(df.index >= start) & (df.index <= end)].copy()
    except Exception:
        return df.copy()


# --------------------------------------------------------------------------------------
# LiveKiteSource
# --------------------------------------------------------------------------------------
class LiveKiteSource(DataSource):
    """
    Reads candles via Kite's historical API + LTP for quick checks.
    Adds a tiny TTL cache to stay under rate limits during frequent ticks/diags.
    Provides a safe fallback (synthetic bar at LTP) so downstream diags don't break.
    """

    def __init__(self, kite: Optional["KiteConnect"]) -> None:
        self.kite = kite
        self._cache = _TTLCache(ttl_sec=4.0)

    # ---- lifecycle ----
    def connect(self) -> None:
        if not self.kite:
            log.info("LiveKiteSource: kite is None (shadow mode).")
        else:
            log.info("LiveKiteSource: connected to Kite.")

    # ---- quick LTP ----
    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        if not self.kite:
            return None
        try:
            # Accept either token int or exchange:symbol string
            if isinstance(symbol_or_token, int):
                data = _retry(self.kite.ltp, [symbol_or_token], tries=2)
                for _, v in (data or {}).items():
                    val = v.get("last_price")
                    return float(val) if isinstance(val, (int, float)) else None
                return None
            else:
                sym = str(symbol_or_token)
                data = _retry(self.kite.ltp, [sym], tries=2)
                v = (data or {}).get(sym)
                if isinstance(v, dict) and "last_price" in v:
                    return float(v["last_price"])
                # fallback: first value
                for _, vv in (data or {}).items():
                    val = vv.get("last_price")
                    return float(val) if isinstance(val, (int, float)) else None
                return None
        except Exception as e:
            log.debug("get_last_price failed for %s: %s", symbol_or_token, e)
            return None

    # ---- main candle fetch ----
    def fetch_ohlc(self, token: int, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        """
        Primary path: Kite historical_data
        Fallback: if empty, synthesize one 'bar' from current LTP to keep diagnostics alive.
        """
        if not self.kite:
            log.warning("LiveKiteSource.fetch_ohlc: kite is None.")
            return pd.DataFrame()

        # Guard inputs
        try:
            token = int(token)
        except Exception:
            log.error("fetch_ohlc: invalid token %r", token)
            return pd.DataFrame()

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            log.error("fetch_ohlc: start/end must be datetime, got %r %r", type(start), type(end))
            return pd.DataFrame()

        if start >= end:
            # Soft auto-correct: if equal or reversed, nudge start back 10 minutes
            start = end - timedelta(minutes=10)

        interval = _coerce_interval(str(timeframe))

        # Try cache
        cached = self._cache.get(token, interval, start, end)
        if cached is not None and not cached.empty:
            return _clip_window(cached, start, end)

        # Kite expects naive or tz-aware UTC; we'll pass naive (already)
        frm = pd.to_datetime(start).to_pydatetime()
        to = pd.to_datetime(end).to_pydatetime()

        # Kite limits historical_data to ~2000 candles per call.  Chunk long
        # requests (e.g., multi‑day backfills) into smaller ranges and stitch
        # the results together.  This keeps the external behaviour the same
        # while avoiding silent truncation.
        interval_minutes = {
            "minute": 1,
            "3minute": 3,
            "5minute": 5,
            "10minute": 10,
            "15minute": 15,
        }
        step: Optional[timedelta] = None
        if interval in interval_minutes:
            step = timedelta(minutes=interval_minutes[interval] * 2000)
        elif interval == "day":
            step = timedelta(days=2000)

        frames: List[pd.DataFrame] = []
        cur = frm
        try:
            while cur < to:
                cur_end = to if step is None else min(to, cur + step)
                rows = _retry(
                    self.kite.historical_data,
                    token,
                    cur,
                    cur_end,
                    interval,
                    continuous=False,
                    oi=False,
                    tries=3,
                )
                part = _safe_dataframe(rows)
                if not part.empty:
                    frames.append(part)
                cur = cur_end

            df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
            df = df[~df.index.duplicated(keep="last")]

            if df.empty:
                # This can happen near the right edge of a session or with brief API hiccups.
                # Use LTP to synthesize a single bar to avoid a hard stop.
                log.warning(
                    "historical_data empty for token=%s interval=%s window=%s→%s; using LTP fallback.",
                    token, interval, start, end,
                )
                ltp = self.get_last_price(token)
                if isinstance(ltp, (int, float)):
                    ts = _now_ist_naive().replace(second=0, microsecond=0)
                    df = pd.DataFrame(
                        {"open": [ltp], "high": [ltp], "low": [ltp], "close": [ltp], "volume": [0]},
                        index=[ts],
                    )

            if df is not None and not df.empty:
                # Clip to requested window and cache the *unclipped* frame covering the window
                clipped = _clip_window(df, start, end)
                self._cache.set(token, interval, df, start, end)
                # Validate schema again (defensive)
                need = {"open", "high", "low", "close"}
                if not clipped.empty and need.issubset(clipped.columns):
                    return clipped

            return pd.DataFrame()

        except Exception as e:
            # Network hiccups are common; degrade to a single LTP bar so callers
            # can continue running diagnostics without spamming errors.
            log.warning("fetch_ohlc failed token=%s interval=%s: %s", token, interval, e)
            ltp = self.get_last_price(token)
            if isinstance(ltp, (int, float)):
                ts = _now_ist_naive().replace(second=0, microsecond=0)
                return pd.DataFrame(
                    {"open": [ltp], "high": [ltp], "low": [ltp], "close": [ltp], "volume": [0]},
                    index=[ts],
                )
            return pd.DataFrame()
