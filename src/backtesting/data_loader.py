# src/backtesting/data_loader.py
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, date
from typing import Iterable, Optional, Tuple, Dict, List

import pandas as pd

try:
    # Only for type hints – we don't import Kite if not available
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = object  # type: ignore

logger = logging.getLogger(__name__)

# ----------------------------- config knobs ----------------------------- #

# Max span per request for different granularities (conservative)
# (Zerodha often limits intraday ranges; these chunks keep calls reliable)
_INTERVAL_MAX_DAYS = {
    "minute": 30,
    "3minute": 30,
    "5minute": 60,
    "10minute": 90,
    "15minute": 90,
    "30minute": 180,
    "60minute": 365,
    "day": 3650,  # ~10 years per call is fine for daily
}

# Canonical aliases
_INTERVAL_ALIASES = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1h": "60minute",
    "d": "day",
}

_NUMERIC_COLS = ("open", "high", "low", "close", "volume", "oi")


# ----------------------------- utils ----------------------------------- #

def _norm_interval(interval: str) -> str:
    i = (interval or "5minute").lower().strip()
    return _INTERVAL_ALIASES.get(i, i)


def _ensure_dt(d: str | datetime | date) -> datetime:
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        # treat as naive midnight local
        return datetime(d.year, d.month, d.day)
    return datetime.strptime(str(d), "%Y-%m-%d")


def _daterange_chunks(start: datetime, end: datetime, interval: str) -> Iterable[Tuple[datetime, datetime]]:
    """Yield [chunk_start, chunk_end] pairs capped by interval limits."""
    start = min(start, end)
    step_days = _INTERVAL_MAX_DAYS.get(interval, 30)
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=step_days))
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _retry(fn, *args, tries: int = 3, backoff: float = 0.7, jitter: float = 0.3, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            last = exc
            if i == tries - 1:
                break
            sleep_s = backoff * (2 ** i) + (jitter * (0.3 + i))
            logger.warning("Transient error calling %s: %s → retry in %.2fs", getattr(fn, "__name__", "call"), exc, sleep_s)
            time.sleep(sleep_s)
    raise last  # re-raise last error


def _to_ist(ts: pd.Series) -> pd.Series:
    # IST = UTC+5:30; data is naive; keep it naive but shifted for consistent backtests
    return ts + pd.Timedelta(hours=5, minutes=30)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Keep only known columns, coerce to numeric
    keep = [c for c in _NUMERIC_COLS if c in df.columns]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Sort & drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[keep].copy()


# ----------------------------- public API ------------------------------- #

def load_csv(path: str, parse_dates: bool = True) -> pd.DataFrame:
    """Simple CSV loader that matches the normalized OHLCV schema if present."""
    try:
        if parse_dates:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index.name = "timestamp"
        else:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
        return _normalize_dataframe(df)
    except Exception as e:
        logger.error("Failed to load CSV %s: %s", path, e)
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_csv(path)
        logger.info("Saved %d rows to %s", len(df), path)
        return True
    except Exception as e:
        logger.error("Failed to save CSV %s: %s", path, e)
        return False


def load_zerodha_historical_data(
    kite: KiteConnect,
    instrument_token: int,
    from_date: str | datetime | date,
    to_date: str | datetime | date,
    interval: str = "5minute",
    include_partial: bool = False,
    cache_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Kite with chunking, retries, and normalization.

    Args:
        kite: authenticated KiteConnect client
        instrument_token: e.g., 256265 (NIFTY 50)
        from_date: 'YYYY-MM-DD' | datetime | date
        to_date: 'YYYY-MM-DD' | datetime | date
        interval: e.g., 'minute','3minute','5minute','15minute','60minute','day'
        include_partial: if False, attempts to drop the last (possibly forming) candle
        cache_csv: optional path to write the final DataFrame as CSV

    Returns:
        DataFrame indexed by 'timestamp' (naive, IST-shifted for consistency)
        with columns: open, high, low, close, volume (oi included if available).
    """
    try:
        iv = _norm_interval(interval)
        if iv not in _INTERVAL_MAX_DAYS:
            raise ValueError(f"Unsupported interval '{interval}'. Use one of: {list(_INTERVAL_MAX_DAYS)}")

        start = _ensure_dt(from_date)
        end = _ensure_dt(to_date)
        if start > end:
            start, end = end, start

        frames: List[pd.DataFrame] = []
        total_rows = 0

        for chunk_start, chunk_end in _daterange_chunks(start, end, iv):
            logger.info("📥 Fetching %s: %s → %s (token=%s)", iv, chunk_start.date(), chunk_end.date(), instrument_token)

            def _hist():
                return kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=chunk_start,
                    to_date=chunk_end,
                    interval=iv,
                    continuous=False,
                    oi=True,  # request OI when available
                )

            data = _retry(_hist, tries=3)
            if not data:
                logger.warning("⚠️ Empty slice %s → %s", chunk_start.date(), chunk_end.date())
                continue

            df = pd.DataFrame(data)
            if "date" not in df.columns:
                logger.warning("Slice missing 'date' field; skipping.")
                continue

            ts = pd.to_datetime(df["date"], utc=False)
            # Shift to IST to align with your backtests (naive index, IST-local)
            ts = _to_ist(ts)
            df = df.set_index(ts)
            df.index.name = "timestamp"
            df.drop(columns=["date"], inplace=True, errors="ignore")

            frames.append(df)
            total_rows += len(df)

        if not frames:
            logger.warning("No data fetched for the requested range.")
            return pd.DataFrame()

        out = pd.concat(frames, axis=0)
        out = _normalize_dataframe(out)

        # Trim last partial candle if asked
        if not include_partial and len(out) >= 2:
            # Heuristic: if the last index is >= the expected end bucket and
            # the bar time is still “live” (e.g., within the last interval), drop it.
            try:
                last_ts = out.index[-1]
                second_last = out.index[-2]
                # infer step minutes
                step_min = max(1, int((last_ts - second_last).total_seconds() // 60))
                now_ist = _to_ist(pd.Series([pd.Timestamp.utcnow()]))[0]
                if (now_ist - last_ts).total_seconds() < (step_min * 60):
                    out = out.iloc[:-1]
            except Exception:
                # best-effort only
                pass

        if cache_csv:
            save_csv(out, cache_csv)

        logger.info("✅ Loaded %d rows (%s → %s) at %s.", len(out), out.index.min(), out.index.max(), iv)
        return out

    except Exception as e:
        logger.error("❌ Exception loading Zerodha historical data: %s", e, exc_info=True)
        return pd.DataFrame()


# Convenience: choose CSV if provided, else broker
def load_data(
    kite: Optional[KiteConnect],
    instrument_token: int,
    from_date: str | datetime | date,
    to_date: str | datetime | date,
    interval: str = "5minute",
    csv_path: Optional[str] = None,
    include_partial: bool = False,
) -> pd.DataFrame:
    """
    If csv_path is given and exists → load_csv; else fetch from Zerodha.
    """
    if csv_path:
        try:
            return load_csv(csv_path)
        except Exception:
            logger.warning("CSV load failed, falling back to broker fetch.")

    if not kite:
        logger.error("Kite client is required when csv_path is not provided.")
        return pd.DataFrame()

    return load_zerodha_historical_data(
        kite=kite,
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        include_partial=include_partial,
    )