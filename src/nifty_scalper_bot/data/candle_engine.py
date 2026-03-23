"""Deterministic candle lifecycle + data hydration helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

import pandas as pd

FetchHistoricalFn = Callable[[str], pd.DataFrame | None]
FetchRecentFn = Callable[[str], pd.DataFrame | None]


@dataclass(slots=True)
class CandleEngine:
    """Build candles from ticks.

    Args: interval/max_bars. Returns: engine. Raises: ValueError.
    """

    interval: str = "1min"
    max_bars: int = 500
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_candle: dict[str, Any] | None = None
    last_candle_close: datetime | None = None

    def on_tick(self, tick: Mapping[str, Any]) -> None:
        """Ingest tick. Args: tick. Returns: None. Raises: ValueError."""
        ts = pd.to_datetime(tick["timestamp"], utc=True)
        raw_price = tick.get("price") or tick.get("ltp") or tick.get("last_price")
        if raw_price is None:
            msg = "tick price is required"
            raise ValueError(msg)
        price = float(raw_price)
        normalized_tick: dict[str, Any] = {
            "timestamp": ts,
            "price": price,
            "volume": float(tick.get("volume") or 0.0),
        }

        if self.current_candle is None:
            self.current_candle = self._init_candle(normalized_tick)
            return

        if ts >= self._candle_close_time(
            pd.Timestamp(self.current_candle["timestamp"])
        ):
            self._finalize_candle()
            self.current_candle = self._init_candle(normalized_tick)
        else:
            self._update_candle(normalized_tick)

    def _init_candle(self, tick: Mapping[str, Any]) -> dict[str, Any]:
        """Create candle. Args: tick. Returns: candle dict. Raises: None."""
        price = float(tick["price"])
        return {
            "timestamp": pd.to_datetime(tick["timestamp"], utc=True),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": float(tick.get("volume") or 0.0),
        }

    def _update_candle(self, tick: Mapping[str, Any]) -> None:
        """Update candle. Args: tick. Returns: None. Raises: None."""
        if self.current_candle is None:
            return
        price = float(tick["price"])
        self.current_candle["high"] = max(float(self.current_candle["high"]), price)
        self.current_candle["low"] = min(float(self.current_candle["low"]), price)
        self.current_candle["close"] = price
        self.current_candle["volume"] = float(
            self.current_candle.get("volume") or 0.0
        ) + float(tick.get("volume") or 0.0)

    def _candle_close_time(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Return close time. Args: ts. Returns: timestamp. Raises: ValueError."""
        if self.interval != "1min":
            msg = f"Unsupported interval: {self.interval}"
            raise ValueError(msg)
        base = ts.floor("min")
        return base + pd.Timedelta(minutes=1)

    def _finalize_candle(self) -> None:
        """Persist candle. Args: none. Returns: None. Raises: None."""
        if self.current_candle is None:
            return
        df_new = pd.DataFrame([self.current_candle])
        merged = pd.concat([self.df, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset="timestamp", keep="last")
        merged = (
            merged.sort_values("timestamp").tail(self.max_bars).reset_index(drop=True)
        )
        self.df = merged
        self.last_candle_close = pd.to_datetime(
            self.current_candle["timestamp"], utc=True
        )

    def get_df(self) -> pd.DataFrame:
        """Return candles. Args: none. Returns: DataFrame. Raises: None."""
        return self.df.copy(deep=True)


def detect_gap(df: pd.DataFrame) -> bool:
    """Detect gaps. Args: df. Returns: bool. Raises: None."""
    if df is None or len(df) < 3 or "timestamp" not in df.columns:
        return False
    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .dropna()
        .sort_values()
    )
    if len(ts) < 3:
        return False
    diffs = ts.diff().dropna()
    expected = diffs.mode().iloc[0]
    return bool((diffs != expected).any())


def repair_with_backfill(
    symbol: str,
    df: pd.DataFrame,
    *,
    fetch_recent_rest: FetchRecentFn,
    max_bars: int = 500,
) -> pd.DataFrame:
    """Repair gaps.

    Args: symbol/df/fetch_recent_rest/max_bars. Returns: DataFrame. Raises: Exception.
    """
    recent = fetch_recent_rest(symbol)
    if recent is None or recent.empty:
        return df.tail(max_bars).copy() if df is not None else pd.DataFrame()

    merged = pd.concat([df, recent], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["timestamp"])
    merged = merged.drop_duplicates(subset="timestamp", keep="last")
    merged = merged.sort_values("timestamp")
    return merged.tail(max_bars).reset_index(drop=True)


def fetch_historical_safe(
    symbol: str,
    *,
    fetch_historical: FetchHistoricalFn,
    min_required: int = 50,
    retries: int = 3,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame | None:
    """Fetch history safely.

    Args: symbol/fetch_historical/min_required/retries/sleep_seconds.
    Returns: DataFrame|None. Raises: None.
    """
    for _ in range(retries):
        df = fetch_historical(symbol)
        if df is not None and len(df) >= min_required:
            return df
        time.sleep(sleep_seconds)
    return None


def validate_dataframe(df: pd.DataFrame | None, min_required: int = 50) -> bool:
    """Validate candles. Args: df/min_required. Returns: bool. Raises: None."""
    if df is None:
        return False
    if len(df) < min_required:
        return False
    if df.isnull().any().any():
        return False
    if df.duplicated(subset="timestamp").any():
        return False
    if detect_gap(df):
        return False
    return True


def normalize_ohlc_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tz. Args: df. Returns: DataFrame. Raises: None."""
    normalized = df.copy()
    ts = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized["timestamp"] = ts.dt.tz_convert("Asia/Kolkata")
    normalized = normalized.dropna(subset=["timestamp"]).set_index("timestamp")
    return normalized


def ensure_valid_data(
    symbol: str,
    engine: CandleEngine,
    *,
    fetch_historical: FetchHistoricalFn,
    fetch_recent_rest: FetchRecentFn,
    min_required: int = 50,
) -> pd.DataFrame | None:
    """Ensure valid candles.

    Args: symbol/engine/fetch_historical/fetch_recent_rest/min_required.
    Returns: DataFrame|None. Raises: None.
    """
    df = engine.get_df()

    if not validate_dataframe(df, min_required=min_required):
        df_hist = fetch_historical_safe(
            symbol,
            fetch_historical=fetch_historical,
            min_required=min_required,
        )
        if df_hist is not None:
            engine.df = df_hist.copy(deep=True)
            return engine.get_df()
        return None

    if detect_gap(df):
        repaired = repair_with_backfill(
            symbol,
            df,
            fetch_recent_rest=fetch_recent_rest,
            max_bars=engine.max_bars,
        )
        engine.df = repaired.copy(deep=True)

    return engine.get_df()
