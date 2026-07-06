"""Deterministic 1-minute candle engine and strict OHLC validation helpers.

Runtime role:
- Owns tick-to-OHLC bar construction, OHLC validation, and bar readiness.
- Consumes normalized ticks from MarketDataManager.
- Must not select contracts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.validator import Tick, validate_tick
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
FetchHistoricalFn = Callable[[str], pd.DataFrame | None]
FetchRecentFn = Callable[[str], pd.DataFrame | None]
IST = ZoneInfo("Asia/Kolkata")
_OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _to_ist_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).tz_convert(IST)


def sanitize(df: pd.DataFrame | None) -> pd.DataFrame:
    """Drop invalid OHLC rows without synthetic repair."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    cleaned = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in cleaned.columns:
            cleaned[col] = 0.0
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned["timestamp"] = pd.to_datetime(
        cleaned.get("timestamp"), utc=True, errors="coerce"
    ).dt.tz_convert(IST)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["timestamp", "open", "high", "low", "close"])
    cleaned = cleaned.drop_duplicates(subset="timestamp", keep="last")
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped > 0:
        LOGGER.warning(
            "tick_invalid", extra={"event": "tick_invalid", "dropped_rows": dropped}
        )
    return cleaned[list(_OHLC_COLUMNS)]


@dataclass(slots=True)
class CandleEngine:
    """Build deterministic 1-minute candles from validated ticks."""

    interval: str = "1min"
    max_bars: int = 500
    df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=_OHLC_COLUMNS)
    )
    current_candle: dict[str, Any] | None = None
    last_candle_close: datetime | None = None
    _last_tick_ts: pd.Timestamp | None = None

    def on_tick(self, tick: Mapping[str, Any]) -> dict[str, Any] | None:
        """Ingest one tick and return a finalized candle when a minute closes."""
        if self.interval != "1min":
            raise ValueError(f"Unsupported interval: {self.interval}")

        raw_ts = (
            tick.get("timestamp")
            if isinstance(tick, dict)
            else getattr(tick, "timestamp", None)
        )
        timestamp = _to_ist_timestamp(raw_ts)
        if pd.isna(timestamp) or getattr(timestamp, "year", 1970) < 2020:
            log_throttled(
                LOGGER,
                "candle_tick_bad_timestamp",
                "CANDLE_TICK_DROPPED reason=bad_timestamp",
                interval_sec=10.0,
                level=logging.WARNING,
            )
            return None
        validated = tick if isinstance(tick, Tick) else validate_tick(dict(tick))
        payload = validated.to_dict()
        timestamp = pd.Timestamp(payload["timestamp"])
        minute = timestamp.floor("1min")

        if self._last_tick_ts is not None:
            last_ts = _to_ist_timestamp(self._last_tick_ts)
            if not pd.isna(last_ts) and timestamp < last_ts:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_{getattr(self, 'symbol', 'unknown')}",
                    "tick_out_of_order symbol=%s tick_ts=%s last_ts=%s"
                    % (
                        getattr(self, "symbol", "unknown"),
                        timestamp.isoformat(),
                        last_ts.isoformat(),
                    ),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None

        finalized: dict[str, Any] | None = None
        if self.current_candle is None:
            self.current_candle = self._start_candle(payload, minute)
            LOGGER.debug(
                "candle_created",
                extra={"event": "candle_created", "minute": minute.isoformat()},
            )
        else:
            current_minute = pd.Timestamp(self.current_candle["timestamp"])
            if minute < current_minute:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_minute_{getattr(self, 'symbol', 'unknown')}",
                    "tick_out_of_order symbol=%s tick_minute=%s current_minute=%s"
                    % (
                        getattr(self, "symbol", "unknown"),
                        minute.isoformat(),
                        current_minute.isoformat(),
                    ),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None
            if minute > current_minute:
                finalized = self._finalize_current_candle()
                self.current_candle = self._start_candle(payload, minute)
                LOGGER.debug(
                    "candle_created",
                    extra={"event": "candle_created", "minute": minute.isoformat()},
                )
            else:
                self._update_candle(payload)

        self._last_tick_ts = timestamp
        return finalized

    def _start_candle(
        self, tick: Mapping[str, Any], minute: pd.Timestamp
    ) -> dict[str, Any]:
        """Initialize a minute candle."""
        price = float(tick["ltp"])
        return {
            "timestamp": minute,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": float(tick.get("volume") or 0.0),
        }

    def _update_candle(self, tick: Mapping[str, Any]) -> None:
        """Update the active candle OHLC state."""
        if self.current_candle is None:
            raise DataIntegrityError("missing current candle")
        price = float(tick["ltp"])
        self.current_candle["high"] = max(float(self.current_candle["high"]), price)
        self.current_candle["low"] = min(float(self.current_candle["low"]), price)
        self.current_candle["close"] = price
        self.current_candle["volume"] = float(self.current_candle["volume"]) + float(
            tick.get("volume") or 0.0
        )

    def _finalize_current_candle(self) -> dict[str, Any] | None:
        """Finalize the active candle."""
        if self.current_candle is None:
            return None
        candle = dict(self.current_candle)
        _validate_ohlc_row(candle)
        incoming_ts = _to_ist_timestamp(candle.get("timestamp"))
        if pd.isna(incoming_ts):
            raise DataIntegrityError("candle timestamp is invalid")
        existing = (
            self.df.dropna(how="all")
            if self.df is not None
            else pd.DataFrame(columns=_OHLC_COLUMNS)
        )
        if not existing.empty:
            last_ts = _to_ist_timestamp(existing["timestamp"].iloc[-1])
            if not pd.isna(last_ts):
                if incoming_ts < last_ts:
                    log_throttled(
                        LOGGER,
                        f"candle_out_of_order_{getattr(self, 'symbol', 'unknown')}",
                        (
                            "candle_out_of_order symbol=%s incoming_ts=%s "
                            "last_ts=%s source=candle_engine"
                        )
                        % (
                            getattr(self, "symbol", "unknown"),
                            incoming_ts.isoformat(),
                            last_ts.isoformat(),
                        ),
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "candle_out_of_order",
                            "symbol": getattr(self, "symbol", "unknown"),
                            "incoming_ts": incoming_ts.isoformat(),
                            "last_ts": last_ts.isoformat(),
                            "source": "candle_engine",
                        },
                    )
                    raise DataIntegrityError("candle timestamps must be monotonic")
                if incoming_ts == last_ts:
                    self.current_candle = None
                    return None
        new_row = pd.DataFrame([candle]).dropna(how="all")
        if new_row.empty:
            return None
        if existing.empty:
            frame = new_row.reset_index(drop=True)
        else:
            frame = pd.concat([existing, new_row], ignore_index=True)
        frame = sanitize(frame).tail(self.max_bars).reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise DataIntegrityError("duplicate candle timestamps")
        if not frame["timestamp"].is_monotonic_increasing:
            raise DataIntegrityError("candle timestamps must be monotonic")
        self.df = frame
        self.last_candle_close = _to_ist_timestamp(candle["timestamp"]).to_pydatetime()
        LOGGER.debug(
            "candle_finalized",
            extra={
                "event": "candle_finalized",
                "timestamp": pd.Timestamp(candle["timestamp"]).isoformat(),
            },
        )
        return candle

    def flush(self) -> dict[str, Any] | None:
        """Flush the active candle into the finalized store."""
        finalized = self._finalize_current_candle()
        if finalized is not None:
            self.current_candle = None
        return finalized

    def get_df(self) -> pd.DataFrame:
        """Return finalized candles. Args: none. Returns: DataFrame. Raises: None."""
        return self.df.copy(deep=True)


def _validate_ohlc_row(row: Mapping[str, Any]) -> None:
    """Validate OHLC consistency invariants."""
    op = float(row["open"])
    hi = float(row["high"])
    lo = float(row["low"])
    cl = float(row["close"])
    if hi < max(op, cl):
        raise DataIntegrityError("high must be >= open/close")
    if lo > min(op, cl):
        raise DataIntegrityError("low must be <= open/close")


def detect_gap(df: pd.DataFrame) -> bool:
    """Detect missing 1-minute buckets. Args: df. Returns: bool. Raises: None."""
    clean = sanitize(df)
    if len(clean) < 2:
        return False
    diffs = clean["timestamp"].diff().dropna()
    return bool((diffs != pd.Timedelta(minutes=1)).any())


def repair_with_backfill(
    symbol: str,
    df: pd.DataFrame,
    *,
    fetch_recent_rest: FetchRecentFn,
    max_bars: int = 500,
) -> pd.DataFrame:
    """Merge deterministic recent candles only without synthetic fills."""
    LOGGER.info(
        "data_integrity_error",
        extra={
            "event": "data_integrity_error",
            "symbol": symbol,
            "reason": "repair_with_backfill_deprecated",
        },
    )
    recent = sanitize(fetch_recent_rest(symbol))
    merged = sanitize(pd.concat([sanitize(df), recent], ignore_index=True))
    return merged.tail(max_bars).reset_index(drop=True)


def fetch_historical_safe(
    symbol: str,
    *,
    fetch_historical: FetchHistoricalFn,
    min_required: int = 50,
    retries: int = 3,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame | None:
    """Fetch historical candles with explicit retries and logs."""
    for attempt in range(1, retries + 1):
        frame = sanitize(fetch_historical(symbol))
        if validate_dataframe(frame, min_required=min_required):
            return frame
        LOGGER.warning(
            "data_integrity_error",
            extra={
                "event": "data_integrity_error",
                "symbol": symbol,
                "attempt": attempt,
                "reason": "historical_validation_failed",
            },
        )
        if attempt < retries:
            time.sleep(sleep_seconds)
    return None


def validate_dataframe(df: pd.DataFrame | None, min_required: int = 50) -> bool:
    """Validate OHLC frame integrity."""
    if df is None:
        return False
    clean = sanitize(df)
    if len(clean) < min_required:
        return False
    if (
        clean["timestamp"].duplicated().any()
        or not clean["timestamp"].is_monotonic_increasing
    ):
        return False
    if detect_gap(clean):
        return False
    try:
        for row in clean.to_dict(orient="records"):
            _validate_ohlc_row(row)
    except DataIntegrityError:
        return False
    return True


def normalize_ohlc_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the timestamp index to Asia/Kolkata."""
    normalized = sanitize(df)
    ts = pd.to_datetime(
        normalized["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(IST)
    if ts.isna().any():
        raise DataIntegrityError("Invalid timestamp in normalize_ohlc_timezone")
    normalized["timestamp"] = ts
    return normalized.set_index("timestamp")


def ensure_valid_data(
    symbol: str,
    engine: CandleEngine,
    *,
    fetch_historical: FetchHistoricalFn,
    fetch_recent_rest: FetchRecentFn,
    min_required: int = 50,
) -> pd.DataFrame | None:
    """Ensure candle cache integrity using startup hydration only."""
    del fetch_recent_rest
    frame = sanitize(engine.get_df())
    if validate_dataframe(frame, min_required=min_required):
        return frame

    hydrated = fetch_historical_safe(
        symbol,
        fetch_historical=fetch_historical,
        min_required=min_required,
        retries=1,
    )
    if hydrated is None:
        LOGGER.error(
            "data_integrity_error",
            extra={
                "event": "data_integrity_error",
                "symbol": symbol,
                "reason": "insufficient_historical",
            },
        )
        return None
    engine.df = hydrated.tail(engine.max_bars).reset_index(drop=True)
    return engine.get_df()
