"""Deterministic 1-minute candle engine and strict OHLC validation helpers."""

from __future__ import annotations

import logging
import os
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
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raw = float(value)
            if raw > 1e12:
                ts = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
            elif raw > 946684800:
                ts = pd.to_datetime(raw, unit="s", utc=True, errors="coerce")
            else:
                ts = pd.NaT
        else:
            ts = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def _future_grace_seconds() -> float:
    try:
        return max(float(os.getenv("MARKETDATA_MAX_FUTURE_CANDLE_SECONDS", "120") or 120), 0.0)
    except (TypeError, ValueError):
        return 120.0


def _is_future_timestamp(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    return bool(ts > pd.Timestamp.now(tz=IST) + pd.Timedelta(seconds=_future_grace_seconds()))


def _series_to_ist(values: Any) -> pd.Series:
    return pd.Series(values).map(_to_ist_timestamp)


def sanitize(df: pd.DataFrame | None) -> pd.DataFrame:
    """Drop invalid OHLC rows without synthetic repair."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    cleaned = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in cleaned.columns:
            cleaned[col] = 0.0
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned["timestamp"] = _series_to_ist(cleaned.get("timestamp"))
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["timestamp", "open", "high", "low", "close"])
    future_mask = cleaned["timestamp"].map(_is_future_timestamp)
    future_dropped = int(future_mask.sum()) if not cleaned.empty else 0
    if future_dropped:
        LOGGER.warning(
            "future_candle_rejected",
            extra={"event": "future_candle_rejected", "source": "sanitize", "dropped_rows": future_dropped},
        )
        cleaned = cleaned.loc[~future_mask]
    cleaned = cleaned.drop_duplicates(subset="timestamp", keep="last")
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped > 0:
        LOGGER.warning("tick_invalid", extra={"event": "tick_invalid", "dropped_rows": dropped})
    return cleaned[list(_OHLC_COLUMNS)]


@dataclass(slots=True)
class CandleEngine:
    """Build deterministic 1-minute candles from validated ticks."""

    interval: str = "1min"
    max_bars: int = 500
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_OHLC_COLUMNS))
    current_candle: dict[str, Any] | None = None
    last_candle_close: datetime | None = None
    _last_tick_ts: pd.Timestamp | None = None

    def on_tick(self, tick: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.interval != "1min":
            raise ValueError(f"Unsupported interval: {self.interval}")
        raw_ts = tick.get("timestamp") if isinstance(tick, dict) else getattr(tick, "timestamp", None)
        timestamp = _to_ist_timestamp(raw_ts)
        if pd.isna(timestamp) or getattr(timestamp, "year", 1970) < 2020:
            log_throttled(LOGGER, "candle_tick_bad_timestamp", "CANDLE_TICK_DROPPED reason=bad_timestamp", interval_sec=10.0, level=logging.WARNING)
            return None
        if _is_future_timestamp(timestamp):
            log_throttled(
                LOGGER,
                f"candle_tick_future:{getattr(self, 'symbol', 'unknown')}",
                "CANDLE_TICK_DROPPED reason=future_timestamp symbol=%s tick_ts=%s" % (getattr(self, "symbol", "unknown"), timestamp.isoformat()),
                interval_sec=30.0,
                level=logging.WARNING,
            )
            return None
        validated = tick if isinstance(tick, Tick) else validate_tick(dict(tick))
        payload = validated.to_dict()
        timestamp = _to_ist_timestamp(payload["timestamp"])
        minute = timestamp.floor("1min")

        if self._last_tick_ts is not None:
            last_ts = _to_ist_timestamp(self._last_tick_ts)
            if not pd.isna(last_ts) and timestamp < last_ts:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_{getattr(self, 'symbol', 'unknown')}",
                    "tick_out_of_order symbol=%s tick_ts=%s last_ts=%s" % (getattr(self, "symbol", "unknown"), timestamp.isoformat(), last_ts.isoformat()),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None

        finalized: dict[str, Any] | None = None
        if self.current_candle is None:
            self.current_candle = self._start_candle(payload, minute)
            LOGGER.debug("candle_created", extra={"event": "candle_created", "minute": minute.isoformat()})
        else:
            current_minute = _to_ist_timestamp(self.current_candle["timestamp"])
            if minute < current_minute:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_minute_{getattr(self, 'symbol', 'unknown')}",
                    "tick_out_of_order symbol=%s tick_minute=%s current_minute=%s" % (getattr(self, "symbol", "unknown"), minute.isoformat(), current_minute.isoformat()),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None
            if minute > current_minute:
                finalized = self._finalize_current_candle()
                self.current_candle = self._start_candle(payload, minute)
                LOGGER.debug("candle_created", extra={"event": "candle_created", "minute": minute.isoformat()})
            else:
                self._update_candle(payload)
        self._last_tick_ts = timestamp
        return finalized

    def _start_candle(self, tick: Mapping[str, Any], minute: pd.Timestamp) -> dict[str, Any]:
        price = float(tick["ltp"])
        return {"timestamp": minute, "open": price, "high": price, "low": price, "close": price, "volume": float(tick.get("volume") or 0.0)}

    def _update_candle(self, tick: Mapping[str, Any]) -> None:
        if self.current_candle is None:
            raise DataIntegrityError("missing current candle")
        price = float(tick["ltp"])
        self.current_candle["high"] = max(float(self.current_candle["high"]), price)
        self.current_candle["low"] = min(float(self.current_candle["low"]), price)
        self.current_candle["close"] = price
        self.current_candle["volume"] = float(self.current_candle["volume"]) + float(tick.get("volume") or 0.0)

    def _finalize_current_candle(self) -> dict[str, Any] | None:
        if self.current_candle is None:
            return None
        candle = dict(self.current_candle)
        _validate_ohlc_row(candle)
        incoming_ts = _to_ist_timestamp(candle.get("timestamp"))
        if pd.isna(incoming_ts):
            raise DataIntegrityError("candle timestamp is invalid")
        if _is_future_timestamp(incoming_ts):
            log_throttled(
                LOGGER,
                f"candle_future_{getattr(self, 'symbol', 'unknown')}",
                "future_candle_rejected symbol=%s incoming_ts=%s source=candle_engine" % (getattr(self, "symbol", "unknown"), incoming_ts.isoformat()),
                interval_sec=30.0,
                level=logging.WARNING,
                extra={"event": "future_candle_rejected", "symbol": getattr(self, "symbol", "unknown"), "incoming_ts": incoming_ts.isoformat(), "source": "candle_engine"},
            )
            self.current_candle = None
            return None
        existing = self.df.dropna(how="all") if self.df is not None else pd.DataFrame(columns=_OHLC_COLUMNS)
        if not existing.empty:
            last_ts = _to_ist_timestamp(existing["timestamp"].iloc[-1])
            if not pd.isna(last_ts):
                if incoming_ts < last_ts:
                    log_throttled(
                        LOGGER,
                        f"candle_out_of_order_{getattr(self, 'symbol', 'unknown')}",
                        "candle_out_of_order symbol=%s incoming_ts=%s last_ts=%s source=candle_engine" % (getattr(self, "symbol", "unknown"), incoming_ts.isoformat(), last_ts.isoformat()),
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={"event": "candle_out_of_order", "symbol": getattr(self, "symbol", "unknown"), "incoming_ts": incoming_ts.isoformat(), "last_ts": last_ts.isoformat(), "source": "candle_engine"},
                    )
                    raise DataIntegrityError("candle timestamps must be monotonic")
                if incoming_ts == last_ts:
                    self.current_candle = None
                    return None
        candle["timestamp"] = incoming_ts
        new_row = pd.DataFrame([candle]).dropna(how="all")
        if new_row.empty:
            return None
        frame = new_row.reset_index(drop=True) if existing.empty else pd.concat([existing, new_row], ignore_index=True)
        frame = sanitize(frame).tail(self.max_bars).reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise DataIntegrityError("duplicate candle timestamps")
        if not frame["timestamp"].is_monotonic_increasing:
            raise DataIntegrityError("candle timestamps must be monotonic")
        self.df = frame
        self.last_candle_close = incoming_ts.to_pydatetime()
        LOGGER.debug("candle_finalized", extra={"event": "candle_finalized", "timestamp": incoming_ts.isoformat()})
        return candle

    def flush(self) -> dict[str, Any] | None:
        finalized = self._finalize_current_candle()
        if finalized is not None:
            self.current_candle = None
        return finalized

    def get_df(self) -> pd.DataFrame:
        return self.df.copy(deep=True)


def _validate_ohlc_row(row: Mapping[str, Any]) -> None:
    op = float(row["open"])
    hi = float(row["high"])
    lo = float(row["low"])
    cl = float(row["close"])
    if hi < max(op, cl):
        raise DataIntegrityError("high must be >= open/close")
    if lo > min(op, cl):
        raise DataIntegrityError("low must be <= open/close")


def detect_gap(df: pd.DataFrame) -> bool:
    clean = sanitize(df)
    if len(clean) < 2:
        return False
    diffs = clean["timestamp"].diff().dropna()
    return bool((diffs != pd.Timedelta(minutes=1)).any())


def repair_with_backfill(symbol: str, df: pd.DataFrame, *, fetch_recent_rest: FetchRecentFn, max_bars: int = 500) -> pd.DataFrame:
    LOGGER.info("repair_with_backfill_deprecated symbol=%s", symbol, extra={"event": "repair_with_backfill_deprecated", "symbol": symbol})
    recent = sanitize(fetch_recent_rest(symbol))
    merged = sanitize(pd.concat([sanitize(df), recent], ignore_index=True))
    return merged.tail(max_bars).reset_index(drop=True)


def fetch_historical_safe(symbol: str, *, fetch_historical: FetchHistoricalFn, min_required: int = 50, retries: int = 3, sleep_seconds: float = 0.5) -> pd.DataFrame | None:
    for attempt in range(1, retries + 1):
        frame = sanitize(fetch_historical(symbol))
        if validate_dataframe(frame, min_required=min_required):
            return frame
        LOGGER.warning(
            "data_integrity_error symbol=%s reason=historical_validation_failed attempt=%d",
            symbol,
            attempt,
            extra={"event": "data_integrity_error", "symbol": symbol, "attempt": attempt, "reason": "historical_validation_failed"},
        )
        if attempt < retries:
            time.sleep(sleep_seconds)
    return None


def validate_dataframe(df: pd.DataFrame | None, min_required: int = 50) -> bool:
    if df is None:
        return False
    clean = sanitize(df)
    if len(clean) < min_required:
        return False
    if clean["timestamp"].duplicated().any() or not clean["timestamp"].is_monotonic_increasing:
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
    normalized = sanitize(df)
    ts = _series_to_ist(normalized["timestamp"])
    if ts.isna().any():
        raise DataIntegrityError("Invalid timestamp in normalize_ohlc_timezone")
    normalized["timestamp"] = ts
    return normalized.set_index("timestamp")


def ensure_valid_data(symbol: str, engine: CandleEngine, *, fetch_historical: FetchHistoricalFn, fetch_recent_rest: FetchRecentFn, min_required: int = 50) -> pd.DataFrame | None:
    del fetch_recent_rest
    frame = sanitize(engine.get_df())
    if validate_dataframe(frame, min_required=min_required):
        return frame
    hydrated = fetch_historical_safe(symbol, fetch_historical=fetch_historical, min_required=min_required, retries=1)
    if hydrated is None:
        LOGGER.error(
            "data_integrity_error symbol=%s reason=insufficient_historical min_required=%d",
            symbol,
            min_required,
            extra={"event": "data_integrity_error", "symbol": symbol, "reason": "insufficient_historical", "min_required": min_required},
        )
        return None
    engine.df = hydrated.tail(engine.max_bars).reset_index(drop=True)
    return engine.get_df()
