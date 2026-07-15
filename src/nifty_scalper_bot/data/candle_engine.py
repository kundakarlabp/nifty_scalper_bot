"""Deterministic 1-minute candle engine and strict OHLC validation helpers."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.time_contract import (
    IST,
    coerce_market_timestamp,
    future_delta_seconds,
    is_future_market_timestamp,
    normalize_market_tick_timestamp,
    normalized_symbol,
)
from nifty_scalper_bot.data.validator import Tick, validate_tick
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
FetchHistoricalFn = Callable[[str], pd.DataFrame | None]
FetchRecentFn = Callable[[str], pd.DataFrame | None]
_OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _to_ist_timestamp(value: Any) -> pd.Timestamp:
    try:
        return coerce_market_timestamp(value)
    except ValueError:
        return pd.NaT


def _future_grace_seconds() -> float:
    try:
        return max(
            float(os.getenv("MARKETDATA_MAX_FUTURE_CANDLE_SECONDS", "120") or 120), 0.0
        )
    except (TypeError, ValueError):
        return 120.0


def _is_future_timestamp(ts: pd.Timestamp, *, now: pd.Timestamp | None = None) -> bool:
    if pd.isna(ts):
        return False
    now_ts = now or pd.Timestamp.now(tz=IST)
    return is_future_market_timestamp(
        ts, now=now_ts, grace_seconds=_future_grace_seconds()
    )


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
            extra={
                "event": "future_candle_rejected",
                "source": "sanitize",
                "dropped_rows": future_dropped,
            },
        )
        cleaned = cleaned.loc[~future_mask]
    cleaned = cleaned.drop_duplicates(subset="timestamp", keep="last")
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped > 0:
        LOGGER.warning(
            "tick_invalid", extra={"event": "tick_invalid", "dropped_rows": dropped}
        )
    return cleaned[list(_OHLC_COLUMNS)]


def _tick_value(tick: Mapping[str, Any] | Tick, key: str, default: Any = None) -> Any:
    if isinstance(tick, Mapping):
        return tick.get(key, default)
    return getattr(tick, key, default)


def _tick_symbol(tick: Mapping[str, Any] | Tick, engine_symbol: str | None) -> str:
    value = (
        _tick_value(tick, "symbol")
        or _tick_value(tick, "trading_symbol")
        or engine_symbol
    )
    return normalized_symbol(value)


def _tick_timestamp(tick: Mapping[str, Any] | Tick) -> tuple[pd.Timestamp, str, Any]:
    if isinstance(tick, Mapping):
        normalized_ts = normalize_market_tick_timestamp(tick)
        return normalized_ts.timestamp, normalized_ts.source, normalized_ts.raw_value
    return (
        coerce_market_timestamp(tick.timestamp),
        getattr(tick, "timestamp_source", "tick.timestamp"),
        tick.timestamp,
    )


@dataclass(slots=True, init=False)
class CandleEngine:
    """Build deterministic 1-minute candles from validated ticks.

    ``df`` is a read-only compatibility alias for ``get_df()``. The getter
    returns a defensive copy; mutating that DataFrame does not mutate this
    engine. Whole-history replacement remains supported through ``engine.df =``
    assignment or the explicit ``replace_history()`` method. Arbitrary pandas
    in-place mutation is intentionally not proxied because the deque is the live
    candle source of truth.
    """

    interval: str = "1min"
    max_bars: int = 500
    current_candle: dict[str, Any] | None = None
    last_candle_close: datetime | None = None
    _last_tick_ts: pd.Timestamp | None = None
    symbol: str | None = None
    _completed_candles: deque[dict[str, Any]] = field(init=False, repr=False)
    _df_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _df_cache_dirty: bool = field(default=True, init=False, repr=False)
    _df_cache_rebuild_total: int = field(default=0, init=False)
    _live_append_total: int = field(default=0, init=False)
    _same_minute_idempotent_total: int = field(default=0, init=False)
    _same_minute_conflict_total: int = field(default=0, init=False)

    def __init__(
        self,
        interval: str = "1min",
        max_bars: int = 500,
        df: pd.DataFrame | None = None,
        current_candle: dict[str, Any] | None = None,
        last_candle_close: datetime | None = None,
        symbol: str | None = None,
    ) -> None:
        self.interval = interval
        self.max_bars = max_bars
        self.current_candle = current_candle
        self.last_candle_close = last_candle_close
        self._last_tick_ts = None
        self.symbol = symbol
        self._completed_candles = deque(maxlen=int(self.max_bars))
        self._df_cache = None
        self._df_cache_dirty = True
        self._df_cache_rebuild_total = 0
        self._live_append_total = 0
        self._same_minute_idempotent_total = 0
        self._same_minute_conflict_total = 0
        if df is not None and not df.empty:
            self.replace_history(df)

    @property
    def df(self) -> pd.DataFrame:
        """Read-only compatibility alias for ``get_df()``.

        The returned DataFrame is a defensive copy. Mutating it with ``loc``,
        ``iloc``, column assignment, or ``inplace=True`` operations does not
        mutate CandleEngine. Use ``replace_history()`` or ``engine.df = frame``
        for whole-frame replacement.
        """
        return self.get_df()

    @df.setter
    def df(self, value: pd.DataFrame | None) -> None:
        self.replace_history(value)

    def replace_history(self, frame: pd.DataFrame | None) -> None:
        """Sanitize and atomically replace the bounded candle store."""
        self._replace_completed_candles(frame)

    def _replace_completed_candles(self, value: pd.DataFrame | None) -> None:
        self._completed_candles.clear()
        clean = (
            sanitize(value)
            if value is not None
            else pd.DataFrame(columns=_OHLC_COLUMNS)
        )
        for row in clean.tail(self.max_bars).to_dict(orient="records"):
            ts = _to_ist_timestamp(row.get("timestamp"))
            if pd.isna(ts):
                continue
            normalized = _normalize_completed_candle(
                row, symbol=self.symbol or "symbol_unset", incoming_ts=ts
            )
            if normalized is not None:
                self._completed_candles.append(normalized)
        self._df_cache_dirty = True

    def _cached_df_copy(self) -> pd.DataFrame:
        if self._df_cache_dirty or self._df_cache is None:
            self._df_cache = pd.DataFrame(
                list(self._completed_candles), columns=list(_OHLC_COLUMNS)
            )
            self._df_cache_dirty = False
            self._df_cache_rebuild_total += 1
        return self._df_cache.copy(deep=True)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "candle_store_size": len(self._completed_candles),
            "candle_store_maxlen": self._completed_candles.maxlen,
            "df_cache_dirty": self._df_cache_dirty,
            "df_cache_rebuild_total": self._df_cache_rebuild_total,
            "live_append_total": self._live_append_total,
            "same_minute_idempotent_total": self._same_minute_idempotent_total,
            "same_minute_conflict_total": self._same_minute_conflict_total,
        }

    def on_tick(self, tick: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.interval != "1min":
            raise ValueError(f"Unsupported interval: {self.interval}")
        try:
            symbol = _tick_symbol(tick, self.symbol)
        except ValueError:
            raw_ts_for_key = _tick_value(tick, "timestamp") or _tick_value(
                tick, "exchange_timestamp"
            )
            log_throttled(
                LOGGER,
                f"candle_tick_missing_symbol:{raw_ts_for_key!r}",
                "CANDLE_TICK_DROPPED reason=missing_symbol",
                interval_sec=10.0,
                level=logging.WARNING,
                extra={
                    "event": "candle_tick_dropped",
                    "reason": "missing_symbol",
                    "raw_ts": repr(raw_ts_for_key),
                    "bypass_filters": True,
                },
            )
            return None
        self.symbol = symbol
        try:
            timestamp, timestamp_source, raw_ts = _tick_timestamp(tick)
        except ValueError:
            raw_ts_for_key = _tick_value(tick, "timestamp") or _tick_value(
                tick, "exchange_timestamp"
            )
            log_throttled(
                LOGGER,
                f"candle_tick_bad_timestamp:{symbol}:{raw_ts_for_key!r}",
                f"CANDLE_TICK_DROPPED reason=bad_timestamp symbol={symbol}",
                interval_sec=10.0,
                level=logging.WARNING,
                extra={
                    "event": "candle_tick_dropped",
                    "symbol": symbol,
                    "reason": "bad_timestamp",
                    "raw_ts": repr(raw_ts_for_key),
                    "bypass_filters": True,
                },
            )
            return None
        if pd.isna(timestamp) or getattr(timestamp, "year", 1970) < 2020:
            log_throttled(
                LOGGER,
                f"candle_tick_bad_timestamp:{symbol}:{raw_ts!r}",
                f"CANDLE_TICK_DROPPED reason=bad_timestamp symbol={symbol}",
                interval_sec=10.0,
                level=logging.WARNING,
                extra={
                    "event": "candle_tick_dropped",
                    "symbol": symbol,
                    "reason": "bad_timestamp",
                    "raw_ts": repr(raw_ts),
                    "bypass_filters": True,
                },
            )
            return None
        now_ist = pd.Timestamp.now(tz=IST)
        if _is_future_timestamp(timestamp, now=now_ist):
            future_by_sec = future_delta_seconds(timestamp, now=now_ist)
            log_throttled(
                LOGGER,
                f"candle_tick_future:{symbol}:{timestamp.isoformat()}",
                (
                    "CANDLE_TICK_DROPPED reason=future_timestamp symbol=%s raw_ts=%r "
                    "tick_ts_ist=%s now_ist=%s future_by_sec=%.3f timestamp_source=%s"
                )
                % (
                    symbol,
                    raw_ts,
                    timestamp.isoformat(),
                    now_ist.isoformat(),
                    future_by_sec,
                    timestamp_source,
                ),
                interval_sec=30.0,
                level=logging.WARNING,
                extra={
                    "event": "candle_tick_dropped",
                    "reason": "future_timestamp",
                    "symbol": symbol,
                    "raw_ts": repr(raw_ts),
                    "tick_ts_ist": timestamp.isoformat(),
                    "now_ist": now_ist.isoformat(),
                    "future_by_sec": float(future_by_sec),
                    "timestamp_source": timestamp_source,
                    "bypass_filters": True,
                },
            )
            return None
        validated = tick if isinstance(tick, Tick) else validate_tick(dict(tick))
        payload = validated.to_dict()
        payload["symbol"] = symbol
        payload["timestamp"] = timestamp
        payload["timestamp_source"] = timestamp_source
        minute = timestamp.floor("1min")

        if self._last_tick_ts is not None:
            last_ts = _to_ist_timestamp(self._last_tick_ts)
            if not pd.isna(last_ts) and timestamp < last_ts:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_{symbol}",
                    "tick_out_of_order symbol=%s tick_ts=%s last_ts=%s"
                    % (symbol, timestamp.isoformat(), last_ts.isoformat()),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None

        finalized: dict[str, Any] | None = None
        if self.current_candle is None:
            self.current_candle = self._start_candle(payload, minute)
            LOGGER.debug(
                "candle_created",
                extra={
                    "event": "candle_created",
                    "symbol": symbol,
                    "minute": minute.isoformat(),
                },
            )
        else:
            current_minute = _to_ist_timestamp(self.current_candle["timestamp"])
            if minute < current_minute:
                log_throttled(
                    LOGGER,
                    f"tick_out_of_order_minute_{symbol}",
                    "tick_out_of_order symbol=%s tick_minute=%s current_minute=%s"
                    % (symbol, minute.isoformat(), current_minute.isoformat()),
                    interval_sec=10.0,
                    level=logging.DEBUG,
                )
                return None
            if minute > current_minute:
                finalized = self._finalize_current_candle()
                self.current_candle = self._start_candle(payload, minute)
                LOGGER.debug(
                    "candle_created",
                    extra={
                        "event": "candle_created",
                        "symbol": symbol,
                        "minute": minute.isoformat(),
                    },
                )
            else:
                self._update_candle(payload)
        self._last_tick_ts = timestamp
        return finalized

    def _start_candle(
        self, tick: Mapping[str, Any], minute: pd.Timestamp
    ) -> dict[str, Any]:
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
        if self.current_candle is None:
            return None
        candle = dict(self.current_candle)
        _validate_ohlc_row(candle)
        incoming_ts = _to_ist_timestamp(candle.get("timestamp"))
        if pd.isna(incoming_ts):
            raise DataIntegrityError("candle timestamp is invalid")
        now_ist = pd.Timestamp.now(tz=IST)
        symbol = self.symbol or "symbol_unset"
        if _is_future_timestamp(incoming_ts, now=now_ist):
            future_by_sec = future_delta_seconds(incoming_ts, now=now_ist)
            log_throttled(
                LOGGER,
                f"candle_future_{symbol}",
                (
                    "future_candle_rejected symbol=%s incoming_ts=%s now_ist=%s "
                    "future_by_sec=%.3f source=candle_engine"
                )
                % (symbol, incoming_ts.isoformat(), now_ist.isoformat(), future_by_sec),
                interval_sec=30.0,
                level=logging.WARNING,
                extra={
                    "event": "future_candle_rejected",
                    "symbol": symbol,
                    "incoming_ts": incoming_ts.isoformat(),
                    "now_ist": now_ist.isoformat(),
                    "future_by_sec": float(future_by_sec),
                    "source": "candle_engine",
                },
            )
            self.current_candle = None
            return None
        if self._completed_candles:
            last = self._completed_candles[-1]
            last_ts = _to_ist_timestamp(last.get("timestamp"))
            if not pd.isna(last_ts):
                if incoming_ts < last_ts:
                    log_throttled(
                        LOGGER,
                        f"candle_out_of_order_{symbol}",
                        "candle_out_of_order symbol=%s incoming_ts=%s last_ts=%s source=candle_engine"
                        % (symbol, incoming_ts.isoformat(), last_ts.isoformat()),
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "candle_out_of_order",
                            "symbol": symbol,
                            "incoming_ts": incoming_ts.isoformat(),
                            "last_ts": last_ts.isoformat(),
                            "source": "candle_engine",
                        },
                    )
                    raise DataIntegrityError("candle timestamps must be monotonic")
        candle["timestamp"] = incoming_ts
        normalized = _normalize_completed_candle(
            candle, symbol=symbol, incoming_ts=incoming_ts
        )
        if normalized is None:
            log_throttled(
                LOGGER,
                f"candle_invalid_{symbol}",
                "invalid_live_candle_rejected symbol=%s ts=%s"
                % (symbol, incoming_ts.isoformat()),
                interval_sec=30.0,
                level=logging.WARNING,
                extra={"event": "invalid_live_candle_rejected", "symbol": symbol},
            )
            self.current_candle = None
            return None
        if self._completed_candles:
            last = self._completed_candles[-1]
            last_ts = _to_ist_timestamp(last.get("timestamp"))
            if not pd.isna(last_ts) and incoming_ts == last_ts:
                if _candles_equivalent(last, normalized):
                    self._same_minute_idempotent_total += 1
                    self.current_candle = None
                    return None
                self._same_minute_conflict_total += 1
                log_throttled(
                    LOGGER,
                    f"finalized_candle_conflict_{symbol}_{incoming_ts.isoformat()}",
                    "FINALIZED_CANDLE_CONFLICT symbol=%s timestamp=%s"
                    % (symbol, incoming_ts.isoformat()),
                    interval_sec=30.0,
                    level=logging.ERROR,
                    extra={
                        "event": "FINALIZED_CANDLE_CONFLICT",
                        "symbol": symbol,
                        "timestamp": incoming_ts.isoformat(),
                        "stored_ohlcv": {
                            k: last.get(k)
                            for k in ("open", "high", "low", "close", "volume")
                        },
                        "incoming_ohlcv": {
                            k: normalized.get(k)
                            for k in ("open", "high", "low", "close", "volume")
                        },
                    },
                )
                self.current_candle = None
                raise DataIntegrityError(
                    f"conflicting finalized candle symbol={symbol} timestamp={incoming_ts}"
                )
        # Live completed candles are stored in a bounded row-oriented deque.
        # Avoid DataFrame row insertion/concat here; pandas frames are rebuilt
        # lazily only for compatibility readers.
        self._completed_candles.append(normalized)
        self._df_cache_dirty = True
        self._live_append_total += 1
        self.last_candle_close = incoming_ts.to_pydatetime()
        LOGGER.debug(
            "candle_finalized",
            extra={
                "event": "candle_finalized",
                "symbol": symbol,
                "timestamp": incoming_ts.isoformat(),
            },
        )
        return dict(normalized)

    def flush(self) -> dict[str, Any] | None:
        finalized = self._finalize_current_candle()
        if finalized is not None:
            self.current_candle = None
        return finalized

    def get_df(self) -> pd.DataFrame:
        """Return a defensive canonical OHLCV copy."""
        return self._cached_df_copy()


def _normalize_completed_candle(
    candle: Mapping[str, Any],
    *,
    symbol: str,
    incoming_ts: pd.Timestamp,
) -> dict[str, Any] | None:
    """O(1) normalizer for ONE completed live candle (no DataFrame work).

    Full-frame sanitize() remains reserved for bootstrap/persisted-history/
    repair paths; running it per live candle was O(history) pandas work at
    every minute boundary across all active symbols (production: lag_ms=851,
    tick_pending=1716 at market open).
    """
    out: dict[str, Any] = {"timestamp": incoming_ts}
    for ohlcv_field in ("open", "high", "low", "close"):
        try:
            value = float(candle.get(ohlcv_field))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        out[ohlcv_field] = value
    try:
        volume = float(candle.get("volume") or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(volume) or volume < 0.0:
        return None
    out["volume"] = volume
    if out["high"] < max(out["open"], out["close"]):
        return None
    if out["low"] > min(out["open"], out["close"]):
        return None
    if out["high"] < out["low"]:
        return None
    return out


def _candles_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if _to_ist_timestamp(left.get("timestamp")) != _to_ist_timestamp(
        right.get("timestamp")
    ):
        return False
    for ohlcv_field in ("open", "high", "low", "close", "volume"):
        try:
            if not math.isclose(
                float(left.get(ohlcv_field)),
                float(right.get(ohlcv_field)),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                return False
        except (TypeError, ValueError):
            return False
    return True


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


def repair_with_backfill(
    symbol: str,
    df: pd.DataFrame,
    *,
    fetch_recent_rest: FetchRecentFn,
    max_bars: int = 500,
) -> pd.DataFrame:
    LOGGER.info(
        "repair_with_backfill_deprecated symbol=%s",
        symbol,
        extra={"event": "repair_with_backfill_deprecated", "symbol": symbol},
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
    for attempt in range(1, retries + 1):
        frame = sanitize(fetch_historical(symbol))
        if validate_dataframe(frame, min_required=min_required):
            return frame
        LOGGER.warning(
            "data_integrity_error symbol=%s reason=historical_validation_failed attempt=%d",
            symbol,
            attempt,
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
    normalized = sanitize(df)
    ts = _series_to_ist(normalized["timestamp"])
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
    del fetch_recent_rest
    frame = sanitize(engine.get_df())
    if validate_dataframe(frame, min_required=min_required):
        return frame
    hydrated = fetch_historical_safe(
        symbol, fetch_historical=fetch_historical, min_required=min_required, retries=1
    )
    if hydrated is None:
        LOGGER.error(
            "data_integrity_error symbol=%s reason=insufficient_historical min_required=%d",
            symbol,
            min_required,
            extra={
                "event": "data_integrity_error",
                "symbol": symbol,
                "reason": "insufficient_historical",
                "min_required": min_required,
            },
        )
        return None
    engine.replace_history(hydrated.tail(engine.max_bars).reset_index(drop=True))
    return engine.get_df()
