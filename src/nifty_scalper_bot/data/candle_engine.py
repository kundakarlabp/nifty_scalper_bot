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

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.validator import Tick, validate_tick
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
FetchHistoricalFn = Callable[[str], pd.DataFrame | None]
FetchRecentFn = Callable[[str], pd.DataFrame | None]
_OHLC_COLUMNS = ('timestamp', 'open', 'high', 'low', 'close', 'volume')



def sanitize(df: pd.DataFrame | None) -> pd.DataFrame:
    """Drop invalid OHLC rows without synthetic repair. Args: df. Returns: cleaned DataFrame. Raises: None."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLC_COLUMNS)
    cleaned = df.copy()
    for col in ('open', 'high', 'low', 'close', 'volume'):
        if col not in cleaned.columns:
            cleaned[col] = 0.0
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    cleaned['timestamp'] = pd.to_datetime(cleaned.get('timestamp'), utc=True, errors='coerce')
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
    cleaned = cleaned.drop_duplicates(subset='timestamp', keep='last')
    cleaned = cleaned.sort_values('timestamp').reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped > 0:
        LOGGER.warning('tick_invalid', extra={'event': 'tick_invalid', 'dropped_rows': dropped})
    return cleaned[list(_OHLC_COLUMNS)]


@dataclass(slots=True)
class CandleEngine:
    """Build deterministic 1-minute candles from validated ticks. Args: interval/max_bars. Returns: engine. Raises: ValueError."""

    interval: str = '1min'
    max_bars: int = 500
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_OHLC_COLUMNS))
    current_candle: dict[str, Any] | None = None
    last_candle_close: datetime | None = None
    _last_tick_ts: pd.Timestamp | None = None

    def on_tick(self, tick: Mapping[str, Any]) -> dict[str, Any] | None:
        """Ingest one tick. Args: tick. Returns: finalized candle|None. Raises: DataIntegrityError."""
        if self.interval != '1min':
            raise ValueError(f'Unsupported interval: {self.interval}')

        raw_ts = tick.get('timestamp') if isinstance(tick, dict) else getattr(tick, 'timestamp', None)
        timestamp = pd.to_datetime(raw_ts, utc=True, errors='coerce')
        if pd.isna(timestamp) or getattr(timestamp, 'year', 1970) < 2020:
            log_throttled(LOGGER, 'candle_tick_bad_timestamp', 'CANDLE_TICK_DROPPED reason=bad_timestamp', interval_sec=10.0, level=logging.WARNING)
            return None
        validated = tick if isinstance(tick, Tick) else validate_tick(dict(tick))
        payload = validated.to_dict()
        timestamp = pd.Timestamp(timestamp)
        # ⚡ Bolt: Replace expensive .floor("1min") with faster .replace() for 1-min bucketing
        minute = timestamp.replace(second=0, microsecond=0, nanosecond=0)

        if self._last_tick_ts is not None:
            last_ts = pd.to_datetime(self._last_tick_ts, utc=True, errors='coerce')
            if not pd.isna(last_ts) and timestamp < last_ts:
                log_throttled(LOGGER, f"tick_out_of_order_{getattr(self, 'symbol', 'unknown')}", "tick_out_of_order symbol=%s tick_ts=%s last_ts=%s" % (getattr(self, 'symbol', 'unknown'), timestamp.isoformat(), last_ts.isoformat()), interval_sec=10.0, level=logging.DEBUG)
                return None

        finalized: dict[str, Any] | None = None
        if self.current_candle is None:
            self.current_candle = self._start_candle(payload, minute)
            LOGGER.debug('candle_created', extra={'event': 'candle_created', 'minute': minute.isoformat()})
        else:
            current_minute = pd.Timestamp(self.current_candle['timestamp'])
            if minute < current_minute:
                log_throttled(LOGGER, f"tick_out_of_order_minute_{getattr(self, 'symbol', 'unknown')}", "tick_out_of_order symbol=%s tick_minute=%s current_minute=%s" % (getattr(self, 'symbol', 'unknown'), minute.isoformat(), current_minute.isoformat()), interval_sec=10.0, level=logging.DEBUG)
                return None
            if minute > current_minute:
                finalized = self._finalize_current_candle()
                self.current_candle = self._start_candle(payload, minute)
                LOGGER.debug('candle_created', extra={'event': 'candle_created', 'minute': minute.isoformat()})
            else:
                self._update_candle(payload)

        self._last_tick_ts = timestamp
        return finalized

    def _start_candle(self, tick: Mapping[str, Any], minute: pd.Timestamp) -> dict[str, Any]:
        """Initialize a minute candle. Args: tick/minute. Returns: candle. Raises: None."""
        price = float(tick['ltp'])
        return {
            'timestamp': minute,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': float(tick.get('volume') or 0.0),
        }

    def _update_candle(self, tick: Mapping[str, Any]) -> None:
        """Update active candle OHLC. Args: tick. Returns: None. Raises: DataIntegrityError."""
        if self.current_candle is None:
            raise DataIntegrityError('missing current candle')
        price = float(tick['ltp'])
        self.current_candle['high'] = max(float(self.current_candle['high']), price)
        self.current_candle['low'] = min(float(self.current_candle['low']), price)
        self.current_candle['close'] = price
        self.current_candle['volume'] = float(self.current_candle['volume']) + float(tick.get('volume') or 0.0)

    def _finalize_current_candle(self) -> dict[str, Any] | None:
        """Finalize active candle. Args: none. Returns: candle|None. Raises: DataIntegrityError."""
        if self.current_candle is None:
            return None
        candle = dict(self.current_candle)
        _validate_ohlc_row(candle)
        new_row = pd.DataFrame([candle]).dropna(how='all')
        if new_row.empty:
            return None
        if self.df is None or self.df.empty:
            frame = new_row.reset_index(drop=True)
        else:
            existing = self.df.dropna(how='all')
            if existing.empty:
                frame = new_row.reset_index(drop=True)
            else:
                frame = pd.concat([existing, new_row], ignore_index=True)
        frame = sanitize(frame).tail(self.max_bars).reset_index(drop=True)
        if frame['timestamp'].duplicated().any():
            raise DataIntegrityError('duplicate candle timestamps')
        if not frame['timestamp'].is_monotonic_increasing:
            raise DataIntegrityError('candle timestamps must be monotonic')
        self.df = frame
        self.last_candle_close = pd.to_datetime(candle['timestamp'], utc=True).to_pydatetime()
        LOGGER.debug('candle_finalized', extra={'event': 'candle_finalized', 'timestamp': pd.Timestamp(candle['timestamp']).isoformat()})
        return candle

    def flush(self) -> dict[str, Any] | None:
        """Flush active candle into finalized store. Args: none. Returns: candle|None. Raises: DataIntegrityError."""
        return self._finalize_current_candle()

    def get_df(self) -> pd.DataFrame:
        """Return finalized candles. Args: none. Returns: DataFrame. Raises: None."""
        return self.df.copy(deep=True)


def _validate_ohlc_row(row: Mapping[str, Any]) -> None:
    """Validate OHLC consistency invariants. Args: row. Returns: None. Raises: DataIntegrityError."""
    op = float(row['open'])
    hi = float(row['high'])
    lo = float(row['low'])
    cl = float(row['close'])
    if hi < max(op, cl):
        raise DataIntegrityError('high must be >= open/close')
    if lo > min(op, cl):
        raise DataIntegrityError('low must be <= open/close')


def detect_gap(df: pd.DataFrame) -> bool:
    """Detect missing 1-minute buckets. Args: df. Returns: bool. Raises: None."""
    clean = sanitize(df)
    if len(clean) < 2:
        return False
    diffs = clean['timestamp'].diff().dropna()
    return bool((diffs != pd.Timedelta(minutes=1)).any())


def repair_with_backfill(
    symbol: str,
    df: pd.DataFrame,
    *,
    fetch_recent_rest: FetchRecentFn,
    max_bars: int = 500,
) -> pd.DataFrame:
    """Merge deterministic recent candles only (no fill/backfill). Args: symbol/df/fetch_recent_rest/max_bars. Returns: DataFrame. Raises: DataIntegrityError."""
    LOGGER.info('data_integrity_error', extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'repair_with_backfill_deprecated'})
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
    """Fetch historical candles with explicit retries and logs. Args: symbol/fetch_historical/min_required/retries/sleep_seconds. Returns: DataFrame|None. Raises: None."""
    for attempt in range(1, retries + 1):
        frame = sanitize(fetch_historical(symbol))
        if validate_dataframe(frame, min_required=min_required):
            return frame
        LOGGER.warning(
            'data_integrity_error',
            extra={'event': 'data_integrity_error', 'symbol': symbol, 'attempt': attempt, 'reason': 'historical_validation_failed'},
        )
        if attempt < retries:
            time.sleep(sleep_seconds)
    return None


def validate_dataframe(df: pd.DataFrame | None, min_required: int = 50) -> bool:
    """Validate OHLC frame integrity. Args: df/min_required. Returns: bool. Raises: None."""
    if df is None:
        return False
    clean = sanitize(df)
    if len(clean) < min_required:
        return False
    if clean['timestamp'].duplicated().any() or not clean['timestamp'].is_monotonic_increasing:
        return False
    if detect_gap(clean):
        return False
    try:
        for row in clean.to_dict(orient='records'):
            _validate_ohlc_row(row)
    except DataIntegrityError:
        return False
    return True


def normalize_ohlc_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp index to Asia/Kolkata. Args: df. Returns: DataFrame. Raises: DataIntegrityError."""
    normalized = sanitize(df)
    ts = pd.to_datetime(normalized['timestamp'], errors='coerce', utc=True)
    if ts.isna().any():
        raise DataIntegrityError('Invalid timestamp in normalize_ohlc_timezone')
    normalized['timestamp'] = ts.dt.tz_convert('Asia/Kolkata')
    return normalized.set_index('timestamp')


def ensure_valid_data(
    symbol: str,
    engine: CandleEngine,
    *,
    fetch_historical: FetchHistoricalFn,
    fetch_recent_rest: FetchRecentFn,
    min_required: int = 50,
) -> pd.DataFrame | None:
    """Ensure candle cache integrity with startup hydration only. Args: symbol/engine/fetch_historical/fetch_recent_rest/min_required. Returns: DataFrame|None. Raises: None."""
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
        LOGGER.error('data_integrity_error', extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'insufficient_historical'})
        return None
    engine.df = hydrated.tail(engine.max_bars).reset_index(drop=True)
    return engine.get_df()
