"""Deterministic single-path market data pipeline."""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
MIN_REQUIRED_CANDLES: int = 50


class _Counter:
    def __init__(self) -> None:
        self._v = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._v += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._v


_DROPPED_TICKS = _Counter()
_DROPPED_CANDLES = _Counter()


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
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"unparseable timestamp: {value!r}") from exc
    if pd.isna(ts):
        raise DataIntegrityError(f"unparseable timestamp: {value!r}")
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def _future_grace_seconds() -> float:
    try:
        return max(float(os.getenv("MARKETDATA_MAX_FUTURE_CANDLE_SECONDS", "120") or 120), 0.0)
    except (TypeError, ValueError):
        return 120.0


def _overlap_seconds() -> float:
    try:
        return max(float(os.getenv("PIPELINE_CANDLE_OVERLAP_TOLERANCE_SECONDS", "120") or 120), 0.0)
    except (TypeError, ValueError):
        return 120.0


def _is_future_timestamp(ts: pd.Timestamp, *, now: pd.Timestamp | None = None) -> bool:
    now_ts = now or pd.Timestamp.now(tz=IST)
    return bool(ts > now_ts + pd.Timedelta(seconds=_future_grace_seconds()))


def _log_future_rejected(symbol: str, ts: pd.Timestamp, source: str) -> None:
    _DROPPED_CANDLES.increment()
    log_throttled(
        LOGGER,
        f"future_candle_rejected:{symbol}:{source}",
        f"future_candle_rejected symbol={symbol} incoming_ts={ts.isoformat()} source={source}",
        interval_sec=30.0,
        level=logging.WARNING,
        extra={"event": "future_candle_rejected", "symbol": symbol, "incoming_ts": ts.isoformat(), "source": source, "total_dropped": _DROPPED_CANDLES.value},
    )


def get_dropped_ticks() -> int:
    return _DROPPED_TICKS.value


def get_dropped_candles() -> int:
    return _DROPPED_CANDLES.value


@dataclass(frozen=True, slots=True)
class ValidatedTick:
    symbol: str
    timestamp: pd.Timestamp
    ltp: float
    volume: float = 0.0


@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "timestamp": self.timestamp, "open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume}


class TickValidator:
    def validate(self, raw: Mapping[str, Any]) -> Optional[ValidatedTick]:
        try:
            symbol_raw = raw.get("symbol") or raw.get("trading_symbol")
            if not symbol_raw or str(symbol_raw).strip() == "":
                raise DataIntegrityError("missing symbol")
            ts_raw = raw.get("exchange_timestamp") or raw.get("timestamp") or raw.get("ts")
            if ts_raw is None:
                raise DataIntegrityError("missing timestamp")
            ts = _to_ist_timestamp(ts_raw)
            if _is_future_timestamp(ts):
                _DROPPED_TICKS.increment()
                log_throttled(
                    LOGGER,
                    f"future_tick_rejected:{symbol_raw}",
                    f"future_tick_rejected symbol={symbol_raw} tick_ts={ts.isoformat()}",
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={"event": "future_tick_rejected", "symbol": str(symbol_raw).strip().upper(), "tick_ts": ts.isoformat(), "total_dropped": _DROPPED_TICKS.value},
                )
                return None
            ltp_raw = raw.get("ltp") or raw.get("last_price") or raw.get("close")
            if ltp_raw is None:
                raise DataIntegrityError("missing ltp/last_price")
            ltp = float(ltp_raw)
            if ltp <= 0:
                raise DataIntegrityError(f"invalid price: {ltp}")
            if "volume_delta" in raw:
                volume_raw = raw.get("volume_delta")
            elif "volume" in raw:
                volume_raw = raw.get("volume")
            elif "vol" in raw:
                volume_raw = raw.get("vol")
            else:
                volume_raw = 0.0
            try:
                volume = float(volume_raw if volume_raw is not None else 0.0)
            except (TypeError, ValueError):
                volume = 0.0
            return ValidatedTick(symbol=str(symbol_raw).strip().upper(), timestamp=ts, ltp=ltp, volume=volume)
        except (DataIntegrityError, ValueError, TypeError) as exc:
            _DROPPED_TICKS.increment()
            LOGGER.error("tick_rejected", extra={"event": "tick_rejected", "reason": str(exc), "total_dropped": _DROPPED_TICKS.value})
            return None


class CandleBuilder:
    def __init__(self) -> None:
        self._active: dict[str, dict[str, Any]] = {}
        self._last_ts: dict[str, pd.Timestamp] = {}
        self._lock = threading.Lock()

    def on_tick(self, tick: ValidatedTick) -> Optional[Candle]:
        with self._lock:
            return self._process(tick)

    def _process(self, tick: ValidatedTick) -> Optional[Candle]:
        sym = tick.symbol
        ts = tick.timestamp
        minute = ts.floor("1min")
        last = self._last_ts.get(sym)
        if last is not None and ts < last:
            _DROPPED_TICKS.increment()
            log_throttled(LOGGER, f"tick_out_of_order:{sym}", f"tick_out_of_order symbol={sym} tick_ts={ts.isoformat()} last_ts={last.isoformat()} total_dropped={_DROPPED_TICKS.value}", interval_sec=30.0, level=logging.DEBUG, extra={"event": "tick_out_of_order", "symbol": sym, "tick_ts": ts.isoformat(), "last_ts": last.isoformat(), "total_dropped": _DROPPED_TICKS.value})
            return None
        self._last_ts[sym] = ts
        active = self._active.get(sym)
        closed: Optional[Candle] = None
        if active is None:
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        elif minute < pd.Timestamp(active["timestamp"]):
            _DROPPED_TICKS.increment()
            log_throttled(LOGGER, f"tick_late_bucket:{sym}", f"tick_out_of_order symbol={sym} tick_minute={minute.isoformat()} current_minute={pd.Timestamp(active['timestamp']).isoformat()} total_dropped={_DROPPED_TICKS.value}", interval_sec=30.0, level=logging.DEBUG, extra={"event": "tick_out_of_order", "symbol": sym, "tick_ts": ts.isoformat(), "tick_minute": minute.isoformat(), "last_ts": last.isoformat() if last is not None else None, "source": "candle_builder", "total_dropped": _DROPPED_TICKS.value})
            return None
        elif minute > pd.Timestamp(active["timestamp"]):
            closed = _finalize(active)
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        else:
            _update_candle(active, tick.ltp, tick.volume)
        if closed is not None:
            if not _check_ohlc(closed):
                _DROPPED_CANDLES.increment()
                LOGGER.error("candle_ohlc_violation", extra={"event": "candle_ohlc_violation", "symbol": sym, "candle": closed.as_dict(), "total_dropped": _DROPPED_CANDLES.value})
                return None
            LOGGER.debug("candle_closed", extra={"event": "candle_closed", "symbol": sym, "ts": closed.timestamp.isoformat()})
        return closed

    def flush(self, symbol: str) -> Optional[Candle]:
        with self._lock:
            active = self._active.pop(symbol, None)
            if active is None:
                return None
            candle = _finalize(active)
            if candle and not _check_ohlc(candle):
                _DROPPED_CANDLES.increment()
                LOGGER.error("candle_ohlc_violation", extra={"event": "candle_ohlc_violation", "symbol": symbol, "candle": candle.as_dict(), "total_dropped": _DROPPED_CANDLES.value})
                return None
            return candle


def _init_candle(sym: str, minute: pd.Timestamp, price: float, volume: float) -> dict[str, Any]:
    return {"symbol": sym, "timestamp": minute, "open": price, "high": price, "low": price, "close": price, "volume": volume}


def _update_candle(candle: dict[str, Any], price: float, volume: float) -> None:
    candle["high"] = max(float(candle["high"]), price)
    candle["low"] = min(float(candle["low"]), price)
    candle["close"] = price
    candle["volume"] = float(candle["volume"]) + volume


def _finalize(candle: dict[str, Any]) -> Candle:
    return Candle(symbol=str(candle["symbol"]), timestamp=_to_ist_timestamp(candle["timestamp"]), open=float(candle["open"]), high=float(candle["high"]), low=float(candle["low"]), close=float(candle["close"]), volume=float(candle["volume"]))


def _normalize_candle_timestamp(candle: Candle) -> Candle:
    return Candle(symbol=candle.symbol, timestamp=_to_ist_timestamp(candle.timestamp), open=candle.open, high=candle.high, low=candle.low, close=candle.close, volume=candle.volume)


def _check_ohlc(c: Candle) -> bool:
    return bool(c.open > 0 and c.high >= c.open and c.high >= c.close and c.low <= c.open and c.low <= c.close and c.low > 0)


class CandleStore:
    def __init__(self, maxlen: int = 1500) -> None:
        self._maxlen = maxlen
        self._store: Dict[str, deque[Candle]] = {}
        self._lock = threading.Lock()

    def push(self, candle: Candle) -> None:
        with self._lock:
            normalized_candle = _normalize_candle_timestamp(candle)
            incoming_ts = normalized_candle.timestamp.floor("1min")
            if _is_future_timestamp(incoming_ts):
                _log_future_rejected(normalized_candle.symbol, incoming_ts, "candle_store")
                raise DataIntegrityError("future candle timestamp")
            if normalized_candle.symbol not in self._store:
                self._store[normalized_candle.symbol] = deque(maxlen=self._maxlen)
            buf = self._store[normalized_candle.symbol]
            if buf:
                last_ts = _to_ist_timestamp(buf[-1].timestamp).floor("1min")
                if incoming_ts == last_ts:
                    return
                if incoming_ts < last_ts:
                    age_delta = max(0.0, (last_ts - incoming_ts).total_seconds())
                    if age_delta <= _overlap_seconds():
                        LOGGER.debug("candle_store_overlap_duplicate symbol=%s incoming_ts=%s last_ts=%s age_delta_s=%.1f", normalized_candle.symbol, incoming_ts.isoformat(), last_ts.isoformat(), age_delta, extra={"event": "candle_store_overlap_duplicate", "symbol": normalized_candle.symbol, "incoming_ts": incoming_ts.isoformat(), "last_ts": last_ts.isoformat(), "age_delta_s": age_delta, "source": "candle_store", "reason": "hydration_live_boundary_overlap"})
                        return
                    _DROPPED_CANDLES.increment()
                    log_throttled(LOGGER, f"candle_store_out_of_order:{normalized_candle.symbol}", f"candle_store_out_of_order symbol={normalized_candle.symbol} incoming_ts={incoming_ts.isoformat()} last_ts={last_ts.isoformat()} source=candle_store", interval_sec=30.0, level=logging.WARNING, extra={"event": "candle_store_out_of_order", "symbol": normalized_candle.symbol, "incoming_ts": incoming_ts.isoformat(), "last_ts": last_ts.isoformat(), "source": "candle_store", "total_dropped": _DROPPED_CANDLES.value})
                    raise DataIntegrityError("candle store timestamps must be monotonic")
            buf.append(normalized_candle)

    def get(self, symbol: str) -> list[Candle]:
        with self._lock:
            return list(self._store.get(symbol, []))

    def candles_ready(self, symbol: str, min_required: int = MIN_REQUIRED_CANDLES) -> bool:
        return len(self.get(symbol)) >= min_required

    def seed(self, symbol: str, bars: list[dict[str, Any]]) -> None:
        with self._lock:
            if symbol in self._store and len(self._store[symbol]) > 0:
                return
            by_minute: dict[pd.Timestamp, Candle] = {}
            for row in bars:
                try:
                    ts_raw = row.get("timestamp") or row.get("date")
                    ts = _to_ist_timestamp(ts_raw).floor("1min")
                    if _is_future_timestamp(ts):
                        _log_future_rejected(symbol, ts, "seed")
                        continue
                    c = Candle(symbol=symbol, timestamp=ts, open=float(row.get("open") or row.get("close") or 0), high=float(row.get("high") or row.get("close") or 0), low=float(row.get("low") or row.get("close") or 0), close=float(row.get("close") or 0), volume=float(row.get("volume") or 0))
                    if not _check_ohlc(c):
                        LOGGER.warning("candle_store_seed_rejected", extra={"event": "candle_store_seed_rejected", "symbol": symbol, "incoming_ts": c.timestamp.isoformat(), "source": "seed", "reason": "ohlc_invalid"})
                        continue
                    by_minute[c.timestamp] = c
                except (TypeError, ValueError, DataIntegrityError) as exc:
                    LOGGER.warning("candle_store_seed_rejected", extra={"event": "candle_store_seed_rejected", "symbol": symbol, "source": "seed", "reason": type(exc).__name__})
            self._store[symbol] = deque((by_minute[key] for key in sorted(by_minute)), maxlen=self._maxlen)
        LOGGER.info("candle_store_seeded", extra={"event": "candle_store_seeded", "symbol": symbol, "bars": len(self._store[symbol])})

    def to_dataframe(self, symbol: str) -> "pd.DataFrame":
        import pandas as _pd
        candles = self.get(symbol)
        if not candles:
            return _pd.DataFrame()
        return _pd.DataFrame([c.as_dict() for c in candles])

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


class MarketDataPipeline:
    def __init__(self, store_maxlen: int = 1500) -> None:
        self.validator = TickValidator()
        self.builder = CandleBuilder()
        self.store = CandleStore(maxlen=store_maxlen)

    def on_tick(self, raw: Mapping[str, Any]) -> Optional[Candle]:
        tick = self.validator.validate(raw)
        if tick is None:
            return None
        candle = self.builder.on_tick(tick)
        if candle is not None:
            try:
                self.store.push(candle)
            except DataIntegrityError as exc:
                log_throttled(LOGGER, f"pipeline_candle_store_rejected:{candle.symbol}", f"pipeline_candle_store_rejected symbol={candle.symbol} incoming_ts={candle.timestamp.isoformat()} error_type={type(exc).__name__} reason={exc} source=market_data_pipeline", interval_sec=30.0, level=logging.WARNING, extra={"event": "pipeline_candle_store_rejected", "symbol": candle.symbol, "incoming_ts": candle.timestamp.isoformat(), "error_type": type(exc).__name__, "reason": str(exc), "source": "market_data_pipeline"})
                return None
        return candle

    def candles_ready(self, symbol: str, min_required: int = MIN_REQUIRED_CANDLES) -> bool:
        return self.store.candles_ready(symbol, min_required)

    def get_candles(self, symbol: str) -> list[Candle]:
        return self.store.get(symbol)

    def flush(self, symbol: str) -> Optional[Candle]:
        candle = self.builder.flush(symbol)
        if candle is not None:
            self.store.push(candle)
        return candle


_PIPELINE: Optional[MarketDataPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline(store_maxlen: int = 1500) -> MarketDataPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = MarketDataPipeline(store_maxlen=store_maxlen)
    return _PIPELINE


def pipeline_health() -> dict:
    pl = get_pipeline()
    syms = pl.store.symbols()
    candle_counts = {sym: len(pl.store.get(sym)) for sym in syms}
    ready = [sym for sym, cnt in candle_counts.items() if cnt >= MIN_REQUIRED_CANDLES]
    return {"dropped_ticks": get_dropped_ticks(), "dropped_candles": get_dropped_candles(), "total_symbols": len(syms), "ready_symbols": len(ready), "candle_counts": candle_counts, "ready": ready}
