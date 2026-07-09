"""Deterministic single-path market data pipeline."""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

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
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
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
        return coerce_market_timestamp(value)
    except ValueError as exc:
        raise DataIntegrityError(f"unparseable timestamp: {value!r}") from exc


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
    return is_future_market_timestamp(ts, now=now_ts, grace_seconds=_future_grace_seconds())


def _log_future_rejected(symbol: str, ts: pd.Timestamp, source: str) -> None:
    _DROPPED_CANDLES.increment()
    now_ist = pd.Timestamp.now(tz=IST)
    future_by_sec = future_delta_seconds(ts, now=now_ist)
    log_throttled(
        LOGGER,
        f"future_candle_rejected:{symbol}:{source}",
        (
            "future_candle_rejected symbol=%s incoming_ts=%s now_ist=%s "
            "future_by_sec=%.3f source=%s"
        )
        % (symbol, ts.isoformat(), now_ist.isoformat(), future_by_sec, source),
        interval_sec=30.0,
        level=logging.WARNING,
        extra={
            "event": "future_candle_rejected",
            "symbol": symbol,
            "incoming_ts": ts.isoformat(),
            "now_ist": now_ist.isoformat(),
            "future_by_sec": float(future_by_sec),
            "source": source,
            "total_dropped": _DROPPED_CANDLES.value,
        },
    )


def _log_candle_store_out_of_order(
    *,
    symbol: str,
    incoming_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    incoming_candle: "Candle",
    last_candle: "Candle | None",
    source: str,
    store_size: int,
    age_delta_s: float,
) -> None:
    """Emit high-signal diagnostics for monotonicity failures."""

    _DROPPED_CANDLES.increment()
    last_close = float(last_candle.close) if last_candle is not None else None
    extra = {
        "event": "candle_store_out_of_order",
        "symbol": symbol,
        "incoming_ts": incoming_ts.isoformat(),
        "last_ts": last_ts.isoformat(),
        "incoming_close": float(incoming_candle.close),
        "last_close": last_close,
        "incoming_volume": float(incoming_candle.volume),
        "age_delta_s": float(age_delta_s),
        "store_size": int(store_size),
        "source": source,
        "reason": "incoming_before_last_store_ts",
        "total_dropped": _DROPPED_CANDLES.value,
        "bypass_filters": True,
    }
    log_throttled(
        LOGGER,
        f"candle_store_out_of_order:{symbol}:{incoming_ts.isoformat()}:{last_ts.isoformat()}",
        (
            "candle_store_out_of_order symbol=%s incoming_ts=%s last_ts=%s "
            "age_delta_s=%.1f incoming_close=%s last_close=%s store_size=%d source=%s"
        )
        % (
            symbol,
            incoming_ts.isoformat(),
            last_ts.isoformat(),
            age_delta_s,
            float(incoming_candle.close),
            last_close,
            store_size,
            source,
        ),
        interval_sec=30.0,
        level=logging.WARNING,
        extra=extra,
    )


def _log_pipeline_store_rejected(candle: "Candle", exc: DataIntegrityError, *, source: str) -> None:
    log_throttled(
        LOGGER,
        (
            f"pipeline_candle_store_rejected:{candle.symbol}:"
            f"{source}:{candle.timestamp.isoformat()}"
        ),
        (
            "pipeline_candle_store_rejected symbol=%s incoming_ts=%s "
            "error_type=%s reason=%s source=%s"
        )
        % (
            candle.symbol,
            candle.timestamp.isoformat(),
            type(exc).__name__,
            exc,
            source,
        ),
        interval_sec=30.0,
        level=logging.WARNING,
        extra={
            "event": "pipeline_candle_store_rejected",
            "symbol": candle.symbol,
            "incoming_ts": candle.timestamp.isoformat(),
            "incoming_close": float(candle.close),
            "incoming_volume": float(candle.volume),
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "source": source,
            "bypass_filters": True,
        },
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
    timestamp_source: str = "unknown"
    raw_timestamp: Any = None


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
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class TickValidator:
    def validate(self, raw: Mapping[str, Any]) -> Optional[ValidatedTick]:
        try:
            symbol = normalized_symbol(raw.get("symbol") or raw.get("trading_symbol"))
            normalized_ts = normalize_market_tick_timestamp(raw)
            ts = normalized_ts.timestamp
            now_ist = pd.Timestamp.now(tz=IST)
            if _is_future_timestamp(ts, now=now_ist):
                _DROPPED_TICKS.increment()
                future_by_sec = future_delta_seconds(ts, now=now_ist)
                log_throttled(
                    LOGGER,
                    f"future_tick_rejected:{symbol}:{ts.isoformat()}",
                    (
                        "future_tick_rejected symbol=%s raw_ts=%r tick_ts_ist=%s now_ist=%s "
                        "future_by_sec=%.3f timestamp_source=%s"
                    )
                    % (
                        symbol,
                        normalized_ts.raw_value,
                        ts.isoformat(),
                        now_ist.isoformat(),
                        future_by_sec,
                        normalized_ts.source,
                    ),
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "future_tick_rejected",
                        "symbol": symbol,
                        "raw_ts": repr(normalized_ts.raw_value),
                        "tick_ts": ts.isoformat(),
                        "tick_ts_ist": ts.isoformat(),
                        "now_ist": now_ist.isoformat(),
                        "future_by_sec": float(future_by_sec),
                        "timestamp_source": normalized_ts.source,
                        "source": str(raw.get("source") or "tick_validator"),
                        "total_dropped": _DROPPED_TICKS.value,
                        "bypass_filters": True,
                    },
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
            return ValidatedTick(
                symbol=symbol,
                timestamp=ts,
                ltp=ltp,
                volume=volume,
                timestamp_source=normalized_ts.source,
                raw_timestamp=normalized_ts.raw_value,
            )
        except (DataIntegrityError, ValueError, TypeError) as exc:
            _DROPPED_TICKS.increment()
            LOGGER.error(
                "tick_rejected",
                extra={
                    "event": "tick_rejected",
                    "reason": str(exc),
                    "symbol": str(raw.get("symbol") or raw.get("trading_symbol") or "").strip().upper() or None,
                    "timestamp_source": "missing_or_invalid",
                    "total_dropped": _DROPPED_TICKS.value,
                },
            )
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
        if last is not None and _is_future_timestamp(last):
            log_throttled(
                LOGGER,
                f"future_last_ts_quarantined:{sym}",
                f"future_last_ts_quarantined symbol={sym} last_ts={last.isoformat()} incoming_ts={ts.isoformat()}",
                interval_sec=30.0,
                level=logging.WARNING,
                extra={
                    "event": "future_last_ts_quarantined",
                    "symbol": sym,
                    "last_ts": last.isoformat(),
                    "incoming_ts": ts.isoformat(),
                    "source": "candle_builder",
                },
            )
            self._last_ts.pop(sym, None)
            last = None
        if last is not None and ts < last:
            _DROPPED_TICKS.increment()
            log_throttled(
                LOGGER,
                f"tick_out_of_order:{sym}",
                (
                    "tick_out_of_order symbol=%s tick_ts=%s last_ts=%s "
                    "age_delta_s=%.1f ltp=%s volume=%s total_dropped=%d"
                )
                % (sym, ts.isoformat(), last.isoformat(), (last - ts).total_seconds(), tick.ltp, tick.volume, _DROPPED_TICKS.value),
                interval_sec=30.0,
                level=logging.DEBUG,
                extra={
                    "event": "tick_out_of_order",
                    "symbol": sym,
                    "tick_ts": ts.isoformat(),
                    "last_ts": last.isoformat(),
                    "age_delta_s": (last - ts).total_seconds(),
                    "ltp": float(tick.ltp),
                    "volume": float(tick.volume),
                    "source": "candle_builder",
                    "total_dropped": _DROPPED_TICKS.value,
                },
            )
            return None
        self._last_ts[sym] = ts
        active = self._active.get(sym)
        closed: Optional[Candle] = None
        if active is None:
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        elif minute < pd.Timestamp(active["timestamp"]):
            _DROPPED_TICKS.increment()
            active_minute = pd.Timestamp(active["timestamp"])
            log_throttled(
                LOGGER,
                f"tick_late_bucket:{sym}",
                (
                    "tick_out_of_order symbol=%s tick_minute=%s current_minute=%s "
                    "last_ts=%s ltp=%s source=candle_builder total_dropped=%d"
                )
                % (
                    sym,
                    minute.isoformat(),
                    active_minute.isoformat(),
                    last.isoformat() if last is not None else None,
                    tick.ltp,
                    _DROPPED_TICKS.value,
                ),
                interval_sec=30.0,
                level=logging.DEBUG,
                extra={
                    "event": "tick_out_of_order",
                    "symbol": sym,
                    "tick_ts": ts.isoformat(),
                    "tick_minute": minute.isoformat(),
                    "current_minute": active_minute.isoformat(),
                    "last_ts": last.isoformat() if last is not None else None,
                    "ltp": float(tick.ltp),
                    "volume": float(tick.volume),
                    "source": "candle_builder",
                    "total_dropped": _DROPPED_TICKS.value,
                },
            )
            return None
        elif minute > pd.Timestamp(active["timestamp"]):
            closed = _finalize(active)
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        else:
            _update_candle(active, tick.ltp, tick.volume)
        if closed is not None:
            if not _check_ohlc(closed):
                _DROPPED_CANDLES.increment()
                LOGGER.error(
                    "candle_ohlc_violation",
                    extra={"event": "candle_ohlc_violation", "symbol": sym, "candle": closed.as_dict(), "total_dropped": _DROPPED_CANDLES.value},
                )
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
                LOGGER.error(
                    "candle_ohlc_violation",
                    extra={"event": "candle_ohlc_violation", "symbol": symbol, "candle": candle.as_dict(), "total_dropped": _DROPPED_CANDLES.value},
                )
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
    return Candle(
        symbol=str(candle["symbol"]),
        timestamp=_to_ist_timestamp(candle["timestamp"]),
        open=float(candle["open"]),
        high=float(candle["high"]),
        low=float(candle["low"]),
        close=float(candle["close"]),
        volume=float(candle["volume"]),
    )


def _normalize_candle_timestamp(candle: Candle) -> Candle:
    return Candle(
        symbol=candle.symbol,
        timestamp=_to_ist_timestamp(candle.timestamp),
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


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
                last_candle = buf[-1]
                last_ts = _to_ist_timestamp(last_candle.timestamp).floor("1min")
                if _is_future_timestamp(last_ts):
                    log_throttled(
                        LOGGER,
                        f"future_last_ts_quarantined:{normalized_candle.symbol}:store",
                        f"future_last_ts_quarantined symbol={normalized_candle.symbol} last_ts={last_ts.isoformat()} incoming_ts={incoming_ts.isoformat()} source=candle_store",
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "future_last_ts_quarantined",
                            "symbol": normalized_candle.symbol,
                            "last_ts": last_ts.isoformat(),
                            "incoming_ts": incoming_ts.isoformat(),
                            "last_close": float(last_candle.close),
                            "incoming_close": float(normalized_candle.close),
                            "store_size": len(buf),
                            "source": "candle_store",
                        },
                    )
                    buf.clear()
                    last_ts = None
                    last_candle = None
                if last_ts is not None and incoming_ts == last_ts:
                    return
                if last_ts is not None and incoming_ts < last_ts:
                    age_delta = max(0.0, (last_ts - incoming_ts).total_seconds())
                    if age_delta <= _overlap_seconds():
                        LOGGER.debug(
                            "candle_store_overlap_duplicate symbol=%s incoming_ts=%s last_ts=%s age_delta_s=%.1f",
                            normalized_candle.symbol,
                            incoming_ts.isoformat(),
                            last_ts.isoformat(),
                            age_delta,
                            extra={
                                "event": "candle_store_overlap_duplicate",
                                "symbol": normalized_candle.symbol,
                                "incoming_ts": incoming_ts.isoformat(),
                                "last_ts": last_ts.isoformat(),
                                "incoming_close": float(normalized_candle.close),
                                "last_close": float(last_candle.close) if last_candle is not None else None,
                                "age_delta_s": age_delta,
                                "store_size": len(buf),
                                "source": "candle_store",
                                "reason": "hydration_live_boundary_overlap",
                            },
                        )
                        return
                    _log_candle_store_out_of_order(
                        symbol=normalized_candle.symbol,
                        incoming_ts=incoming_ts,
                        last_ts=last_ts,
                        incoming_candle=normalized_candle,
                        last_candle=last_candle,
                        source="candle_store",
                        store_size=len(buf),
                        age_delta_s=age_delta,
                    )
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
                    c = Candle(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(row.get("open") or row.get("close") or 0),
                        high=float(row.get("high") or row.get("close") or 0),
                        low=float(row.get("low") or row.get("close") or 0),
                        close=float(row.get("close") or 0),
                        volume=float(row.get("volume") or 0),
                    )
                    if not _check_ohlc(c):
                        LOGGER.warning(
                            "candle_store_seed_rejected",
                            extra={"event": "candle_store_seed_rejected", "symbol": symbol, "incoming_ts": c.timestamp.isoformat(), "source": "seed", "reason": "ohlc_invalid"},
                        )
                        continue
                    by_minute[c.timestamp] = c
                except (TypeError, ValueError, DataIntegrityError) as exc:
                    LOGGER.warning(
                        "candle_store_seed_rejected",
                        extra={"event": "candle_store_seed_rejected", "symbol": symbol, "source": "seed", "reason": type(exc).__name__},
                    )
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
                _log_pipeline_store_rejected(candle, exc, source="market_data_pipeline")
                return None
        return candle

    def candles_ready(self, symbol: str, min_required: int = MIN_REQUIRED_CANDLES) -> bool:
        return self.store.candles_ready(symbol, min_required)

    def get_candles(self, symbol: str) -> list[Candle]:
        return self.store.get(symbol)

    def flush(self, symbol: str) -> Optional[Candle]:
        candle = self.builder.flush(symbol)
        if candle is not None:
            try:
                self.store.push(candle)
            except DataIntegrityError as exc:
                _log_pipeline_store_rejected(candle, exc, source="market_data_pipeline_flush")
                return None
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
    return {
        "dropped_ticks": get_dropped_ticks(),
        "dropped_candles": get_dropped_candles(),
        "total_symbols": len(syms),
        "ready_symbols": len(ready),
        "candle_counts": candle_counts,
        "ready": ready,
    }
