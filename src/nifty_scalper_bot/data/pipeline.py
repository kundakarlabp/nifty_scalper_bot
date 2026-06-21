"""Deterministic single-path market data pipeline.

Design contract
---------------
* TickValidator  – rejects malformed ticks with logged error + counter; no silent drops.
* CandleBuilder  – time-bucket (1-min floor) per symbol; emits ONLY closed candles.
* CandleStore    – ring-buffer (deque) per symbol; initialized ONCE, never reset.
* Pipeline       – composes the three above; exposes a single ``on_tick`` entry point.

Rules (matches spec)
--------------------
* Only CLOSED candles are emitted downstream.
* No partial candles passed forward.
* Ticks with timestamp < last seen timestamp → DROP + ERROR log.
* OHLC invariants enforced on every closed candle; violating candles are dropped.
* Strategy gate: ``pipeline.candles_ready(symbol, n)`` → ``len >= n``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)

MIN_REQUIRED_CANDLES: int = 50  # hard strategy gate


# ---------------------------------------------------------------------------
# Counters (simple thread-safe ints; no new dependencies)
# ---------------------------------------------------------------------------
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


def get_dropped_ticks() -> int:
    return _DROPPED_TICKS.value


def get_dropped_candles() -> int:
    return _DROPPED_CANDLES.value


# ---------------------------------------------------------------------------
# Validated tick
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ValidatedTick:
    symbol: str
    timestamp: pd.Timestamp
    ltp: float
    volume: float = 0.0


# ---------------------------------------------------------------------------
# Closed candle
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Candle:
    symbol: str
    timestamp: pd.Timestamp   # minute-bucket open time
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


# ---------------------------------------------------------------------------
# STEP 1a: TickValidator
# ---------------------------------------------------------------------------
class TickValidator:
    """Validate raw tick dicts. Rejects silently-corrupt payloads with error log."""

    def validate(self, raw: Mapping[str, Any]) -> Optional[ValidatedTick]:
        """Return ValidatedTick or None (never raises). Errors are logged + counted."""
        try:
            symbol_raw = raw.get("symbol") or raw.get("trading_symbol")
            if not symbol_raw or str(symbol_raw).strip() == "":
                raise DataIntegrityError("missing symbol")

            ts_raw = raw.get("exchange_timestamp") or raw.get("timestamp") or raw.get("ts")
            if ts_raw is None:
                raise DataIntegrityError("missing timestamp")
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.isna(ts):
                raise DataIntegrityError(f"unparseable timestamp: {ts_raw!r}")

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
                symbol=str(symbol_raw).strip().upper(),
                timestamp=ts,
                ltp=ltp,
                volume=volume,
            )
        except (DataIntegrityError, ValueError, TypeError) as exc:
            _DROPPED_TICKS.increment()
            LOGGER.error(
                "tick_rejected",
                extra={
                    "event": "tick_rejected",
                    "reason": str(exc),
                    "total_dropped": _DROPPED_TICKS.value,
                },
            )
            return None


# ---------------------------------------------------------------------------
# STEP 2: CandleBuilder (time-bucket, one active candle per symbol)
# ---------------------------------------------------------------------------
class CandleBuilder:
    """Build closed 1-minute candles from validated ticks.

    * One active candle per symbol.
    * Bucket boundary = floor(tick.timestamp, '1min').
    * On bucket change → finalize previous candle EXACTLY ONCE.
    * Out-of-order tick (ts < last_ts) → DROP + ERROR.
    * OHLC invariant validated before emission; violating candles are dropped.
    """

    def __init__(self) -> None:
        self._active: dict[str, dict[str, Any]] = {}   # symbol → active candle
        self._last_ts: dict[str, pd.Timestamp] = {}    # symbol → last tick ts
        self._lock = threading.Lock()

    def on_tick(self, tick: ValidatedTick) -> Optional[Candle]:
        """Return a closed Candle when minute boundary crossed, else None."""
        with self._lock:
            return self._process(tick)

    def _process(self, tick: ValidatedTick) -> Optional[Candle]:
        sym = tick.symbol
        ts = tick.timestamp
        # Performance optimization: .replace is much faster than .floor for pandas timestamps
        minute = ts.replace(second=0, microsecond=0, nanosecond=0)

        # STEP 2 guard: monotonic timestamp enforcement
        last = self._last_ts.get(sym)
        if last is not None and ts < last:
            _DROPPED_TICKS.increment()
            log_throttled(
                LOGGER,
                f"tick_out_of_order:{sym}",
                (
                    f"tick_out_of_order symbol={sym} "
                    f"tick_ts={ts.isoformat()} last_ts={last.isoformat()} "
                    f"total_dropped={_DROPPED_TICKS.value}"
                ),
                interval_sec=30.0,
                level=logging.DEBUG,
                extra={
                    "event": "tick_out_of_order",
                    "symbol": sym,
                    "tick_ts": ts.isoformat(),
                    "last_ts": last.isoformat(),
                    "total_dropped": _DROPPED_TICKS.value,
                },
            )
            return None
        self._last_ts[sym] = ts

        active = self._active.get(sym)
        closed: Optional[Candle] = None

        if active is None:
            # First tick for this symbol
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        elif minute > pd.Timestamp(active["timestamp"]):
            # Bucket boundary crossed → finalize
            closed = _finalize(active)
            self._active[sym] = _init_candle(sym, minute, tick.ltp, tick.volume)
        else:
            # Same minute — update
            _update_candle(active, tick.ltp, tick.volume)

        if closed is not None:
            if not _check_ohlc(closed):
                _DROPPED_CANDLES.increment()
                LOGGER.error(
                    "candle_ohlc_violation",
                    extra={
                        "event": "candle_ohlc_violation",
                        "symbol": sym,
                        "candle": closed.as_dict(),
                        "total_dropped": _DROPPED_CANDLES.value,
                    },
                )
                return None
            LOGGER.debug(
                "candle_closed",
                extra={"event": "candle_closed", "symbol": sym,
                       "ts": closed.timestamp.isoformat()},
            )
        return closed

    def flush(self, symbol: str) -> Optional[Candle]:
        """Force-close active candle for symbol (e.g. market close)."""
        with self._lock:
            active = self._active.pop(symbol, None)
            if active is None:
                return None
            candle = _finalize(active)
            if candle and not _check_ohlc(candle):
                _DROPPED_CANDLES.increment()
                LOGGER.error(
                    "candle_ohlc_violation",
                    extra={"event": "candle_ohlc_violation", "symbol": symbol,
                           "candle": candle.as_dict(),
                           "total_dropped": _DROPPED_CANDLES.value},
                )
                return None
            return candle


def _init_candle(
    sym: str, minute: pd.Timestamp, price: float, volume: float
) -> dict[str, Any]:
    return {
        "symbol": sym,
        "timestamp": minute,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": volume,
    }


def _update_candle(candle: dict[str, Any], price: float, volume: float) -> None:
    candle["high"] = max(float(candle["high"]), price)
    candle["low"] = min(float(candle["low"]), price)
    candle["close"] = price
    candle["volume"] = float(candle["volume"]) + volume


def _finalize(candle: dict[str, Any]) -> Candle:
    return Candle(
        symbol=str(candle["symbol"]),
        timestamp=pd.Timestamp(candle["timestamp"]),
        open=float(candle["open"]),
        high=float(candle["high"]),
        low=float(candle["low"]),
        close=float(candle["close"]),
        volume=float(candle["volume"]),
    )


def _check_ohlc(c: Candle) -> bool:
    """STEP 9: OHLC integrity. Returns False if violated."""
    ok = (
        c.open > 0
        and c.high >= c.open
        and c.high >= c.close
        and c.low <= c.open
        and c.low <= c.close
        and c.low > 0
    )
    return ok


# ---------------------------------------------------------------------------
# STEP 3 / STEP 6: CandleStore — ring-buffer, initialized ONCE, never reset
# ---------------------------------------------------------------------------
class CandleStore:
    """Thread-safe ring-buffer of closed Candle objects per symbol.

    Initialized ONCE at startup. Never reset during runtime.
    Readiness check: ``len(store.get(symbol)) >= MIN_REQUIRED_CANDLES``
    — no boolean flags.
    """

    def __init__(self, maxlen: int = 1500) -> None:
        self._maxlen = maxlen
        self._store: Dict[str, deque[Candle]] = {}
        self._lock = threading.Lock()

    def push(self, candle: Candle) -> None:
        with self._lock:
            if candle.symbol not in self._store:
                self._store[candle.symbol] = deque(maxlen=self._maxlen)
            self._store[candle.symbol].append(candle)

    def get(self, symbol: str) -> list[Candle]:
        with self._lock:
            return list(self._store.get(symbol, []))

    def candles_ready(self, symbol: str, min_required: int = MIN_REQUIRED_CANDLES) -> bool:
        """STEP 7 gate: no flags — just count >= min_required."""
        return len(self.get(symbol)) >= min_required

    def seed(self, symbol: str, bars: list[dict[str, Any]]) -> None:
        """Seed store from historical OHLCV dicts (hydration path).

        Called ONCE per symbol at startup. No-op if already seeded.
        """
        with self._lock:
            if symbol in self._store and len(self._store[symbol]) > 0:
                return  # already seeded — never overwrite
            buf: deque[Candle] = deque(maxlen=self._maxlen)
            for row in bars:
                try:
                    ts_raw = row.get("timestamp") or row.get("date")
                    ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    c = Candle(
                        symbol=symbol,
                        # Performance optimization: .replace is much faster than .floor for pandas timestamps
                        timestamp=ts.replace(second=0, microsecond=0, nanosecond=0),
                        open=float(row.get("open") or row.get("close") or 0),
                        high=float(row.get("high") or row.get("close") or 0),
                        low=float(row.get("low") or row.get("close") or 0),
                        close=float(row.get("close") or 0),
                        volume=float(row.get("volume") or 0),
                    )
                    if not _check_ohlc(c):
                        continue
                    buf.append(c)
                except Exception:
                    continue
            self._store[symbol] = buf
        LOGGER.info(
            "candle_store_seeded",
            extra={
                "event": "candle_store_seeded",
                "symbol": symbol,
                "bars": len(self._store[symbol]),
            },
        )

    def to_dataframe(self, symbol: str) -> "pd.DataFrame":
        import pandas as _pd
        candles = self.get(symbol)
        if not candles:
            return _pd.DataFrame()
        rows = [c.as_dict() for c in candles]
        return _pd.DataFrame(rows)

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


# ---------------------------------------------------------------------------
# Main Pipeline — single entry point
# ---------------------------------------------------------------------------
class MarketDataPipeline:
    """Deterministic single-path pipeline: tick → validate → candle → store.

    Usage::

        pipeline = MarketDataPipeline()
        # seed from hydration
        pipeline.store.seed("NFO:NIFTY2632422500CE", historical_bars)
        # process live tick
        candle = pipeline.on_tick(raw_tick_dict)
        # strategy gate
        if pipeline.candles_ready("NFO:NIFTY2632422500CE", 50):
            ...
    """

    def __init__(self, store_maxlen: int = 1500) -> None:
        self.validator = TickValidator()
        self.builder = CandleBuilder()
        self.store = CandleStore(maxlen=store_maxlen)

    def on_tick(self, raw: Mapping[str, Any]) -> Optional[Candle]:
        """Single entry point. Returns closed Candle or None.

        Flow: raw dict → TickValidator → CandleBuilder → CandleStore → return candle
        Errors are logged with counters; never raise to caller.
        """
        tick = self.validator.validate(raw)
        if tick is None:
            return None
        candle = self.builder.on_tick(tick)
        if candle is not None:
            self.store.push(candle)
        return candle

    def candles_ready(self, symbol: str, min_required: int = MIN_REQUIRED_CANDLES) -> bool:
        """STEP 7: strategy gate — pure count, no boolean flags."""
        return self.store.candles_ready(symbol, min_required)

    def get_candles(self, symbol: str) -> list[Candle]:
        return self.store.get(symbol)

    def flush(self, symbol: str) -> Optional[Candle]:
        """Force-close active candle and push to store."""
        candle = self.builder.flush(symbol)
        if candle is not None:
            self.store.push(candle)
        return candle


# Singleton for the process — runner imports this
_PIPELINE: Optional[MarketDataPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline(store_maxlen: int = 1500) -> MarketDataPipeline:
    """Return process-level singleton pipeline (initialized once)."""
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = MarketDataPipeline(store_maxlen=store_maxlen)
    return _PIPELINE


def pipeline_health() -> dict:
    """Return a snapshot of pipeline health for logging and /status endpoints.

    Returns:
        dict with dropped_ticks, dropped_candles, per-symbol candle counts,
        ready_symbols (>= MIN_REQUIRED_CANDLES), total_symbols.
    """
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
