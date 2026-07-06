"""Runtime hardening hooks for MarketDataManager safety handling.

The hooks here are deliberately narrow.  They harden live-data failure modes
without changing contract selection, strategy logic, or order execution.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import queue
import threading
import time
from typing import Any, Mapping

import pandas as pd

_INSTALLED_ATTR = "_freshness_hardening_installed"
_ORIGINAL_INIT_ATTR = "_freshness_hardening_original_init"
_ORIGINAL_NORMALIZE_ATTR = "_freshness_hardening_original_normalize_ws_tick"
_ORIGINAL_FAST_RECORD_ATTR = "_freshness_hardening_original_record_ws_arrival_fast"
_ORIGINAL_ENQUEUE_ATTR = "_fallback_hardening_original_enqueue_tick_threadsafe"
_ORIGINAL_ENSURE_CONSUMER_ATTR = "_candle_flush_original_ensure_tick_consumer"
_ORIGINAL_STOP_ATTR = "_candle_flush_original_stop"

_SYNTHETIC_QUALITIES = {"synthetic", "unknown", "invalid"}
_ALLOWED_WS_SOURCES = {"ws", "ws_full", "full"}
_IST_TZ = "Asia/Kolkata"


def install_market_data_manager_hardening(manager_cls: type[Any]) -> None:
    """Install idempotent MDM market-data hardening hooks."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_init = getattr(manager_cls, "__init__", None)
    if callable(original_init):
        setattr(manager_cls, _ORIGINAL_INIT_ATTR, original_init)

        def _init_with_hardening(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            _initialise_hardening_state(self)

        setattr(manager_cls, "__init__", _init_with_hardening)

    original_normalize = getattr(manager_cls, "_normalize_ws_tick", None)
    if callable(original_normalize):
        setattr(manager_cls, _ORIGINAL_NORMALIZE_ATTR, original_normalize)

        def _normalize_ws_tick_with_quality(self: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
            quality = _timestamp_quality(raw)
            normalized = original_normalize(self, raw)
            if normalized is not None:
                normalized["timestamp_quality"] = quality
            return normalized

        setattr(manager_cls, "_normalize_ws_tick", _normalize_ws_tick_with_quality)

    original_fast_record = getattr(manager_cls, "_record_ws_arrival_fast", None)
    if callable(original_fast_record):
        setattr(manager_cls, _ORIGINAL_FAST_RECORD_ATTR, original_fast_record)

        def _record_ws_arrival_fast_with_quality(self: Any, *, symbol: str, token: int | None, ltp: Any, raw_tick: dict[str, Any]) -> None:
            quality = _timestamp_quality(raw_tick)
            original_fast_record(self, symbol=symbol, token=token, ltp=ltp, raw_tick=raw_tick)
            _tag_fast_cache_quality(self, symbol=symbol, token=token, quality=quality)

        setattr(manager_cls, "_record_ws_arrival_fast", _record_ws_arrival_fast_with_quality)

    original_enqueue = getattr(manager_cls, "_enqueue_tick_threadsafe", None)
    if callable(original_enqueue):
        setattr(manager_cls, _ORIGINAL_ENQUEUE_ATTR, original_enqueue)
        setattr(manager_cls, "_enqueue_tick_threadsafe", _enqueue_tick_threadsafe_hardened)

    original_ensure_consumer = getattr(manager_cls, "_ensure_tick_consumer", None)
    if callable(original_ensure_consumer):
        setattr(manager_cls, _ORIGINAL_ENSURE_CONSUMER_ATTR, original_ensure_consumer)

        def _ensure_tick_consumer_with_candle_flush(self: Any, reason: str) -> None:
            original_ensure_consumer(self, reason)
            self._ensure_candle_flush_task(reason=reason)

        setattr(manager_cls, "_ensure_tick_consumer", _ensure_tick_consumer_with_candle_flush)

    original_stop = getattr(manager_cls, "stop", None)
    if callable(original_stop):
        setattr(manager_cls, _ORIGINAL_STOP_ATTR, original_stop)

        def _stop_with_hardening(self: Any) -> None:
            self._stop_candle_flush_task()
            self._stop_fallback_tick_worker()
            original_stop(self)

        setattr(manager_cls, "stop", _stop_with_hardening)

    setattr(manager_cls, "has_fresh_ws_ltp", _has_fresh_ws_ltp_strict)
    setattr(manager_cls, "_ensure_tick_worker", _ensure_tick_worker_thread_queue)
    setattr(manager_cls, "_tick_worker_loop", _tick_worker_loop_thread_queue)
    setattr(manager_cls, "_put_fallback_tick_nowait", _put_fallback_tick_nowait)
    setattr(manager_cls, "_ensure_candle_flush_task", _ensure_candle_flush_task)
    setattr(manager_cls, "_stop_candle_flush_task", _stop_candle_flush_task)
    setattr(manager_cls, "_stop_fallback_tick_worker", _stop_fallback_tick_worker)
    setattr(manager_cls, "flush_due_candles", _flush_due_candles)
    setattr(manager_cls, _INSTALLED_ATTR, True)


def _initialise_hardening_state(self: Any) -> None:
    maxsize = max(int(getattr(self, "_tick_queue_maxsize", 10_000) or 10_000), 1)
    self._fallback_tick_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
    if not isinstance(getattr(self, "_tick_worker_stop", None), threading.Event):
        self._tick_worker_stop = threading.Event()
    self._candle_flush_task: asyncio.Task[None] | None = None
    self._candle_flush_interval_s = _float_env("MDM_CANDLE_FLUSH_INTERVAL_SECONDS", 1.0, minimum=0.25)
    self._candle_flush_grace_s = _float_env("MDM_CANDLE_FLUSH_GRACE_SECONDS", 1.5, minimum=0.0)
    self._last_candle_flush_log_mono = 0.0


def _float_env(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(float(value), float(minimum))


def _timestamp_quality(raw: Mapping[str, Any]) -> str:
    """Classify source timestamp quality before any synthetic fallback."""
    if _valid_timestamp(raw.get("exchange_timestamp")):
        return "exchange"
    if _valid_timestamp(raw.get("timestamp")):
        return "broker"
    if _valid_timestamp(raw.get("received_at")):
        return "received_at"
    return "synthetic"


def _valid_timestamp(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return False
    if pd.isna(ts):
        return False
    try:
        return int(ts.year) >= 2020
    except Exception:
        return False


def _tag_fast_cache_quality(self: Any, *, symbol: str, token: int | None, quality: str) -> None:
    try:
        canonical = self._canonical_symbol(symbol)
    except Exception:
        canonical = str(symbol or "")
    keys = {canonical}
    try:
        if self._is_nifty_spot_tick(canonical, token):
            keys.add("NSE:NIFTY")
    except Exception:
        pass
    lock = getattr(self, "_lock", None)
    if lock is None:
        return
    with lock:
        for key in keys:
            tick = self._latest_ticks.get(key)
            if isinstance(tick, dict):
                tick["timestamp_quality"] = quality
            cached = self._tick_cache.get(key)
            if isinstance(cached, dict):
                cached["timestamp_quality"] = quality


def _tick_age_seconds(tick_ts: Any, now: float) -> float | None:
    if isinstance(tick_ts, datetime):
        ts = tick_ts if tick_ts.tzinfo is not None else tick_ts.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    if tick_ts is None:
        return None
    try:
        raw = float(tick_ts)
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(tick_ts, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return max(now - float(pd.Timestamp(parsed).timestamp()), 0.0)
    if raw > 1e12:
        raw /= 1000.0
    if raw < 946684800:
        return None
    return max(now - raw, 0.0)


def _has_fresh_ws_ltp_strict(self: Any, symbols: list[str] | tuple[str, ...] | None = None, max_age_seconds: float = 5.0) -> bool:
    """Return fresh WS LTP proof only when timestamp quality is acceptable."""
    now = time.time()
    max_age = max(float(max_age_seconds), 0.1)
    with self._lock:
        candidate_symbols = (
            [self._canonical_symbol(sym) for sym in symbols]
            if symbols is not None
            else list(self._active_subscribed_symbols)
        )
        if not candidate_symbols:
            candidate_symbols = list(self._latest_ticks.keys())
        for symbol in candidate_symbols:
            tick = self._latest_ticks.get(symbol)
            if not isinstance(tick, Mapping):
                continue
            source = str(tick.get("source") or self._last_tick_source.get(symbol) or "").lower()
            if source not in _ALLOWED_WS_SOURCES:
                continue
            quality = str(tick.get("timestamp_quality") or "").lower()
            if quality in _SYNTHETIC_QUALITIES:
                continue
            price = tick.get("ltp", tick.get("last_price", tick.get("price", 0)))
            try:
                ltp = float(price)
            except (TypeError, ValueError):
                continue
            if ltp <= 0:
                continue
            age = _tick_age_seconds(tick.get("exchange_timestamp") or tick.get("timestamp"), now)
            if age is not None and age <= max_age:
                return True
    return False


def _enqueue_tick_threadsafe_hardened(self: Any, tick: dict[str, Any]) -> None:
    """Non-blocking WS enqueue with thread-safe fallback queue."""
    try:
        payload = dict(tick)
    except Exception:
        return
    payload.setdefault("_enqueued_monotonic", time.monotonic())

    loop = getattr(self, "_main_loop", None)
    if loop is not None and loop.is_running():
        self._enqueue_latest_tick_for_drain(payload, loop)
        return

    self._ensure_tick_worker()
    if not self._put_fallback_tick_nowait(payload):
        self._tick_queue_dropped += 1
        now = time.monotonic()
        if now - float(getattr(self, "_last_tick_queue_full_log", 0.0) or 0.0) > 5.0:
            self._last_tick_queue_full_log = now
            self._logger.warning(
                "MDM_FALLBACK_TICK_QUEUE_FULL dropped=%d",
                self._tick_queue_dropped,
                extra={"event": "MDM_FALLBACK_TICK_QUEUE_FULL", "dropped": self._tick_queue_dropped},
            )
    self._emit_priority_summaries_if_due()


def _ensure_tick_worker_thread_queue(self: Any) -> None:
    """Start fallback worker backed by queue.Queue, not asyncio.Queue."""
    if not isinstance(getattr(self, "_fallback_tick_queue", None), queue.Queue):
        maxsize = max(int(getattr(self, "_tick_queue_maxsize", 10_000) or 10_000), 1)
        self._fallback_tick_queue = queue.Queue(maxsize=maxsize)
    thread = getattr(self, "_tick_worker_thread", None)
    if thread is not None and thread.is_alive():
        return
    self._tick_worker_stop.clear()
    self._tick_worker_thread = threading.Thread(
        target=self._tick_worker_loop,
        name="mdm-tick-worker",
        daemon=True,
    )
    self._tick_worker_thread.start()
    self._logger.info(
        "MDM_FALLBACK_TICK_WORKER_STARTED queue_type=threading",
        extra={"event": "MDM_FALLBACK_TICK_WORKER_STARTED", "queue_type": "queue.Queue"},
    )


def _safe_queue_task_done(q: queue.Queue[Any]) -> None:
    try:
        q.task_done()
    except ValueError:
        pass


def _put_fallback_tick_nowait(self: Any, tick_payload: dict[str, Any]) -> bool:
    """Insert fallback tick with priority-aware eviction/coalescing."""
    q: queue.Queue[dict[str, Any]] = self._fallback_tick_queue
    symbol = self._resolve_tick_symbol_for_priority(tick_payload)
    priority, bucket = self._tick_priority(symbol)
    tick_payload["_mdm_priority"] = priority
    tick_payload["_mdm_priority_bucket"] = bucket
    if priority == 0:
        self._log_open_position_priority_if_needed(symbol)
    try:
        q.put_nowait(tick_payload)
        self._bus_priority_counts[bucket] += 1
        return True
    except queue.Full:
        pass

    retained: list[dict[str, Any]] = []
    removed = False
    removed_bucket = bucket
    removed_reason = ""
    while True:
        try:
            existing = q.get_nowait()
            _safe_queue_task_done(q)
        except queue.Empty:
            break
        existing_symbol = self._resolve_tick_symbol_for_priority(existing if isinstance(existing, Mapping) else {})
        existing_priority, existing_bucket = self._tick_priority(existing_symbol)
        if not removed and existing_priority > priority:
            removed = True
            removed_bucket = existing_bucket
            removed_reason = "drop_lower_priority"
            continue
        if not removed and existing_symbol == symbol and existing_priority >= priority:
            removed = True
            removed_bucket = existing_bucket
            removed_reason = "coalesce_same_symbol"
            continue
        retained.append(existing)

    for item in retained:
        try:
            q.put_nowait(item)
        except queue.Full:
            self._tick_queue_priority_drops[str(item.get("_mdm_priority_bucket") or "unknown")] += 1
            break

    if removed:
        if removed_reason == "coalesce_same_symbol":
            self._tick_queue_priority_coalesced[removed_bucket] += 1
        else:
            self._tick_queue_priority_drops[removed_bucket] += 1
    elif priority >= 1:
        self._tick_queue_priority_drops[bucket] += 1
        return False
    else:
        try:
            dropped = q.get_nowait()
            _safe_queue_task_done(q)
            dropped_symbol = self._resolve_tick_symbol_for_priority(dropped if isinstance(dropped, Mapping) else {})
            self._tick_queue_priority_drops["open_position"] += 1
            self._logger.critical(
                "MDM_OPEN_POSITION_TICK_FORCED_DROP dropped_symbol=%s incoming_symbol=%s",
                dropped_symbol,
                symbol,
                extra={
                    "event": "MDM_OPEN_POSITION_TICK_FORCED_DROP",
                    "dropped_symbol": dropped_symbol,
                    "incoming_symbol": symbol,
                },
            )
        except queue.Empty:
            pass

    try:
        q.put_nowait(tick_payload)
        self._bus_priority_counts[bucket] += 1
        return True
    except queue.Full:
        self._tick_queue_priority_drops[bucket] += 1
        return False


def _tick_worker_loop_thread_queue(self: Any) -> None:
    """Fallback serial tick worker used only before the asyncio loop is wired."""
    q: queue.Queue[dict[str, Any]] = self._fallback_tick_queue
    while not self._tick_worker_stop.is_set():
        try:
            raw = q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            self._process_queued_tick(raw)
            self._tick_processed_total += 1
        except Exception as exc:  # noqa: BLE001
            self._record_tick_drop(str(raw.get("_mdm_priority_bucket") or "unknown"), "fallback_process_error")
            self._logger.error(
                "MDM_FALLBACK_TICK_WORKER_ERROR error=%r",
                exc,
                exc_info=True,
                extra={"event": "MDM_FALLBACK_TICK_WORKER_ERROR", "error": repr(exc)},
            )
        finally:
            _safe_queue_task_done(q)


def _stop_fallback_tick_worker(self: Any) -> None:
    thread = getattr(self, "_tick_worker_thread", None)
    if thread is None:
        return
    self._tick_worker_stop.set()
    if thread.is_alive():
        thread.join(timeout=2.0)
    self._tick_worker_thread = None
    self._tick_worker_stop.clear()


def _ensure_candle_flush_task(self: Any, *, reason: str) -> None:
    """Start a loop task that finalizes idle one-minute candles."""
    loop = getattr(self, "_main_loop", None)
    if loop is None or not loop.is_running():
        return
    task = getattr(self, "_candle_flush_task", None)
    if task is not None and not task.done():
        return

    async def _runner() -> None:
        interval = max(float(getattr(self, "_candle_flush_interval_s", 1.0) or 1.0), 0.25)
        try:
            while True:
                await asyncio.sleep(interval)
                self.flush_due_candles()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive task guard
            self._logger.error(
                "MDM_CANDLE_FLUSH_TASK_STOPPED error=%r",
                exc,
                exc_info=True,
                extra={"event": "MDM_CANDLE_FLUSH_TASK_STOPPED", "error": repr(exc)},
            )

    def _start() -> None:
        task_inner = getattr(self, "_candle_flush_task", None)
        if task_inner is None or task_inner.done():
            self._candle_flush_task = loop.create_task(_runner())
            self._logger.info(
                "MDM_CANDLE_FLUSH_TASK_STARTED reason=%s",
                reason,
                extra={"event": "MDM_CANDLE_FLUSH_TASK_STARTED", "reason": reason},
            )

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        _start()
    else:
        loop.call_soon_threadsafe(_start)


def _stop_candle_flush_task(self: Any) -> None:
    task = getattr(self, "_candle_flush_task", None)
    if task is None or task.done():
        self._candle_flush_task = None
        return
    loop = getattr(self, "_main_loop", None)
    try:
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        else:
            task.cancel()
    except Exception as exc:  # noqa: BLE001
        self._logger.debug("candle flush task cancel skipped: %s", exc)
    self._candle_flush_task = None


def _coerce_ist_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(_IST_TZ)
    return ts.tz_convert(_IST_TZ)


def _flush_due_candles(self: Any, *, now: Any | None = None, grace_seconds: float | None = None) -> int:
    """Finalize candles whose minute has ended even if no next tick arrives."""
    grace = (
        max(float(grace_seconds), 0.0)
        if grace_seconds is not None
        else max(float(getattr(self, "_candle_flush_grace_s", 1.5) or 1.5), 0.0)
    )
    now_ts = _coerce_ist_timestamp(now) if now is not None else pd.Timestamp.now(tz=_IST_TZ)
    if now_ts is None:
        return 0

    engines = list(getattr(self, "_engines", {}).items())
    flushed = 0
    for symbol, engine in engines:
        current = getattr(engine, "current_candle", None)
        if not isinstance(current, Mapping):
            continue
        candle_minute = _coerce_ist_timestamp(current.get("timestamp"))
        if candle_minute is None:
            continue
        due_after = candle_minute + pd.Timedelta(minutes=1, seconds=grace)
        if now_ts < due_after:
            continue
        try:
            candle = engine.flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "MDM_CANDLE_CLOCK_FLUSH_FAILED symbol=%s error=%r",
                symbol,
                exc,
                exc_info=True,
                extra={"event": "MDM_CANDLE_CLOCK_FLUSH_FAILED", "symbol": symbol, "error": repr(exc)},
            )
            continue
        if not candle:
            continue
        bar = {
            "symbol": symbol,
            "timestamp": candle["timestamp"] if isinstance(candle, dict) else candle.timestamp,
            "open": float(candle["open"] if isinstance(candle, dict) else candle.open),
            "high": float(candle["high"] if isinstance(candle, dict) else candle.high),
            "low": float(candle["low"] if isinstance(candle, dict) else candle.low),
            "close": float(candle["close"] if isinstance(candle, dict) else candle.close),
            "volume": int(float((candle.get("volume", 0) if isinstance(candle, dict) else getattr(candle, "volume", 0)) or 0)),
            "source": "clock_flush_candle",
        }
        with self._lock:
            self._ohlc[symbol].append(bar)
        publisher = getattr(self, "_publish_closed_bar", None)
        if callable(publisher):
            try:
                publisher(bar)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("clock-flush bar publish skipped: %s", exc)
        flushed += 1
        now_mono = time.monotonic()
        if now_mono - float(getattr(self, "_last_candle_flush_log_mono", 0.0) or 0.0) >= 10.0:
            self._last_candle_flush_log_mono = now_mono
            self._logger.info(
                "MDM_CANDLE_CLOCK_FLUSHED symbol=%s close=%s",
                symbol,
                bar["close"],
                extra={"event": "MDM_CANDLE_CLOCK_FLUSHED", "symbol": symbol, "source": "clock_flush_candle"},
            )
    return flushed
