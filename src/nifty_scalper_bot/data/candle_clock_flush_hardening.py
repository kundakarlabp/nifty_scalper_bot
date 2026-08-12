"""Race-safe clock finalization for MarketDataManager candle engines."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.data.candle_state_hardening import reconcile_stale_current

_IST_TZ = "Asia/Kolkata"
_INSTALLED_ATTR = "_candle_clock_flush_hardening_installed"
_RESERVED_MINUTES_ATTR = "_candle_tick_reserved_minutes"
_RESERVED_TICKS_ATTR = "_candle_tick_reserved_ids"
_LAST_PUBLISHED: dict[int, dict[str, pd.Timestamp]] = defaultdict(dict)


def _coerce_ist_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(_IST_TZ)
    return timestamp.tz_convert(_IST_TZ)


def _raw_tick_minute(tick: Mapping[str, Any]) -> pd.Timestamp | None:
    for key in ("exchange_timestamp", "timestamp", "last_trade_time"):
        value = tick.get(key)
        if value in (None, ""):
            continue
        parsed = _coerce_ist_timestamp(value)
        if parsed is not None:
            return parsed.floor("1min")
    return None


def _tick_reservation_key(
    manager: Any, tick: Mapping[str, Any]
) -> tuple[str, pd.Timestamp] | None:
    """Return canonical symbol/minute identity for a popped-but-unapplied tick."""
    symbol = str(tick.get("symbol") or "")
    if not symbol:
        try:
            token = int(tick.get("instrument_token") or tick.get("token") or 0)
        except (TypeError, ValueError):
            token = 0
        symbol = str((getattr(manager, "_symbol_by_token", {}) or {}).get(token) or "")
    if not symbol:
        return None
    canonicalize = getattr(manager, "_canonical_symbol", None)
    canonical = canonicalize(symbol) if callable(canonicalize) else symbol
    minute = _raw_tick_minute(tick)
    if not canonical or minute is None:
        return None
    return canonical, minute


def _reserve_popped_tick_locked(manager: Any, tick: Mapping[str, Any]) -> None:
    """Keep a local-batch tick visible to the clock finalization guard."""
    tick_id = id(tick)
    reserved_ids = getattr(manager, _RESERVED_TICKS_ATTR, None)
    if not isinstance(reserved_ids, dict):
        reserved_ids = {}
        setattr(manager, _RESERVED_TICKS_ATTR, reserved_ids)
    # Budget exhaustion may requeue the same dict and pop it again later. That
    # must remain one reservation, not increment the minute count twice.
    if tick_id in reserved_ids:
        return
    key = _tick_reservation_key(manager, tick)
    if key is None:
        return
    reservations = getattr(manager, _RESERVED_MINUTES_ATTR, None)
    if not isinstance(reservations, dict):
        reservations = {}
        setattr(manager, _RESERVED_MINUTES_ATTR, reservations)
    reservations[key] = int(reservations.get(key, 0) or 0) + 1
    reserved_ids[tick_id] = key


def _release_popped_tick_locked(manager: Any, tick: Mapping[str, Any]) -> None:
    """Release one local-batch reservation after its processing attempt ends."""
    reserved_ids = getattr(manager, _RESERVED_TICKS_ATTR, None)
    if not isinstance(reserved_ids, dict):
        return
    key = reserved_ids.pop(id(tick), None)
    if key is None:
        return
    reservations = getattr(manager, _RESERVED_MINUTES_ATTR, None)
    if not isinstance(reservations, dict):
        return
    remaining = int(reservations.get(key, 0) or 0) - 1
    if remaining > 0:
        reservations[key] = remaining
    else:
        reservations.pop(key, None)


def _has_unapplied_tick_for_minute(
    manager: Any, symbol: str, expected_minute: pd.Timestamp
) -> bool:
    """Return whether an already-received same-minute tick is not applied yet."""
    canonicalize = getattr(manager, "_canonical_symbol", None)
    canonical = canonicalize(symbol) if callable(canonicalize) else symbol
    lock = getattr(manager, "_pending_tick_lock", None)
    if lock is None:
        return False
    with lock:
        if getattr(manager, "_candle_tick_inflight_symbol", None) == canonical:
            return True
        reservations = getattr(manager, _RESERVED_MINUTES_ATTR, {}) or {}
        if int(reservations.get((canonical, expected_minute), 0) or 0) > 0:
            return True
        queues = getattr(manager, "_pending_tick_queues", {}) or {}
        queue = list(queues.get(canonical, ()))
        far = (getattr(manager, "_pending_far_ticks", {}) or {}).get(canonical)
        if isinstance(far, Mapping):
            queue.append(far)
    return any(
        isinstance(tick, Mapping)
        and _raw_tick_minute(tick) == expected_minute
        for tick in queue
    )


def install_candle_clock_flush_hardening(manager_cls: type[Any]) -> None:
    """Install race-safe clock flush and critical-context queue fairness."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_process_tick: Any = getattr(manager_cls, "_process_queued_tick", None)
    has_tick_queue = callable(original_process_tick) and callable(
        getattr(manager_cls, "_pop_pending_tick_batch", None)
    )

    def _process_queued_tick(self: Any, raw: Mapping[str, Any]) -> Any:
        symbol = str(raw.get("symbol") or "")
        if not symbol:
            try:
                token = int(raw.get("instrument_token") or raw.get("token") or 0)
            except (TypeError, ValueError):
                token = 0
            symbol = str((getattr(self, "_symbol_by_token", {}) or {}).get(token) or "")
        canonicalize = getattr(self, "_canonical_symbol", None)
        canonical = canonicalize(symbol) if callable(canonicalize) and symbol else symbol
        lock = getattr(self, "_pending_tick_lock", None)
        if lock is not None:
            with lock:
                self._candle_tick_inflight_symbol = canonical or None
        try:
            return original_process_tick(self, raw)
        finally:
            if lock is not None:
                with lock:
                    if getattr(self, "_candle_tick_inflight_symbol", None) == canonical:
                        self._candle_tick_inflight_symbol = None
                    _release_popped_tick_locked(self, raw)

    def _pop_pending_tick_batch(self: Any) -> list[dict[str, Any]]:
        """Preserve FIFO but prefer critical spot/future context on priority ties."""
        batch: list[dict[str, Any]] = []
        with self._pending_tick_lock:
            if self._pending_count_locked() <= 0:
                self._tick_drain_scheduled = False
                return []
            for _ in range(self._tick_drain_batch_size):
                selected_key: str | None = None
                selected_rank: tuple[int, int, float] | None = None
                for key, queue in self._pending_tick_queues.items():
                    if not queue:
                        continue
                    head = queue[0]
                    priority = int(head.get("_mdm_priority", 99))
                    bucket = str(head.get("_mdm_priority_bucket") or "")
                    bucket_rank = 1 if bucket == "near_atm" else 0
                    enqueued = head.get("_mdm_enqueued_mono")
                    age_rank = (
                        float(enqueued)
                        if isinstance(enqueued, (int, float))
                        else float("inf")
                    )
                    rank = (priority, bucket_rank, age_rank)
                    if selected_rank is None or rank < selected_rank:
                        selected_key = key
                        selected_rank = rank
                if selected_key is not None:
                    queue = self._pending_tick_queues[selected_key]
                    tick = queue.popleft()
                    _reserve_popped_tick_locked(self, tick)
                    batch.append(tick)
                    self._pending_decrement_locked(1)
                    if not queue:
                        self._pending_tick_queues.pop(selected_key, None)
                    continue
                if self._pending_far_ticks:
                    _key, tick = self._pending_far_ticks.popitem()
                    _reserve_popped_tick_locked(self, tick)
                    self._pending_decrement_locked(1)
                    batch.append(tick)
                    continue
                break
            prune = getattr(self, "_pending_heap_prune_locked", None)
            if callable(prune):
                prune()
        return batch

    def flush_due_candles(
        self: Any,
        *,
        now: Any | None = None,
        grace_seconds: float | None = None,
    ) -> int:
        grace = (
            max(float(grace_seconds), 0.0)
            if grace_seconds is not None
            else max(float(getattr(self, "_candle_flush_grace_s", 1.5) or 1.5), 0.0)
        )
        now_ts = (
            _coerce_ist_timestamp(now)
            if now is not None
            else pd.Timestamp.now(tz=_IST_TZ)
        )
        if now_ts is None:
            return 0

        engines = list(getattr(self, "_engines", {}).items())
        flushed = 0
        for symbol, engine in engines:
            current = getattr(engine, "current_candle", None)
            if not isinstance(current, Mapping):
                continue
            expected_minute = _coerce_ist_timestamp(current.get("timestamp"))
            if expected_minute is None:
                continue
            if now_ts < expected_minute + pd.Timedelta(minutes=1, seconds=grace):
                continue
            # An already-received tick remains visible across queue -> local
            # batch -> processing transitions, including a budget requeue cycle.
            if _has_unapplied_tick_for_minute(self, symbol, expected_minute):
                continue

            flush_expected = getattr(engine, "flush_if_current_minute", None)
            try:
                if callable(flush_expected):
                    candle = flush_expected(expected_minute)
                else:
                    if reconcile_stale_current(engine, reason="clock_flush"):
                        continue
                    candle = engine.flush()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "MDM_CANDLE_CLOCK_FLUSH_FAILED symbol=%s error=%r",
                    symbol,
                    exc,
                    exc_info=True,
                    extra={
                        "event": "MDM_CANDLE_CLOCK_FLUSH_FAILED",
                        "symbol": symbol,
                        "error": repr(exc),
                    },
                )
                continue

            if not candle:
                continue
            timestamp = _coerce_ist_timestamp(
                candle["timestamp"] if isinstance(candle, dict) else candle.timestamp
            )
            if timestamp is None:
                continue

            published = _LAST_PUBLISHED[id(self)]
            previous = published.get(symbol)
            if previous is not None and timestamp <= previous:
                self._logger.warning(
                    (
                        "MDM_CANDLE_DUPLICATE_PUBLISH_SUPPRESSED "
                        "symbol=%s timestamp=%s previous=%s"
                    ),
                    symbol,
                    timestamp.isoformat(),
                    previous.isoformat(),
                    extra={
                        "event": "MDM_CANDLE_DUPLICATE_PUBLISH_SUPPRESSED",
                        "symbol": symbol,
                        "timestamp": timestamp.isoformat(),
                        "previous_timestamp": previous.isoformat(),
                    },
                )
                continue

            bar = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": float(
                    candle["open"] if isinstance(candle, dict) else candle.open
                ),
                "high": float(
                    candle["high"] if isinstance(candle, dict) else candle.high
                ),
                "low": float(candle["low"] if isinstance(candle, dict) else candle.low),
                "close": float(
                    candle["close"] if isinstance(candle, dict) else candle.close
                ),
                "volume": int(
                    float(
                        (
                            candle.get("volume", 0)
                            if isinstance(candle, dict)
                            else getattr(candle, "volume", 0)
                        )
                        or 0
                    )
                ),
                "source": "clock_flush_candle",
            }
            refresher = getattr(self, "_refresh_candle_projection", None)
            if callable(refresher):
                refresher(symbol, source="clock_flush_candle")
            with self._lock:
                published[symbol] = timestamp

            publisher = getattr(self, "_publish_closed_bar", None)
            if callable(publisher):
                try:
                    publisher(bar)
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("clock-flush bar publish skipped: %s", exc)

            flushed += 1
            now_mono = time.monotonic()
            if (
                now_mono
                - float(getattr(self, "_last_candle_flush_log_mono", 0.0) or 0.0)
                >= 10.0
            ):
                self._last_candle_flush_log_mono = now_mono
                self._logger.info(
                    "MDM_CANDLE_CLOCK_FLUSHED symbol=%s close=%s",
                    symbol,
                    bar["close"],
                    extra={
                        "event": "MDM_CANDLE_CLOCK_FLUSHED",
                        "symbol": symbol,
                        "source": "clock_flush_candle",
                    },
                )
        return flushed

    if has_tick_queue:
        manager_cls._process_queued_tick = _process_queued_tick
        manager_cls._pop_pending_tick_batch = _pop_pending_tick_batch
    manager_cls.flush_due_candles = flush_due_candles
    setattr(manager_cls, _INSTALLED_ATTR, True)


__all__ = ["install_candle_clock_flush_hardening"]
