"""Bound required-symbol tick backlog without weakening live safety gates.

The MarketDataManager native queue keeps every priority 0-2 tick. Under a
sustained Kite burst that can exceed ``_tick_queue_maxsize`` because the
existing overflow loop only evicts ``_pending_far_ticks``. This installer
keeps open-position ticks lossless, compacts selected/context ticks by symbol
and minute, and applies a real total-pending bound to non-position work.
"""

from __future__ import annotations

from datetime import timezone
import time
from typing import Any, Mapping

import pandas as pd

_INSTALLED_ATTR = "_required_tick_backlog_hardening_installed"
_ORIGINAL_ENQUEUE_ATTR = "_required_tick_backlog_original_enqueue"
_ORIGINAL_PROCESS_ATTR = "_required_tick_backlog_original_process"


def install_required_tick_backlog_hardening(manager_cls: type[Any]) -> None:
    """Install idempotent bounded queue and compacted-tick replay wrappers."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_enqueue = getattr(manager_cls, "_enqueue_latest_tick_for_drain", None)
    original_process = getattr(manager_cls, "_process_queued_tick", None)
    if not callable(original_enqueue) or not callable(original_process):
        raise RuntimeError("MarketDataManager tick queue API unavailable")

    setattr(manager_cls, _ORIGINAL_ENQUEUE_ATTR, original_enqueue)
    setattr(manager_cls, _ORIGINAL_PROCESS_ATTR, original_process)
    setattr(manager_cls, "_enqueue_latest_tick_for_drain", _enqueue_bounded_required_tick)
    setattr(manager_cls, "_process_queued_tick", _process_compacted_tick)
    setattr(manager_cls, _INSTALLED_ATTR, True)


def _tick_timestamp(payload: Mapping[str, Any]) -> pd.Timestamp | None:
    for key in ("exchange_timestamp", "timestamp", "last_trade_time", "received_at"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            ts = pd.Timestamp(value)
        except Exception:
            continue
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        return ts.tz_convert("Asia/Kolkata")
    return None


def _minute_key(payload: Mapping[str, Any]) -> int | None:
    ts = _tick_timestamp(payload)
    if ts is None:
        return None
    return int(ts.floor("1min").timestamp())


def _price(payload: Mapping[str, Any]) -> float | None:
    for key in ("last_price", "ltp", "price"):
        value = payload.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _latest_cumulative_volume(payload: Mapping[str, Any]) -> Any:
    for key in (
        "volume_traded_today",
        "volume_cumulative",
        "cumulative_volume",
        "volume",
    ):
        if payload.get(key) is not None:
            return payload.get(key)
    return None


def _merge_same_minute(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge same-symbol/same-minute ticks into one replayable payload."""
    merged = dict(incoming)
    existing_prices = list(existing.get("_mdm_compacted_prices") or [])
    if not existing_prices:
        existing_price = _price(existing)
        if existing_price is not None:
            existing_prices.append(existing_price)
    incoming_price = _price(incoming)
    if incoming_price is not None:
        existing_prices.append(incoming_price)

    if existing_prices:
        first = float(existing_prices[0])
        high = max(float(value) for value in existing_prices)
        low = min(float(value) for value in existing_prices)
        latest = float(existing_prices[-1])
        replay: list[float] = []
        for value in (first, high, low, latest):
            if not replay or replay[-1] != value:
                replay.append(value)
        merged["_mdm_compacted_prices"] = replay
        merged["_mdm_compacted"] = True
        merged["_mdm_first_price"] = first
        merged["_mdm_high_price"] = high
        merged["_mdm_low_price"] = low
        merged["_mdm_latest_price"] = latest

    merged["_mdm_enqueued_mono"] = min(
        float(existing.get("_mdm_enqueued_mono", time.monotonic())),
        float(incoming.get("_mdm_enqueued_mono", time.monotonic())),
    )
    merged["_mdm_subscription_generation"] = incoming.get(
        "_mdm_subscription_generation",
        existing.get("_mdm_subscription_generation"),
    )
    merged["_mdm_compacted_count"] = int(existing.get("_mdm_compacted_count", 1)) + 1
    cumulative = _latest_cumulative_volume(incoming)
    if cumulative is not None:
        merged["_mdm_latest_cumulative_volume"] = cumulative
    return merged


def _evict_one_non_position_locked(self: Any) -> bool:
    """Evict one oldest non-position pending item under the pending lock."""
    candidate_key: str | None = None
    candidate_priority = -1
    candidate_ts = float("inf")
    for key, queue in list(self._pending_tick_queues.items()):
        if not queue:
            self._pending_tick_queues.pop(key, None)
            continue
        head = queue[0]
        priority = int(head.get("_mdm_priority", 99))
        if priority <= 0:
            continue
        enqueued = float(head.get("_mdm_enqueued_mono", 0.0) or 0.0)
        if priority > candidate_priority or (
            priority == candidate_priority and enqueued < candidate_ts
        ):
            candidate_key = key
            candidate_priority = priority
            candidate_ts = enqueued

    if candidate_key is not None:
        queue = self._pending_tick_queues[candidate_key]
        dropped = queue.popleft()
        self._pending_decrement_locked(1)
        if not queue:
            self._pending_tick_queues.pop(candidate_key, None)
        bucket = str(dropped.get("_mdm_priority_bucket") or "context_or_far")
        self._record_tick_drop(bucket, "pending_limit_required")
        return True

    if self._pending_far_ticks:
        _key, dropped = self._pending_far_ticks.popitem()
        self._pending_decrement_locked(1)
        bucket = str(dropped.get("_mdm_priority_bucket") or "context_or_far")
        self._record_tick_drop(bucket, "pending_limit_far")
        return True
    return False


def _enqueue_bounded_required_tick(self: Any, tick: dict[str, Any], loop: Any) -> None:
    """Bound priority-1/2 pending work while preserving priority-0 FIFO."""
    key, priority, bucket, reason = self._resolve_tick_key_and_priority(tick)
    with self._pending_tick_lock:
        self._tick_submitted_total += 1
        if key is None:
            self._record_tick_drop(bucket, reason)
            return

        tick["_mdm_priority"] = priority
        tick["_mdm_priority_bucket"] = bucket
        tick["_mdm_subscription_generation"] = int(
            self._symbol_subscription_generation.get(key, self._subscription_generation)
        )
        tick.setdefault("_mdm_enqueued_mono", time.monotonic())

        compacted = False
        queue_for_key = self._pending_tick_queues[key]
        per_symbol_soft_limit = 8 if priority == 1 else 2
        if priority in (1, 2) and queue_for_key:
            latest = queue_for_key[-1]
            same_minute = _minute_key(latest) == _minute_key(tick)
            if same_minute and len(queue_for_key) >= per_symbol_soft_limit:
                queue_for_key[-1] = _merge_same_minute(latest, tick)
                self._tick_coalesced_total += 1
                self._tick_coalesced_by_priority[bucket] += 1
                self._tick_queue_priority_coalesced[bucket] += 1
                compacted = True
                now = time.monotonic()
                last_log = float(
                    getattr(self, "_last_required_tick_compact_log", 0.0) or 0.0
                )
                if now - last_log >= 30.0:
                    self._last_required_tick_compact_log = now
                    self._logger.warning(
                        "MDM_REQUIRED_TICK_COMPACTED symbol=%s priority=%s pending=%s",
                        key,
                        priority,
                        self._pending_count_locked(),
                        extra={
                            "event": "MDM_REQUIRED_TICK_COMPACTED",
                            "symbol": key,
                            "priority": priority,
                            "priority_bucket": bucket,
                            "same_minute": True,
                            "pending_after": self._pending_count_locked(),
                            "coalesced_total": self._tick_coalesced_total,
                        },
                    )

        if not compacted:
            if priority <= 2:
                queue_for_key.append(tick)
                self._pending_increment_locked(tick, key)
                self._bus_priority_counts[bucket] += 1
                if priority == 0:
                    self._log_open_position_priority_if_needed(key)
            else:
                if key in self._pending_far_ticks:
                    self._tick_coalesced_total += 1
                    self._pending_decrement_locked(1)
                    self._tick_coalesced_by_priority[bucket] += 1
                    self._tick_queue_priority_coalesced[bucket] += 1
                self._pending_far_ticks[key] = tick
                self._pending_increment_locked(tick, key)
                self._bus_priority_counts[bucket] += 1

        pending = self._pending_count_locked()
        self._tick_pending_max_seen = max(self._tick_pending_max_seen, pending)
        hard_limit = max(1, int(self._tick_queue_maxsize))
        while pending > hard_limit:
            if not _evict_one_non_position_locked(self):
                # Only priority-0 work remains. Preserve protective-exit ticks and
                # surface the exceptional overage instead of silently discarding.
                self._logger.critical(
                    "MDM_OPEN_POSITION_PENDING_LIMIT_EXCEEDED pending=%s limit=%s",
                    pending,
                    hard_limit,
                    extra={
                        "event": "MDM_OPEN_POSITION_PENDING_LIMIT_EXCEEDED",
                        "pending_ticks": pending,
                        "limit": hard_limit,
                    },
                )
                break
            pending = self._pending_count_locked()

        self._update_pipeline_overload_locked()
        self._schedule_tick_drain_locked(loop)


def _process_compacted_tick(self: Any, raw: dict[str, Any]) -> None:
    """Replay compacted OHLC extrema through the original canonical tick path."""
    original = getattr(type(self), _ORIGINAL_PROCESS_ATTR)
    prices = list(raw.get("_mdm_compacted_prices") or [])
    if not prices:
        original(self, raw)
        return

    latest_index = len(prices) - 1
    for index, value in enumerate(prices):
        replay = dict(raw)
        replay.pop("_mdm_compacted_prices", None)
        replay["last_price"] = float(value)
        replay["ltp"] = float(value)
        replay["price"] = float(value)
        if index != latest_index:
            for key in (
                "depth",
                "volume_traded_today",
                "volume_cumulative",
                "cumulative_volume",
                "volume",
            ):
                replay.pop(key, None)
        else:
            cumulative = raw.get("_mdm_latest_cumulative_volume")
            if cumulative is not None:
                replay["volume_traded_today"] = cumulative
        original(self, replay)
