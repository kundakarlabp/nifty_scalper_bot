"""Exact tick-pipeline accounting for batches already removed from pending.

This module changes telemetry only. It does not alter tick ordering, queueing,
coalescing, overload thresholds, strategy evaluation, or order execution.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_INSTALLED_ATTR = "_tick_accounting_hardening_installed"
_ORIGINAL_POP_ATTR = "_tick_accounting_original_pop_pending_tick_batch"
_ORIGINAL_DRAIN_ATTR = "_tick_accounting_original_drain_latest_ticks"
_ORIGINAL_STATS_ATTR = "_tick_accounting_original_get_tick_pressure_stats"
_INFLIGHT_ATTR = "_tick_accounting_inflight_batch_size"


def _set_inflight(self: Any, count: int) -> None:
    value = max(int(count), 0)
    lock = getattr(self, "_pending_tick_lock", None)
    if lock is None:
        setattr(self, _INFLIGHT_ATTR, value)
        return
    with lock:
        setattr(self, _INFLIGHT_ATTR, value)


def install_tick_accounting_hardening(manager_cls: type[Any]) -> None:
    """Classify popped-but-not-terminal ticks as in-flight, not lost."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_pop = getattr(manager_cls, "_pop_pending_tick_batch", None)
    original_drain = getattr(manager_cls, "_drain_latest_ticks", None)
    original_stats = getattr(manager_cls, "get_tick_pressure_stats", None)
    if (
        not callable(original_pop)
        or not callable(original_drain)
        or not callable(original_stats)
    ):
        raise RuntimeError("tick_accounting_required_methods_missing")

    setattr(manager_cls, _ORIGINAL_POP_ATTR, original_pop)
    setattr(manager_cls, _ORIGINAL_DRAIN_ATTR, original_drain)
    setattr(manager_cls, _ORIGINAL_STATS_ATTR, original_stats)

    @wraps(original_pop)
    def _pop_pending_tick_batch_accounted(self: Any) -> list[dict[str, Any]]:
        batch = original_pop(self)
        _set_inflight(self, len(batch))
        return batch

    @wraps(original_drain)
    async def _drain_latest_ticks_accounted(self: Any) -> None:
        try:
            await original_drain(self)
        finally:
            # Original drain owns cancellation/requeue semantics. Once it has
            # returned or unwound, no popped batch remains owned by that drain.
            _set_inflight(self, 0)

    @wraps(original_stats)
    def _get_tick_pressure_stats_accounted(self: Any) -> dict[str, Any]:
        stats = dict(original_stats(self))
        submitted = max(int(stats.get("submitted_total", 0) or 0), 0)
        processed = max(int(stats.get("processed_total", 0) or 0), 0)
        coalesced = max(int(stats.get("coalesced_total", 0) or 0), 0)
        dropped = max(int(stats.get("dropped_total", 0) or 0), 0)
        pending = max(int(stats.get("pending_ticks", 0) or 0), 0)
        active_drains = max(int(stats.get("active_drains", 0) or 0), 0)

        residual = max(submitted - processed - coalesced - dropped - pending, 0)
        tracked_batch = max(int(getattr(self, _INFLIGHT_ATTR, 0) or 0), 0)
        inflight = min(residual, tracked_batch) if active_drains > 0 else 0
        unexplained = max(residual - inflight, 0)

        stats["inflight_ticks"] = inflight
        stats["unexplained_loss"] = unexplained
        stats["accounting_total"] = (
            processed + coalesced + dropped + pending + inflight + unexplained
        )
        stats["accounting_balanced"] = stats["accounting_total"] == submitted
        return stats

    setattr(manager_cls, "_pop_pending_tick_batch", _pop_pending_tick_batch_accounted)
    setattr(manager_cls, "_drain_latest_ticks", _drain_latest_ticks_accounted)
    setattr(manager_cls, "get_tick_pressure_stats", _get_tick_pressure_stats_accounted)
    setattr(manager_cls, _INSTALLED_ATTR, True)


__all__ = ["install_tick_accounting_hardening"]
