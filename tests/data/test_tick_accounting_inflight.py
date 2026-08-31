from __future__ import annotations

import asyncio
from collections import deque
import threading

from nifty_scalper_bot.data.tick_accounting_hardening import (
    install_tick_accounting_hardening,
)


class _FakeMarketDataManager:
    def __init__(self) -> None:
        self._pending_tick_lock = threading.RLock()
        self._pending = deque([{"id": 1}, {"id": 2}, {"id": 3}])
        self._tick_submitted_total = 3
        self._tick_processed_total = 0
        self._tick_coalesced_total = 0
        self._tick_dropped_total = 0
        self._tick_active_drains = 0
        self._tick_drain_scheduled = True

    def _pop_pending_tick_batch(self) -> list[dict[str, int]]:
        with self._pending_tick_lock:
            batch = list(self._pending)
            self._pending.clear()
            return batch

    async def _drain_latest_ticks(self) -> None:
        with self._pending_tick_lock:
            self._tick_active_drains += 1
        try:
            batch = self._pop_pending_tick_batch()
            for _raw in batch:
                await asyncio.sleep(0)
                self._tick_processed_total += 1
        finally:
            with self._pending_tick_lock:
                self._tick_active_drains -= 1
                self._tick_drain_scheduled = False

    def get_tick_pressure_stats(self) -> dict[str, int | bool]:
        with self._pending_tick_lock:
            pending = len(self._pending)
            unexplained = (
                self._tick_submitted_total
                - self._tick_processed_total
                - self._tick_coalesced_total
                - self._tick_dropped_total
                - pending
            )
            return {
                "submitted_total": self._tick_submitted_total,
                "processed_total": self._tick_processed_total,
                "coalesced_total": self._tick_coalesced_total,
                "dropped_total": self._tick_dropped_total,
                "pending_ticks": pending,
                "unexplained_loss": unexplained,
                "active_drains": self._tick_active_drains,
                "drain_scheduled": self._tick_drain_scheduled,
            }


def _patched_manager() -> _FakeMarketDataManager:
    class Manager(_FakeMarketDataManager):
        pass

    install_tick_accounting_hardening(Manager)
    return Manager()


def test_popped_batch_is_reported_as_inflight_not_unexplained() -> None:
    manager = _patched_manager()
    manager._tick_active_drains = 1

    batch = manager._pop_pending_tick_batch()
    stats = manager.get_tick_pressure_stats()

    assert len(batch) == 3
    assert stats["pending_ticks"] == 0
    assert stats["inflight_ticks"] == 3
    assert stats["unexplained_loss"] == 0
    assert stats["accounting_balanced"] is True


def test_partial_processing_reduces_inflight_without_creating_loss() -> None:
    manager = _patched_manager()
    manager._tick_active_drains = 1
    manager._pop_pending_tick_batch()
    manager._tick_processed_total = 2

    stats = manager.get_tick_pressure_stats()

    assert stats["processed_total"] == 2
    assert stats["inflight_ticks"] == 1
    assert stats["unexplained_loss"] == 0
    assert stats["accounting_balanced"] is True


def test_completed_drain_clears_inflight_and_preserves_invariant() -> None:
    manager = _patched_manager()

    asyncio.run(manager._drain_latest_ticks())
    stats = manager.get_tick_pressure_stats()

    assert stats["processed_total"] == 3
    assert stats["pending_ticks"] == 0
    assert stats["inflight_ticks"] == 0
    assert stats["unexplained_loss"] == 0
    assert stats["accounting_balanced"] is True


def test_true_residual_is_not_hidden_when_no_drain_is_active() -> None:
    manager = _patched_manager()
    manager._pending.clear()
    manager._tick_submitted_total = 5
    manager._tick_processed_total = 3
    manager._tick_active_drains = 0
    manager._tick_drain_scheduled = False

    stats = manager.get_tick_pressure_stats()

    assert stats["inflight_ticks"] == 0
    assert stats["unexplained_loss"] == 2
    assert stats["accounting_balanced"] is True


def test_existing_coalesced_and_dropped_terminals_remain_unchanged() -> None:
    manager = _patched_manager()
    manager._pending.clear()
    manager._tick_submitted_total = 7
    manager._tick_processed_total = 3
    manager._tick_coalesced_total = 2
    manager._tick_dropped_total = 2
    manager._tick_active_drains = 0
    manager._tick_drain_scheduled = False

    stats = manager.get_tick_pressure_stats()

    assert stats["coalesced_total"] == 2
    assert stats["dropped_total"] == 2
    assert stats["inflight_ticks"] == 0
    assert stats["unexplained_loss"] == 0
    assert stats["accounting_balanced"] is True
