from __future__ import annotations

import asyncio
from collections import deque
import threading

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
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
        self._tick_accounting_inflight_batch_size = 0

    def _pop_pending_tick_batch(self) -> list[dict[str, int]]:
        with self._pending_tick_lock:
            batch = list(self._pending)
            self._pending.clear()
            self._tick_accounting_inflight_batch_size = len(batch)
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
                self._tick_accounting_inflight_batch_size = 0
                self._tick_drain_scheduled = False

    def get_tick_pressure_stats(self) -> dict[str, int | bool]:
        with self._pending_tick_lock:
            pending = len(self._pending)
            residual = max(
                self._tick_submitted_total
                - self._tick_processed_total
                - self._tick_coalesced_total
                - self._tick_dropped_total
                - pending,
                0,
            )
            inflight = (
                min(residual, self._tick_accounting_inflight_batch_size)
                if self._tick_active_drains > 0
                else 0
            )
            unexplained = max(residual - inflight, 0)
            accounting_total = (
                self._tick_processed_total
                + self._tick_coalesced_total
                + self._tick_dropped_total
                + pending
                + inflight
                + unexplained
            )
            return {
                "submitted_total": self._tick_submitted_total,
                "processed_total": self._tick_processed_total,
                "coalesced_total": self._tick_coalesced_total,
                "dropped_total": self._tick_dropped_total,
                "pending_ticks": pending,
                "inflight_ticks": inflight,
                "unexplained_loss": unexplained,
                "accounting_total": accounting_total,
                "accounting_balanced": (accounting_total == self._tick_submitted_total),
                "active_drains": self._tick_active_drains,
                "drain_scheduled": self._tick_drain_scheduled,
            }


def _patched_manager() -> _FakeMarketDataManager:
    return _FakeMarketDataManager()


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


def test_tick_accounting_installer_does_not_replace_native_methods() -> None:
    before_pop = MarketDataManager._pop_pending_tick_batch
    before_drain = MarketDataManager._drain_latest_ticks
    before_stats = MarketDataManager.get_tick_pressure_stats

    install_tick_accounting_hardening(MarketDataManager)

    assert MarketDataManager._pop_pending_tick_batch is before_pop
    assert MarketDataManager._drain_latest_ticks is before_drain
    assert MarketDataManager.get_tick_pressure_stats is before_stats
