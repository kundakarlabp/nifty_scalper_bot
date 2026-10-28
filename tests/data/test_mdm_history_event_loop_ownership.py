"""History-loop ownership regressions; no broker calls or live state."""

from __future__ import annotations

import asyncio

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _bare_manager() -> MarketDataManager:
    manager = object.__new__(MarketDataManager)
    manager._canonical_history_lock = asyncio.Lock()
    manager._canonical_history_inflight_lock = asyncio.Lock()
    manager._canonical_history_inflight = {}
    return manager


def test_completed_simulation_loop_can_be_replaced_without_reusing_locks() -> None:
    manager = _bare_manager()
    first = asyncio.new_event_loop()
    try:
        first.run_until_complete(_claim(manager))
        original_lock = manager._canonical_history_lock
        original_inflight_lock = manager._canonical_history_inflight_lock
        assert manager._canonical_history_event_loop is first
    finally:
        first.close()

    second = asyncio.new_event_loop()
    try:
        second.run_until_complete(_claim(manager))
        assert manager._canonical_history_event_loop is second
        assert manager._canonical_history_lock is not original_lock
        assert manager._canonical_history_inflight_lock is not original_inflight_lock
        assert manager._canonical_history_inflight == {}
    finally:
        second.close()


async def _claim(manager: MarketDataManager) -> None:
    manager._claim_canonical_history_loop()


@pytest.mark.asyncio
async def test_concurrent_second_loop_rejected_without_discarding_state() -> None:
    manager = _bare_manager()
    manager._claim_canonical_history_loop()
    first_loop = asyncio.get_running_loop()
    original_lock = manager._canonical_history_lock

    def _second_loop() -> str:
        async def attempt() -> str:
            with pytest.raises(
                RuntimeError, match="canonical_history_cross_event_loop_active"
            ):
                manager._claim_canonical_history_loop()
            return "blocked"

        return asyncio.run(attempt())

    assert await asyncio.to_thread(_second_loop) == "blocked"
    assert manager._canonical_history_event_loop is first_loop
    assert manager._canonical_history_lock is original_lock
