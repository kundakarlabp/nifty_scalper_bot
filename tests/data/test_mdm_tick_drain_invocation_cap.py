"""Live tick drain fairness: bounded invocations without dropping protected ticks."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


@pytest.mark.asyncio
async def test_drain_invocation_cap_preserves_order_and_all_protected_ticks(
    monkeypatch,
) -> None:
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._symbol_by_token[1] = symbol
    mdm._symbol_to_token[symbol] = 1
    mdm.set_open_position_symbols([symbol])
    mdm._tick_drain_batch_size = 20
    mdm._tick_drain_budget_s = 30.0
    mdm._tick_drain_invocation_budget_s = 30.0
    mdm._tick_drain_invocation_max_ticks = 2
    received: list[float] = []
    monkeypatch.setattr(
        mdm,
        "_process_queued_tick",
        lambda raw: received.append(float(raw["last_price"])),
    )
    mdm.set_event_loop(asyncio.get_running_loop())
    try:
        for i in range(8):
            mdm._enqueue_tick_threadsafe(
                {
                    "instrument_token": 1,
                    "last_price": float(i + 1),
                    "timestamp": float(i + 1),
                }
            )
        await mdm.drain_pending_ticks(timeout=3.0)
        assert received == [float(i + 1) for i in range(8)]
        stats = mdm.get_tick_pressure_stats()
        assert stats["max_active_drains"] == 1
        assert stats["unexplained_loss"] == 0
        assert stats["drain_callbacks_completed"] >= 4
    finally:
        await mdm.stop_tick_drain(timeout=1.0)


@pytest.mark.asyncio
async def test_tick_drain_avoids_busy_default_executor_and_keeps_trace(monkeypatch):
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._symbol_by_token[1] = symbol
    mdm._symbol_to_token[symbol] = 1
    mdm.set_open_position_symbols([symbol])
    trace = ContextVar("mdm_tick_trace", default=None)
    trace.set("expected")
    received = []
    monkeypatch.setattr(
        mdm,
        "_process_queued_tick",
        lambda raw: received.append((raw["instrument_token"], trace.get())),
    )

    blocker = threading.Event()
    worker_started = threading.Event()
    default_executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(default_executor)

    def hold_unrelated_worker() -> None:
        worker_started.set()
        blocker.wait(timeout=5.0)

    occupied = loop.run_in_executor(None, hold_unrelated_worker)
    try:
        assert await asyncio.to_thread(worker_started.wait, 1.0) is True
        mdm.set_event_loop(loop)
        mdm._enqueue_tick_threadsafe(
            {"instrument_token": 1, "last_price": 101.0, "timestamp": 100.0}
        )
        await asyncio.wait_for(mdm.drain_pending_ticks(timeout=1.0), timeout=2.0)
        assert received == [(1, "expected")]
    finally:
        blocker.set()
        await occupied
        await mdm.async_stop()
        default_executor.shutdown(wait=True)
