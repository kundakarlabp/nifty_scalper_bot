"""Live tick drain fairness: bounded invocations without dropping protected ticks."""

from __future__ import annotations

import asyncio

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
