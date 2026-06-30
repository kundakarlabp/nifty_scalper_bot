from __future__ import annotations

import asyncio
import threading

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _make_mdm():
    mdm = MarketDataManager(kite=None)
    mdm._symbol_by_token[1] = "NFO:NIFTY26JUN24000CE"
    mdm._symbol_to_token["NFO:NIFTY26JUN24000CE"] = 1
    return mdm


async def _stop_mdm(mdm):
    await mdm.stop_tick_drain(timeout=1.0)
    task = getattr(mdm, "_tick_consumer_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        mdm._tick_consumer_task = None


@pytest.mark.asyncio
async def test_threadsafe_tick_ingress_coalesces_and_loop_runs(monkeypatch):
    mdm = _make_mdm()
    processed = []
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(dict(raw))
    )
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)

    def submit():
        for i in range(50_000):
            mdm._enqueue_tick_threadsafe({
                "instrument_token": 1,
                "last_price": i,
                "timestamp": i,
            })

    thread = threading.Thread(target=submit)
    thread.start()
    unrelated_ticks = 0
    while thread.is_alive():
        unrelated_ticks += 1
        await asyncio.sleep(0)
    thread.join()
    await asyncio.sleep(0.05)
    stats = mdm.get_tick_pressure_stats()
    assert stats["drain_callbacks_scheduled"] < 1_000
    assert stats["coalesced_total"] > 0
    assert stats["pending_max_seen"] < 50_000
    assert unrelated_ticks > 0
    assert processed
    assert processed[-1]["last_price"] == 49_999
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_selected_ticks_span_multiple_budget_batches_without_loss(monkeypatch):
    mdm = _make_mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm.set_open_position_symbols([symbol])
    processed = []
    monkeypatch.setattr(mdm, "_tick_drain_batch_size", 3)
    monkeypatch.setattr(mdm, "_tick_drain_budget_s", 0.000001)
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(raw["last_price"])
    )
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    for i in range(10):
        mdm._enqueue_tick_threadsafe({
            "instrument_token": 1,
            "last_price": i,
            "timestamp": i,
        })
    await mdm.drain_pending_ticks(timeout=2.0)
    assert processed == list(range(10))
    stats = mdm.get_tick_pressure_stats()
    assert stats["unexplained_loss"] == 0
    assert stats["max_active_drains"] == 1
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_direct_push_tick_uses_bounded_drain(monkeypatch):
    mdm = _make_mdm()
    processed = []
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(raw["last_price"])
    )
    mdm.set_event_loop(asyncio.get_running_loop())
    await mdm.push_tick({"instrument_token": 1, "last_price": 10, "timestamp": 1})
    await mdm.drain_pending_ticks(timeout=2.0)
    assert processed == [10]
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_malformed_ticks_are_bounded_and_accounted(monkeypatch):
    mdm = _make_mdm()
    mdm.set_event_loop(asyncio.get_running_loop())
    for _ in range(100):
        mdm._enqueue_tick_threadsafe({"last_price": 1})
    await asyncio.sleep(0.01)
    stats = mdm.get_tick_pressure_stats()
    assert stats["pending_ticks"] == 0
    assert stats["dropped_by_reason"]["missing_symbol_token"] == 100
    assert stats["unexplained_loss"] == 0
    await _stop_mdm(mdm)
