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
