from __future__ import annotations

import asyncio
import threading
import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_set_event_loop_starts_consumer_after_started_flag() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm._started = True

    loop = asyncio.new_event_loop()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    try:
        time.sleep(0.05)
        mdm.set_event_loop(loop)
        time.sleep(0.2)
        task = getattr(mdm, '_tick_consumer_task', None)
        assert task is not None
        assert not task.done()
    finally:
        task = getattr(mdm, '_tick_consumer_task', None)
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()
