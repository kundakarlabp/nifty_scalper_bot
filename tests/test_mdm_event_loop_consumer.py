from __future__ import annotations

import asyncio
import threading
import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_set_event_loop_starts_consumer_after_start() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm.start()

    loop = asyncio.new_event_loop()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    try:
        time.sleep(0.05)
        mdm.set_event_loop(loop)
        time.sleep(0.1)
        task = getattr(mdm, '_tick_consumer_task', None)
        assert task is not None
        assert not task.done()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
