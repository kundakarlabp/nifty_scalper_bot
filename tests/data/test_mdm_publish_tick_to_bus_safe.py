import asyncio
import concurrent.futures
import logging

import pandas as pd

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _make_mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = logging.getLogger('test_mdm')
    mdm.bus = None
    mdm._tick_bus = None
    mdm._main_loop = None
    return mdm


def test_publish_tick_no_bus_no_exception() -> None:
    mdm = _make_mdm()
    mdm._publish_tick_to_bus_safe({'symbol': 'NSE:NIFTY', 'ltp': 1.0})


def test_publish_tick_bus_not_running_no_exception() -> None:
    mdm = _make_mdm()

    class Bus:
        running = False

    mdm.bus = Bus()
    mdm._publish_tick_to_bus_safe({'symbol': 'NSE:NIFTY', 'ltp': 1.0})


def test_publish_tick_missing_loop_no_exception() -> None:
    mdm = _make_mdm()

    class Bus:
        running = True

    mdm.bus = Bus()
    mdm._publish_tick_to_bus_safe({'symbol': 'NSE:NIFTY', 'ltp': 1.0})


def test_publish_tick_calls_bus_publish_once() -> None:
    mdm = _make_mdm()
    called = []

    class Bus:
        running = True

        async def publish(self, msg):
            called.append(msg)

    class Loop:
        def is_closed(self):
            return False

    fut = concurrent.futures.Future()
    fut.set_result(None)
    orig = asyncio.run_coroutine_threadsafe
    asyncio.run_coroutine_threadsafe = lambda coro, loop: (asyncio.get_event_loop().run_until_complete(coro), fut)[1]
    try:
        mdm.bus = Bus()
        mdm._main_loop = Loop()
        mdm._publish_tick_to_bus_safe({'symbol': 'NSE:NIFTY', 'ltp': 1.0})
    finally:
        asyncio.run_coroutine_threadsafe = orig

    assert len(called) == 1


def test_publish_tick_payload_json_safe() -> None:
    mdm = _make_mdm()
    captured = {}

    class Bus:
        running = True

        async def publish(self, msg):
            captured['msg'] = msg

    class Loop:
        def is_closed(self):
            return False

    fut = concurrent.futures.Future()
    fut.set_result(None)
    orig = asyncio.run_coroutine_threadsafe
    asyncio.run_coroutine_threadsafe = lambda coro, loop: (asyncio.get_event_loop().run_until_complete(coro), fut)[1]
    try:
        mdm.bus = Bus()
        mdm._main_loop = Loop()
        mdm._publish_tick_to_bus_safe({'symbol': 'NSE:NIFTY', 'ltp': 1.0, 'timestamp': pd.Timestamp.utcnow()})
    finally:
        asyncio.run_coroutine_threadsafe = orig

    assert isinstance(captured['msg'].data['timestamp'], str)
