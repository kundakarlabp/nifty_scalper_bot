from __future__ import annotations

import asyncio
from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    pass


def test_sync_callback_none_no_error() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    seen: list[dict[str, Any]] = []

    def callback(tick: dict[str, Any]) -> None:
        seen.append(tick)
        return None

    mdm.subscribe('NSE:NIFTY', callback)
    mdm._emit_tick('NSE:NIFTY', {'symbol': 'NSE:NIFTY', 'ltp': 10.0}, source='ws')  # noqa: SLF001
    assert len(seen) == 1


def test_sync_callback_custom_awaitable_no_type_error() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)

    class CustomAwaitable:
        def __await__(self):
            if False:
                yield None
            return None

    def callback(_tick: dict[str, Any]) -> CustomAwaitable:
        return CustomAwaitable()

    mdm.subscribe('NSE:NIFTY', callback)
    mdm._emit_tick('NSE:NIFTY', {'symbol': 'NSE:NIFTY', 'ltp': 10.0}, source='ws')  # noqa: SLF001


def test_async_callback_scheduled_on_main_loop() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    ran = {'ok': False}

    async def callback(_tick: dict[str, Any]) -> None:
        ran['ok'] = True

    async def runner() -> None:
        mdm.set_event_loop(asyncio.get_running_loop())
        mdm.subscribe('NSE:NIFTY', callback)
        mdm._emit_tick('NSE:NIFTY', {'symbol': 'NSE:NIFTY', 'ltp': 11.0}, source='ws')  # noqa: SLF001
        await asyncio.sleep(0.01)

    asyncio.run(runner())
    assert ran['ok'] is True


def test_datahub_ingest_tick_sync_returns_none() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    hub = DataHub(mdm)
    result = hub.ingest_tick_sync({'symbol': 'NSE:NIFTY', 'ltp': 1.0})
    assert result is None
