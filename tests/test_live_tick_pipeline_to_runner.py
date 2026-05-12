"""Regression test for live tick path MDM -> DataHub -> Runner logs."""

from __future__ import annotations

import asyncio
from datetime import datetime

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.data.data_hub import DataHub


def test_live_tick_pipeline_to_runner_logs(caplog) -> None:
    """Validate live tick pipeline logging. Args: caplog. Returns: none. Raises: none."""

    async def _run() -> None:
        bus = MessageBus()
        hub = DataHub(market_data=None, bus=bus)
        bus.subscribe_once(
            MessageType.TICK,
            hub.ingest_tick_from_bus,
            subscriber_id='data_hub.ingest_tick_from_bus',
        )
        bus.start()
        await bus.publish(
            Message(
                type=MessageType.TICK,
                timestamp=datetime.utcnow(),
                data={
                    'symbol': 'NFO:NIFTY24APR23900CE',
                    'token': 12345,
                    'last_price': 201.5,
                    'depth': {'buy': [{'price': 201.4}], 'sell': [{'price': 201.6}]},
                },
                source='test',
            )
        )
        await asyncio.sleep(0.05)
        await bus.stop()

    asyncio.run(_run())
    assert 'DATAHUB_TICK_INGESTED_FROM_BUS' in caplog.text
    assert 'DATAHUB_QUOTE_CACHED' in caplog.text
