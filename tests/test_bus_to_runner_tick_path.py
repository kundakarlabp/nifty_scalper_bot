import asyncio
from datetime import datetime
from nifty_scalper_bot.core.message_bus import MessageBus, Message, MessageType
from nifty_scalper_bot.data.data_hub import DataHub


async def _noop_runner_tick(_: dict):
    return None


def test_bus_to_datahub_tick_path_logs(caplog):
    async def run():
        bus = MessageBus()
        hub = DataHub(market_data=None, bus=bus)
        bus.subscribe_once(MessageType.TICK, hub.ingest_tick_from_bus, subscriber_id='data_hub.ingest_tick_from_bus')
        bus.start()
        await bus.publish(Message(type=MessageType.TICK, timestamp=datetime.utcnow(), data={"symbol":"NSE:NIFTY","last_price":24000.0}, source='test'))
        await asyncio.sleep(0.05)
        await bus.stop()
    asyncio.run(run())
    assert 'MESSAGE_BUS_TICK_DELIVERED_TO_DATAHUB' in caplog.text
    assert 'DATAHUB_TICK_INGESTED_FROM_BUS' in caplog.text
