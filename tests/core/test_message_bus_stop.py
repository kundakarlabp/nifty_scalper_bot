from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType


@pytest.mark.asyncio
async def test_message_bus_stop_is_idempotent_and_cancels_tasks() -> None:
    bus = MessageBus()

    async def handler(_message: Message) -> None:
        await asyncio.sleep(0)

    bus.subscribe(MessageType.TICK, handler)
    assert bus.start() is True
    assert bus._tasks

    await bus.publish(
        Message(
            type=MessageType.TICK,
            timestamp=__import__('datetime').datetime.utcnow(),
            data={},
            source='test',
        )
    )

    await bus.stop()
    await bus.stop()

    assert bus.running is False
    assert bus._tasks == []


def test_message_bus_stop_source_has_no_sync_taskgroup() -> None:
    source = Path('src/nifty_scalper_bot/core/message_bus.py').read_text(encoding='utf-8')
    assert 'with asyncio.TaskGroup()' not in source
