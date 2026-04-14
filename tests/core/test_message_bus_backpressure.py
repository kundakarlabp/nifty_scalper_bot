"""Coverage for message bus backpressure behaviour."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType


@pytest.mark.asyncio
async def test_publish_tick_drops_oldest_under_backpressure() -> None:
    bus = MessageBus(max_queue_size=5)
    bus._running = True

    with pytest.raises(RuntimeError, match="Queue full for MessageType.TICK"):
        for index in range(8):
            await bus.publish(
                Message(
                    type=MessageType.TICK,
                    timestamp=datetime.now(timezone.utc),
                    data={"idx": index},
                    source="test",
                )
            )


@pytest.mark.asyncio
async def test_dispatch_does_not_block_on_slow_handler() -> None:
    bus = MessageBus(max_queue_size=10)
    seen: list[int] = []

    async def slow_handler(message: Message) -> None:
        seen.append(int(message.data["idx"]))
        await asyncio.sleep(0.05)

    bus.subscribe(MessageType.TICK, slow_handler)
    bus.start()

    await bus.publish(
        Message(
            type=MessageType.TICK,
            timestamp=datetime.now(timezone.utc),
            data={"idx": 0},
            source="test",
        )
    )
    await bus.publish(
        Message(
            type=MessageType.TICK,
            timestamp=datetime.now(timezone.utc),
            data={"idx": 1},
            source="test",
        )
    )

    await asyncio.sleep(0.01)
    bus._running = False
    for task in list(bus._tasks):
        task.cancel()
    for task in list(bus._tasks):
        with suppress(asyncio.CancelledError):
            await task

    assert seen == [0, 1]
    await asyncio.sleep(0.06)
