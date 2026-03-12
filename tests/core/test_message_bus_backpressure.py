"""Coverage for message bus backpressure behaviour."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType


@pytest.mark.asyncio
async def test_publish_tick_drops_oldest_under_backpressure() -> None:
    bus = MessageBus(max_queue_size=5)
    bus._running = True

    for index in range(8):
        await bus.publish(
            Message(
                type=MessageType.TICK,
                timestamp=datetime.now(timezone.utc),
                data={'idx': index},
                source='test',
            )
        )

    diagnostics = bus.queue_diagnostics()
    assert diagnostics['tick']['depth'] <= 5
    assert diagnostics['tick']['dropped'] >= 1
