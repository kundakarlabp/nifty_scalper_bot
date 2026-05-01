from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from nifty_scalper_bot.core.app import BotContext, _wire_and_start_message_bus
from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType


def test_bot_context_declares_runtime_state() -> None:
    names = {f.name for f in fields(BotContext)}
    assert 'message_bus_tick_subscribed' in names
    assert 'message_bus_running' in names
    assert 'deferred_basket_retry_started' in names
    assert 'deferred_basket_retry_task' in names
    assert 'last_deferred_basket_retry_ts' in names


def test_wire_message_bus_idempotent() -> None:
    bus = MessageBus()

    class Hub:
        async def ingest_tick_from_bus(self, message: Message) -> None:
            return None

    class Runner:
        async def on_data(self, message: Message) -> None:
            return None

    ctx = SimpleNamespace(
        message_bus=bus,
        data_hub=Hub(),
        strategy_runner=Runner(),
        market_data_manager=None,
        message_bus_tick_subscribed=False,
        message_bus_running=False,
    )
    _wire_and_start_message_bus(ctx)
    _wire_and_start_message_bus(ctx)

    assert len(bus.subscribers[MessageType.TICK]) == 1
    assert bus.subscribers[MessageType.TICK][0].__name__ == 'ingest_tick_from_bus'


def test_wire_message_bus_runner_fallback_without_datahub() -> None:
    bus = MessageBus()

    class Runner:
        async def on_data(self, message: Message) -> None:
            return None

    ctx = SimpleNamespace(
        message_bus=bus,
        data_hub=None,
        strategy_runner=Runner(),
        market_data_manager=None,
        message_bus_tick_subscribed=False,
        message_bus_running=False,
    )
    _wire_and_start_message_bus(ctx)
    _wire_and_start_message_bus(ctx)

    assert len(bus.subscribers[MessageType.TICK]) == 1
    assert bus.subscribers[MessageType.TICK][0].__name__ == 'on_data'


def test_message_bus_subscribe_once_dedupes() -> None:
    bus = MessageBus()

    async def handler(message: Message) -> None:
        return None

    assert bus.subscribe_once(MessageType.TICK, handler, subscriber_id='h') is True
    assert bus.subscribe_once(MessageType.TICK, handler, subscriber_id='h') is False
    assert len(bus.subscribers[MessageType.TICK]) == 1


def test_message_bus_start_without_subscribers_does_not_raise() -> None:
    bus = MessageBus()
    assert bus.start() is False


def test_runner_has_single_on_data() -> None:
    text = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert text.count('async def on_data') == 1

def test_market_data_manager_skips_bus_publish_when_bus_not_running() -> None:
    text = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text(encoding='utf-8')
    assert 'MDM_BUS_PUBLISH_SKIPPED reason=bus_not_running' in text
    assert 'getattr(bus, "running", False)' in text
