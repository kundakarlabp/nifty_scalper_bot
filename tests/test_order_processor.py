from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.core.message_bus import Message, MessageType
from nifty_scalper_bot.execution.order_processor import OrderProcessor


class DummyBus:
    def __init__(self) -> None:
        self.published: list[Message] = []

    async def publish(self, message: Message) -> None:
        self.published.append(message)

    async def subscribe(self, _message_type: MessageType, _handler) -> None:
        return None


class DummyOrderManager:
    def __init__(self) -> None:
        self.bracket_calls: list[dict] = []
        self.order_calls: list[dict] = []

    def start_monitoring(self) -> None:
        return None

    def stop_monitoring(self) -> None:
        return None

    def place_bracket_order(self, **kwargs):
        self.bracket_calls.append(kwargs)
        return ('entry-1', 'stop-1', 'target-1')

    def place_order(self, **kwargs):
        self.order_calls.append(kwargs)
        return 'order-1'


class DummyRiskManager:
    def __init__(self) -> None:
        self.last_signal = None

    def check_order(self, signal, live_enabled: bool):
        self.last_signal = signal
        return True, ""


class DummyPositionManager:
    def get_position(self, _symbol: str):
        return None

    def get_all_positions(self):
        return []


class DummyDataHub:
    def __init__(
        self, price: float, indicators: dict[str, float | str] | None = None
    ) -> None:
        self.price = price
        self.indicators = indicators or {}

    def get_quote(self, _symbol: str, allow_pull: bool = False):
        return {'ltp': self.price}

    def get_indicator(self, _symbol: str, indicator: str):
        return self.indicators.get(indicator)


@pytest.mark.asyncio
async def test_order_processor_bracket_update_tracks_ids_and_trailing():
    bus = DummyBus()
    order_manager = DummyOrderManager()
    risk_manager = DummyRiskManager()
    pos_manager = DummyPositionManager()
    data_hub = DummyDataHub(price=100.0)

    processor = OrderProcessor(
        message_bus=bus,
        safe_order_manager=order_manager,
        risk_manager=risk_manager,
        position_manager=pos_manager,
        data_hub=data_hub,
    )
    processor._running = True

    payload = {
        'symbol': 'NFO:TEST',
        'side': 'BUY',
        'quantity': 1,
        'price': 0.0,
        'stop_loss': 95.0,
        'take_profit': 110.0,
        'strategy_name': 'test_strategy',
        'metadata': {'trailing_spec': {'kind': 'atr', 'mult': 1.0}},
    }

    message = Message(
        type=MessageType.SIGNAL,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        source='test',
    )

    await processor.on_strategy_signal(message)

    assert risk_manager.last_signal is not None
    assert risk_manager.last_signal.price == 100.0
    assert order_manager.bracket_calls
    assert bus.published

    update = bus.published[0].data
    assert update['order_id'] == 'entry-1'
    assert update['stop_order_id'] == 'stop-1'
    assert update['target_order_id'] == 'target-1'
    assert update['trailing_spec'] == {'kind': 'atr', 'mult': 1.0}


@pytest.mark.asyncio
async def test_order_processor_builds_virtual_bracket_defaults():
    bus = DummyBus()
    order_manager = DummyOrderManager()
    risk_manager = DummyRiskManager()
    pos_manager = DummyPositionManager()
    data_hub = DummyDataHub(price=200.0, indicators={'atr': 20.0, 'regime': 'trend'})

    processor = OrderProcessor(
        message_bus=bus,
        safe_order_manager=order_manager,
        risk_manager=risk_manager,
        position_manager=pos_manager,
        data_hub=data_hub,
    )
    processor._running = True

    payload = {
        'symbol': 'NFO:DEFAULT',
        'side': 'BUY',
        'quantity': 1,
        'price': 0.0,
        'stop_loss': 0.0,
        'take_profit': 0.0,
        'strategy_name': 'test_strategy',
        'metadata': {},
    }

    message = Message(
        type=MessageType.SIGNAL,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        source='test',
    )

    await processor.on_strategy_signal(message)

    assert risk_manager.last_signal is not None
    assert risk_manager.last_signal.stop_loss is not None
    assert risk_manager.last_signal.take_profit is not None
    assert order_manager.bracket_calls
    bracket_call = order_manager.bracket_calls[0]
    assert bracket_call['stop_loss'] < 200.0
    assert bracket_call['take_profit'] > 200.0
    assert bracket_call['trailing_atr_mult'] is not None
