"""Reliability tests for LifecycleManager exit and trailing behaviour."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
from nifty_scalper_bot.execution.order_queue import OrderQueue


class _DataHub:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def subscribe_ticks(self, symbol: str, handler: object) -> None:
        self.handlers[symbol] = handler

    def unsubscribe_ticks(self, symbol: str, _handler: object) -> None:
        self.handlers.pop(symbol, None)


def test_partial_exit_reduces_remaining_qty() -> None:
    """Ensure TP1 partial exit updates in-memory remaining quantity."""
    queue = OrderQueue()
    manager = LifecycleManager(data_hub=_DataHub(), order_queue=queue)

    manager.on_fill(
        symbol='NFO:NIFTYTEST',
        entry_price=100.0,
        quantity=10,
        atr=5.0,
        regime='RANGE',
    )
    position = manager._positions['NFO:NIFTYTEST']
    manager.partial_exit('NFO:NIFTYTEST', position)

    assert position.tp1_taken is True
    assert position.quantity == 4


def test_tick_is_normalized_to_tick_size_for_stop_loss() -> None:
    """Ensure stop-loss trigger compares normalized prices to avoid missed exits."""
    queue = OrderQueue()
    manager = LifecycleManager(data_hub=_DataHub(), order_queue=queue)

    manager.on_fill(
        symbol='NFO:NIFTYTEST2',
        entry_price=100.0,
        quantity=1,
        atr=5.0,
        regime='TREND',
    )
    position = manager._positions['NFO:NIFTYTEST2']
    position.tick_size = 0.05
    position.sl = 99.95

    manager._handle_tick('NFO:NIFTYTEST2', {'ltp': 99.949})

    request = queue.get_next_request(timeout=0.1)
    assert request is not None
    assert request.intent == 'EXIT_SL'


def test_trailing_updates_authoritative_stop() -> None:
    """Ensure computed trail stop also advances authoritative stop-loss."""
    queue = OrderQueue()
    manager = LifecycleManager(data_hub=_DataHub(), order_queue=queue)

    manager.on_fill(
        symbol='NFO:NIFTYTEST3',
        entry_price=100.0,
        quantity=1,
        atr=5.0,
        regime='TREND',
    )
    position = manager._positions['NFO:NIFTYTEST3']
    position.highest_price = 120.0

    manager.update_all_positions(datetime.now(timezone.utc))

    assert position.trail_stop is not None
    assert position.sl >= position.trail_stop
