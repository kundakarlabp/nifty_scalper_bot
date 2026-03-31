"""Reliability coverage for queue priority and exit dedup behaviour."""

from __future__ import annotations

from nifty_scalper_bot.execution.order_queue import OrderQueue, OrderRequest


def test_exit_sl_is_dequeued_before_entry() -> None:
    """Ensure STOP exits bypass entry queue backlog."""
    queue = OrderQueue()
    queue.submit_order_request(
        OrderRequest(symbol='NFO:NIFTY1', side='BUY', quantity=1, intent='ENTRY')
    )
    queue.submit_order_request(
        OrderRequest(symbol='NFO:NIFTY1', side='SELL', quantity=1, intent='EXIT_SL')
    )

    first = queue.get_next_request(timeout=0.1)
    second = queue.get_next_request(timeout=0.1)

    assert first is not None
    assert second is not None
    assert first.intent == 'EXIT_SL'
    assert second.intent == 'ENTRY'


def test_duplicate_emergency_exit_is_allowed() -> None:
    """Ensure emergency EXIT_SL requests remain idempotent and unblocked."""
    queue = OrderQueue()
    request = OrderRequest(
        symbol='NFO:NIFTY2',
        side='SELL',
        quantity=1,
        intent='EXIT_SL',
    )

    queue.submit_order_request(request)
    queue.submit_order_request(
        OrderRequest(
            symbol='NFO:NIFTY2',
            side='SELL',
            quantity=1,
            intent='EXIT_SL',
        )
    )

    first = queue.get_next_request(timeout=0.1)
    second = queue.get_next_request(timeout=0.1)

    assert first is not None
    assert second is not None
    assert first.intent == 'EXIT_SL'
    assert second.intent == 'EXIT_SL'
