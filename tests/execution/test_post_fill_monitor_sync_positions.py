"""Compatibility tests for PostFillMonitor broker position fetch modes."""

from __future__ import annotations

import asyncio

from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor


class _SyncBroker:
    def get_positions(self) -> list[dict[str, object]]:
        return [{'symbol': 'NFO:NIFTY', 'quantity': 1, 'avg_price': 100.0}]


class _Tracker:
    def __init__(self) -> None:
        self.updated: list[tuple[str, dict[str, object]]] = []

    def get_open_positions(self) -> list[dict[str, object]]:
        return []

    def update_position(self, symbol: str, payload: dict[str, object]) -> None:
        self.updated.append((symbol, payload))


def test_reconcile_supports_sync_get_positions() -> None:
    """Ensure monitor accepts synchronous broker clients."""

    async def _run() -> None:
        tracker = _Tracker()
        monitor = PostFillMonitor(
            broker_client=_SyncBroker(),
            state_tracker=tracker,
            interval_sec=15,
        )
        await monitor.reconcile()
        assert tracker.updated

    asyncio.run(_run())
