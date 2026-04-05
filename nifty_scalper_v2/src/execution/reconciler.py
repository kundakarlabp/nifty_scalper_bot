"""Async broker reconciliation loop."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..broker.base import BrokerGateway

log = logging.getLogger(__name__)


class AsyncReconciler:
    """Polls broker every N seconds to sync order/position state."""

    def __init__(self, broker: "BrokerGateway", interval_sec: int = 30) -> None:
        self._broker = broker
        self._interval = interval_sec
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.reconcile()
            except Exception as e:
                log.error("Reconciler error: %s", e)
            await asyncio.sleep(self._interval)

    async def reconcile(self) -> None:
        """Fetch broker state and update local order cache."""
        try:
            await self._broker.get_orders()
            await self._broker.get_positions()
        except Exception as e:
            log.warning("Reconcile failed: %s", e)
