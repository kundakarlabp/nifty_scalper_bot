"""REST polling fallback stream."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.tick_hub import TickHub
    from ..broker.base import BrokerGateway
    from ..config.settings import StreamingSettings

log = logging.getLogger(__name__)
_BATCH_SIZE = 200


class PollingStream:
    """Polls Zerodha REST quote endpoint as fallback when WebSocket unavailable."""

    def __init__(
        self,
        tick_hub: "TickHub",
        broker: "BrokerGateway",
        settings: "StreamingSettings",
        token_map: dict[int, str],
    ) -> None:
        self._hub = tick_hub
        self._broker = broker
        self._settings = settings
        self._token_map = token_map
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_counts: dict[str, int] = {}

    async def start(self) -> None:
        self._running = True
        for sym in self._token_map.values():
            self._hub.register_symbol(sym)
        self._task = asyncio.create_task(self._poll_loop())
        log.info("Polling stream started for %d symbols", len(self._token_map))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        interval = self._settings.polling_interval_ms / 1000.0
        symbols = [f"NFO:{sym}" for sym in self._token_map.values()]

        while self._running:
            try:
                # Batch requests
                for i in range(0, len(symbols), _BATCH_SIZE):
                    batch = symbols[i:i + _BATCH_SIZE]
                    quotes = await self._broker.get_quote(batch)
                    await self._process_quotes(quotes)
            except Exception as e:
                log.error("Polling error: %s", e)
            await asyncio.sleep(interval)

    async def _process_quotes(self, quotes: dict) -> None:
        from ..data.tick_hub import Tick
        for sym_key, data in quotes.items():
            sym = sym_key.replace("NFO:", "")
            count = self._tick_counts.get(sym, 0) + 1
            self._tick_counts[sym] = count
            is_live = count > self._settings.warmup_ticks

            tick = Tick(
                symbol=sym,
                instrument_token=data.get("instrument_token", 0),
                ltp=data.get("last_price", 0.0),
                bid=data.get("depth", {}).get("buy", [{}])[0].get("price", 0.0) if data.get("depth") else 0.0,
                ask=data.get("depth", {}).get("sell", [{}])[0].get("price", 0.0) if data.get("depth") else 0.0,
                volume=data.get("volume", 0),
                oi=data.get("oi", 0),
                timestamp=datetime.now(timezone.utc),
                is_backfill=not is_live,
            )
            await self._hub.on_tick(tick)
