"""Tick hub: central in-memory tick cache + async event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable


@dataclass(slots=True, frozen=True)
class Tick:
    """Immutable tick snapshot. is_backfill=True during warmup — strategies must ignore."""
    symbol: str
    instrument_token: int
    ltp: float
    bid: float
    ask: float
    volume: int
    oi: int
    timestamp: datetime
    is_backfill: bool = False

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ltp

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.spread / m) * 100.0 if m > 0 else 0.0


EventCallback = Callable[..., Awaitable[None] | None]


class EventBus:
    """Async pub/sub. Non-blocking publish from OS threads."""

    def __init__(self) -> None:
        self._subs: dict[str, list[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: str, cb: EventCallback) -> None:
        self._subs[event_type].append(cb)

    def publish(self, event_type: str, data: Any) -> None:
        """Fire callbacks; schedule coroutines on the running loop."""
        for cb in self._subs.get(event_type, []):
            result = cb(data)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass

    async def publish_async(self, event_type: str, data: Any) -> None:
        for cb in self._subs.get(event_type, []):
            result = cb(data)
            if asyncio.iscoroutine(result):
                await result


class TickHub:
    """Central tick cache and event dispatcher. Thread-safe."""

    def __init__(self, warmup_threshold: int = 100) -> None:
        self._cache: dict[str, Tick] = {}
        self._warmup_threshold = warmup_threshold
        self._warmup_counts: dict[str, int] = defaultdict(int)
        self._warmup_complete = False
        self._subscribed: set[str] = set()
        self.event_bus = EventBus()

    def register_symbol(self, symbol: str) -> None:
        self._subscribed.add(symbol)

    def on_tick_sync(self, tick: Tick) -> None:
        """Thread-safe ingestion from kiteconnect OS thread."""
        self._cache[tick.symbol] = tick
        if tick.symbol in self._subscribed:
            self._warmup_counts[tick.symbol] += 1
        self.event_bus.publish("tick", tick)
        self._check_warmup()

    async def on_tick(self, tick: Tick) -> None:
        self._cache[tick.symbol] = tick
        if tick.symbol in self._subscribed:
            self._warmup_counts[tick.symbol] += 1
        await self.event_bus.publish_async("tick", tick)
        self._check_warmup()

    def _check_warmup(self) -> None:
        if not self._warmup_complete and self._subscribed:
            if all(
                self._warmup_counts.get(s, 0) >= self._warmup_threshold
                for s in self._subscribed
            ):
                self.set_warmup_complete()

    def set_warmup_complete(self) -> None:
        if not self._warmup_complete:
            self._warmup_complete = True
            self.event_bus.publish("warmup_complete", None)

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    def get_latest(self, symbol: str) -> Tick | None:
        return self._cache.get(symbol)

    def get_all_latest(self) -> dict[str, Tick]:
        return dict(self._cache)
