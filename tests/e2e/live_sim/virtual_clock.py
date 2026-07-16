from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(order=True)
class _Scheduled:
    when: datetime
    order: int
    callback: Callable[[], None] = field(compare=False)


class VirtualClock:
    def __init__(self, start: datetime | None = None) -> None:
        start = start or datetime(2026, 7, 15, 8, 55, tzinfo=IST)
        if start.tzinfo is None:
            start = start.replace(tzinfo=IST)
        self._now = start.astimezone(IST)
        self._mono = 0.0
        self._callbacks: list[_Scheduled] = []
        self._counter = 0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def time(self) -> float:
        return self._now.timestamp()

    def call_at(self, when: datetime, callback: Callable[[], None]) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=IST)
        self._counter += 1
        heapq.heappush(
            self._callbacks, _Scheduled(when.astimezone(IST), self._counter, callback)
        )

    def call_later(self, seconds: float, callback: Callable[[], None]) -> None:
        self.call_at(self._now + timedelta(seconds=seconds), callback)

    def advance(self, *, seconds: float = 0.0, milliseconds: float = 0.0) -> None:
        self.advance_to(
            self._now + timedelta(seconds=seconds, milliseconds=milliseconds)
        )

    def advance_to(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)
        target = timestamp.astimezone(IST)
        if target < self._now:
            raise ValueError("virtual clock cannot move backwards")
        while self._callbacks and self._callbacks[0].when <= target:
            item = heapq.heappop(self._callbacks)
            self._mono += (item.when - self._now).total_seconds()
            self._now = item.when
            item.callback()
        self._mono += (target - self._now).total_seconds()
        self._now = target

    def advance_to_next_minute(self) -> None:
        self.advance_to(
            self._now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        )

    def sleep(self, seconds: float) -> None:
        self.advance(seconds=seconds)

    @property
    def pending_callbacks(self) -> int:
        return len(self._callbacks)
