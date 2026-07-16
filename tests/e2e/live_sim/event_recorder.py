from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RecordedEvent:
    sequence: int
    timestamp: datetime
    monotonic_time: float
    event: str
    symbol: str | None = None
    payload: dict[str, Any] | None = None


class EventRecorder:
    def __init__(self, clock) -> None:
        self.clock = clock
        self._events: list[RecordedEvent] = []

    @property
    def events(self) -> tuple[RecordedEvent, ...]:
        return tuple(self._events)

    def record(
        self, event: str, symbol: str | None = None, **payload: Any
    ) -> RecordedEvent:
        item = RecordedEvent(
            len(self._events) + 1,
            self.clock.now(),
            self.clock.monotonic(),
            event,
            symbol,
            dict(payload),
        )
        self._events.append(item)
        return item

    def filter(
        self, *, event: str | None = None, symbol: str | None = None
    ) -> list[RecordedEvent]:
        return [
            item
            for item in self._events
            if (event is None or item.event == event)
            and (symbol is None or item.symbol == symbol)
        ]

    def last(self, event: str) -> RecordedEvent:
        matches = self.filter(event=event)
        if not matches:
            raise AssertionError(
                f"event {event!r} was not recorded; seen={self.names()}"
            )
        return matches[-1]

    def assert_present(self, event: str) -> None:
        if not self.filter(event=event):
            raise AssertionError(f"event {event!r} missing; seen={self.names()}")

    def assert_absent(self, event: str) -> None:
        if self.filter(event=event):
            raise AssertionError(f"event {event!r} unexpectedly present")

    def assert_exactly_once(self, event: str) -> None:
        self.assert_count(event, 1)

    def assert_count(self, event: str, n: int) -> None:
        actual = len(self.filter(event=event))
        if actual != n:
            raise AssertionError(
                f"event {event!r} count {actual} != {n}; seen={self.names()}"
            )

    def assert_before(self, first: str, second: str) -> None:
        if self.last(first).sequence >= self.last(second).sequence:
            raise AssertionError(f"{first!r} did not occur before {second!r}")

    def assert_sequence(self, names: list[str]) -> None:
        cursor = 0
        for name in names:
            for idx in range(cursor, len(self._events)):
                if self._events[idx].event == name:
                    cursor = idx + 1
                    break
            else:
                raise AssertionError(
                    "sequence item "
                    f"{name!r} missing after {cursor}; seen={self.names()}"
                )

    def names(self) -> list[str]:
        return [item.event for item in self._events]
