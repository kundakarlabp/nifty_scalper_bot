"""Thread-safe event bus for decoupling publishers and strategy workers."""

from __future__ import annotations

import queue
from typing import Any, Iterator


class EventBus:
    """Provide bounded in-memory event queue."""

    def __init__(self) -> None:
        """Initialize the event queue. Args: none. Returns: none. Raises: none."""
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=10000)

    def publish(self, event: Any) -> None:
        """Publish an event to consumers."""
        self.queue.put_nowait(event)

    def consume(self) -> Iterator[Any]:
        """Yield events forever. Args: none. Returns: iterator. Raises: none."""
        while True:
            yield self.queue.get()
