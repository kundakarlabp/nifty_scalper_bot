"""Helpers for tracking logical execution traces across subsystems."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


@dataclass(slots=True)
class Trace:
    """Represents a logical trace for related log and diagnostic events."""

    trace_id: str
    signal_id: str | None = None
    order_client_id: str | None = None


def new_trace(signal_id: str | None = None) -> Trace:
    """Return a new :class:`Trace` with a random ``trace_id``."""

    return Trace(trace_id=uuid4().hex, signal_id=signal_id)


__all__ = ["Trace", "new_trace"]

