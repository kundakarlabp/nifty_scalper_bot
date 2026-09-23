"""Best-effort background persistence for structured trade telemetry."""

from __future__ import annotations

import logging
import os
from queue import Empty, Full, Queue
from threading import Event, Thread

import requests

from nifty_scalper_bot.infra.trade_observability import TradeEvent

LOGGER = logging.getLogger(__name__)


class TradeEventSink:
    """Bounded asynchronous writer; persistence never blocks trading."""

    def __init__(self, queue_size: int = 4096, timeout_seconds: float = 3.0) -> None:
        self._queue: Queue[TradeEvent] = Queue(maxsize=max(1, queue_size))
        self._timeout_seconds = timeout_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.dropped = 0
        self.failed = 0
        self.persisted = 0

    @property
    def enabled(self) -> bool:
        return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._thread = Thread(target=self._run, name="trade-event-sink", daemon=True)
        self._thread.start()

    def emit(self, event: TradeEvent) -> bool:
        if not self.enabled:
            return False
        self.start()
        try:
            self._queue.put_nowait(event)
        except Full:
            self.dropped += 1
            return False
        return True

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                self._persist(event)
                self.persisted += 1
            except Exception as exc:  # noqa: BLE001
                self.failed += 1
                LOGGER.warning(
                    "trade_event_persist_failed",
                    extra={
                        "event": "trade_event_persist_failed",
                        "error_type": type(exc).__name__,
                    },
                )
            finally:
                self._queue.task_done()

    def _persist(self, event: TradeEvent) -> None:
        base_url = os.environ["SUPABASE_URL"].rstrip("/")
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        response = requests.post(
            f"{base_url}/rest/v1/nifty_trade_events",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=event.row(),
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()


TRADE_EVENT_SINK = TradeEventSink()

__all__ = ["TradeEventSink", "TRADE_EVENT_SINK"]
