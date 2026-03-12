"""Priority-based order request queue."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Literal

from nifty_scalper_bot.execution.metrics import QUEUE_DEPTH, QUEUE_SUBMISSIONS
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

OrderIntent = Literal["ENTRY", "EXIT_SL", "EXIT_TP1", "EXIT_TP2", "ADJUST_TRAIL"]

_PRIORITY_MAP: dict[OrderIntent, int] = {
    "EXIT_SL": 20,
    "EXIT_TP1": 8,
    "EXIT_TP2": 8,
    "ADJUST_TRAIL": 5,
    "ENTRY": 3,
}

_EMERGENCY_EXIT_INTENTS: set[OrderIntent] = {"EXIT_SL", "EXIT_TP1", "EXIT_TP2"}

_COUNTER = itertools.count()


@dataclass(slots=True)
class OrderRequest:
    """Order request payload stored in the priority queue."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    intent: OrderIntent
    parent_id: str | None = None
    source: str = ""
    priority: int = field(init=False)
    created_ts: float = field(default_factory=time.monotonic, init=False)
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Derive ordering priority from the request intent.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If the intent is not supported.
        """

        LOGGER.debug(
            "Entered OrderRequest.__post_init__",
            extra={"event": "order_request_init", "intent": self.intent},
        )
        try:
            priority_value = _PRIORITY_MAP[self.intent]
        except KeyError as exc:
            LOGGER.error(
                "Failure in OrderRequest.__post_init__: %s",
                exc,
                extra={"event": "order_request_invalid_intent", "intent": self.intent},
                exc_info=exc,
            )
            raise ValueError(f"Unsupported intent: {self.intent!s}") from exc
        self.priority = int(priority_value)


class OrderQueue:
    """Thread-safe priority queue for downstream execution routing."""

    def __init__(self) -> None:
        """Initialise queue and deduplication helpers.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderQueue.__init__",
            extra={"event": "order_queue_init"},
        )
        self._queue: PriorityQueue[tuple[int, int, OrderRequest]] = PriorityQueue()
        self._lock = threading.Lock()
        self._dedup: set[tuple[str, str, OrderIntent]] = set()
        self._paused = False

    def submit_order_request(self, request: OrderRequest) -> None:
        """Add ``request`` to the queue when it is not a duplicate.

        Args:
            request: OrderRequest instance to enqueue.

        Returns:
            None.

        Raises:
            RuntimeError: If the queue is paused or enqueue fails unexpectedly.
            ValueError: If a duplicate request is submitted.
        """

        LOGGER.debug(
            "Entered OrderQueue.submit_order_request",
            extra={
                "event": "order_queue_submit_enter",
                "intent": request.intent,
                "symbol": request.symbol,
            },
        )
        try:
            with self._lock:
                if self._paused:
                    raise RuntimeError("Order queue is paused")
                dedup_key = (
                    request.symbol.upper(),
                    request.side.upper(),
                    request.intent,
                )
                if dedup_key in self._dedup and request.intent not in _EMERGENCY_EXIT_INTENTS:
                    LOGGER.warning(
                        "order_queue_duplicate_rejected",
                        extra={
                            "event": "order_queue_duplicate",
                            "symbol": request.symbol,
                            "side": request.side,
                            "intent": request.intent,
                        },
                    )
                    raise ValueError(
                        "Duplicate order request: "
                        f"{request.symbol} {request.side} {request.intent}"
                    )
                self._dedup.add(dedup_key)
                priority_tuple = (-request.priority, next(_COUNTER), request)
                self._queue.put(priority_tuple)
                QUEUE_SUBMISSIONS.labels(
                    intent=request.intent, symbol=request.symbol
                ).inc()
                QUEUE_DEPTH.set(self._queue.qsize())
            LOGGER.debug(
                "order_queue_submitted",
                extra={
                    "event": "order_queue_submitted",
                    "symbol": request.symbol,
                    "intent": request.intent,
                    "priority": request.priority,
                    "queue_size": self._queue.qsize(),
                },
            )
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            LOGGER.error(
                "Failure in OrderQueue.submit_order_request: %s",
                exc,
                extra={"event": "order_queue_submit_error"},
                exc_info=exc,
            )
            raise RuntimeError("Failed to enqueue order request") from exc

    def get_next_request(self, timeout: float | None = None) -> OrderRequest | None:
        """Retrieve the highest priority request from the queue.

        Args:
            timeout: Maximum seconds to wait for an item. ``None`` blocks indefinitely.

        Returns:
            OrderRequest | None: Next request or ``None`` if the queue is empty.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderQueue.get_next_request",
            extra={"event": "order_queue_get_enter"},
        )
        try:
            priority_tuple = self._queue.get(timeout=timeout)
        except Empty:
            LOGGER.debug(
                "order_queue_timeout",
                extra={"event": "order_queue_timeout"},
            )
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderQueue.get_next_request: %s",
                exc,
                extra={"event": "order_queue_get_error"},
                exc_info=exc,
            )
            return None

        _, _, request = priority_tuple
        dedup_key = (request.symbol.upper(), request.side.upper(), request.intent)
        with self._lock:
            self._dedup.discard(dedup_key)
        LOGGER.debug(
            "order_queue_dequeued",
            extra={
                "event": "order_queue_dequeued",
                "symbol": request.symbol,
                "intent": request.intent,
                "queue_size": self._queue.qsize(),
            },
        )
        QUEUE_DEPTH.set(self._queue.qsize())
        return request

    def pause(self) -> None:
        """Mark the queue as paused to reject further submissions.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        with self._lock:
            self._paused = True
        LOGGER.info(
            "order_queue_paused",
            extra={"event": "order_queue_paused"},
        )

    def resume(self) -> None:
        """Resume queue processing after a pause.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        with self._lock:
            self._paused = False
        LOGGER.info(
            "order_queue_resumed",
            extra={"event": "order_queue_resumed"},
        )

    def is_paused(self) -> bool:
        """Return ``True`` when the queue currently rejects submissions.

        Args:
            None.

        Returns:
            bool: ``True`` if paused, otherwise ``False``.

        Raises:
            None.
        """

        with self._lock:
            paused = self._paused
        LOGGER.debug(
            "order_queue_is_paused",
            extra={"event": "order_queue_is_paused", "paused": paused},
        )
        return paused

    def get_queue_snapshot(self) -> list[dict[str, object]]:
        """Return a diagnostic snapshot of queued requests.

        Args:
            None.

        Returns:
            list[dict[str, object]]: Pending order request metadata.

        Raises:
            None.
        """

        snapshot: list[dict[str, object]] = []
        with self._queue.mutex:  # type: ignore[attr-defined]
            for priority_value, _, request in list(self._queue.queue):  # type: ignore[attr-defined]
                snapshot.append(
                    {
                        "symbol": request.symbol,
                        "side": request.side,
                        "intent": request.intent,
                        "priority": -priority_value,
                        "age_sec": time.monotonic() - request.created_ts,
                        "source": request.source,
                    }
                )
        LOGGER.debug(
            "order_queue_snapshot_generated",
            extra={"event": "order_queue_snapshot", "size": len(snapshot)},
        )
        return snapshot


__all__ = ["OrderQueue", "OrderRequest", "OrderIntent"]
