"""Async message bus for decoupled component communication."""

import asyncio
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Awaitable, Callable, TypeVar

from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


# --- Message Definitions ---
class MessageType(Enum):
    """Message types flowing through the bus."""

    TICK = "tick"
    DATA_READY = "data_ready"
    SIGNAL = "signal"  # Strategy signal/request
    ORDER_REQUEST = "order_request"  # Order to execute
    ORDER_UPDATE = "order_update"  # Execution confirmation (Fill/Cancel/Reject)
    POSITION_UPDATE = "position_update"  # Position change


T = TypeVar("T")


@dataclass(frozen=True)
class Message:
    """Standardized message envelope."""

    type: MessageType
    timestamp: datetime
    data: dict[str, Any]
    source: str  # Component that created message


# --- Message Bus Core ---
class MessageBus:
    """
    Central async message bus.
    Components communicate ONLY via this bus - no direct calls.
    """

    def __init__(self, max_queue_size: int = 5000):
        """Initialize message queues and overflow accounting."""
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        self._max_queue_size = int(max_queue_size)
        self._soft_drop_threshold = max(int(self._max_queue_size * 0.8), 1)
        # Separate queue per message type
        self.queues: dict[MessageType, asyncio.Queue] = {
            msg_type: asyncio.Queue(maxsize=self._max_queue_size)
            for msg_type in MessageType
        }
        # Subscribers store (message_type -> list of handler functions)
        self.subscribers: dict[
            MessageType, list[Callable[[Message], Awaitable[None]]]
        ] = {msg_type: [] for msg_type in MessageType}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._subscriber_ids: dict[MessageType, set[str]] = {
            msg_type: set() for msg_type in MessageType
        }
        self._dropped_counts: dict[MessageType, int] = defaultdict(int)
        self._first_tick_delivered_symbols: set[str] = set()
        LOGGER.info(
            "MessageBus initialized with max_queue_size=%s", self._max_queue_size
        )

    async def publish(self, message: Message) -> None:
        """Publish a message to subscribed handlers."""
        if not self._running:
            raise RuntimeError(
                f"Publishing {message.type} before MessageBus.start()"
            )
        try:
            self.queues[message.type].put_nowait(message)
        except asyncio.QueueFull:
            self._dropped_counts[message.type] += 1
            raise RuntimeError(f"Queue full for {message.type}")
    def subscribe_once(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Awaitable[None]],
        *,
        subscriber_id: str | None = None,
    ) -> bool:
        """Subscribe async handler once. Args: message_type/handler/subscriber_id. Returns: bool. Raises: TypeError."""
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(f"Handler for {message_type.value} must be an async function.")

        sid = subscriber_id or (
            f"{getattr(handler, '__module__', '')}."
            f"{getattr(handler, '__qualname__', repr(handler))}"
        )
        if sid in self._subscriber_ids[message_type]:
            LOGGER.info(
                "MESSAGE_BUS_SUBSCRIPTION_ALREADY_EXISTS message_type=%s subscriber_id=%s",
                message_type.value,
                sid,
            )
            return False

        self.subscribers[message_type].append(handler)
        self._subscriber_ids[message_type].add(sid)
        LOGGER.info("MESSAGE_BUS_SUBSCRIBED message_type=%s subscriber_id=%s", message_type.value, sid)
        return True

    def subscribe(
        self, message_type: MessageType, handler: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe an async handler function to a message type."""
        self.subscribe_once(message_type, handler)

    @property
    def running(self) -> bool:
        """Args: none. Returns: running flag. Raises: none."""
        return self._running

    def queue_diagnostics(self) -> dict[str, dict[str, int]]:
        """Return queue depth and drop counters per message type."""
        return {
            msg_type.value: {
                "depth": self.queues[msg_type].qsize(),
                "dropped": int(self._dropped_counts.get(msg_type, 0)),
            }
            for msg_type in MessageType
        }

    def _log_task_failure(self, task: asyncio.Task[Any]) -> None:
        """Log async handler task failures safely. Args: task. Returns: none. Raises: none."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception as err:  # noqa: BLE001 - defensive callback safety
            LOGGER.error(
                "MESSAGE_BUS_HANDLER_FAILED error=%r",
                err,
                extra={"event": "MESSAGE_BUS_HANDLER_FAILED"},
            )
            return
        if exc is not None:
            LOGGER.error(
                "MESSAGE_BUS_HANDLER_FAILED",
                extra={"event": "MESSAGE_BUS_HANDLER_FAILED"},
                exc_info=exc,
            )

    async def _dispatch_loop(self, message_type: MessageType) -> None:
        """Dispatch messages from a queue to its subscribers."""
        queue = self.queues[message_type]
        handlers = self.subscribers[message_type]

        while self._running:
            try:
                # Wait for message
                message = await queue.get()

                # Dispatch without head-of-line blocking.
                for handler in handlers:
                    try:
                        result = handler(message)
                        if asyncio.iscoroutine(result):
                            task = safe_task(result)
                            task.add_done_callback(self._log_task_failure)
                    except Exception as exc:
                        LOGGER.error(
                            "Failure in MessageBus handler dispatch: %s",
                            exc,
                            extra={"event": "message_bus_handler_dispatch_error"},
                            exc_info=exc,
                        )
                    if message_type == MessageType.TICK:
                        symbol = str((message.data or {}).get("symbol") or "")
                        if symbol and symbol not in self._first_tick_delivered_symbols:
                            self._first_tick_delivered_symbols.add(symbol)
                            LOGGER.info(
                                "MESSAGE_BUS_TICK_DELIVERED_TO_DATAHUB symbol=%s",
                                symbol,
                                extra={
                                    "event": "MESSAGE_BUS_TICK_DELIVERED_TO_DATAHUB",
                                    "symbol": symbol,
                                },
                            )

                queue.task_done()

            except asyncio.CancelledError:
                LOGGER.debug("%s dispatch loop cancelled.", message_type.value)
                raise
            except Exception as exc:  # noqa: BLE001 - defensive logging
                LOGGER.error(
                    "Critical dispatch error in %s loop: %s",
                    message_type.value,
                    exc,
                    exc_info=exc,
                )

    def start(self) -> bool:
        """Start all dispatch loops. Args: none. Returns: started. Raises: none."""
        if self._running:
            return True
        if not any(self.subscribers.values()):
            LOGGER.warning("MESSAGE_BUS_START_SKIPPED reason=no_subscribers")
            return False
        self._running = True

        for msg_type in MessageType:
            if self.subscribers[msg_type]:
                task = safe_task(self._dispatch_loop(msg_type))
                self._tasks.append(task)

        LOGGER.info("Message bus started with %d active dispatchers.", len(self._tasks))
        return True

    async def stop(self) -> None:
        """Stop dispatchers and drain queues. Safe to call repeatedly."""
        if not getattr(self, "_running", False):
            return

        self._running = False

        tasks = list(getattr(self, "_tasks", []) or [])
        try:
            self._tasks.clear()
        except Exception:
            pass

        for task in tasks:
            if task is not None and not task.done():
                task.cancel()

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, Exception):
                    LOGGER.warning(
                        "MESSAGE_BUS_TASK_STOP_FAILED error=%r",
                        result,
                    )

        queues = getattr(self, "queues", None)
        if isinstance(queues, dict):
            for queue in queues.values():
                try:
                    while not queue.empty():
                        queue.get_nowait()
                        queue.task_done()
                except Exception:
                    pass

        LOGGER.info("MESSAGE_BUS_STOPPED")

    async def _await_cancellation(self, task: asyncio.Task) -> None:
        """Safely await a task during cancellation, suppressing CancelledError."""
        try:
            with suppress(asyncio.CancelledError):
                await task
        except Exception as exc:  # noqa: BLE001 - defensive logging
            LOGGER.error(
                "Failure in MessageBus._await_cancellation: %s",
                exc,
                extra={"event": "message_bus_cancel_error"},
                exc_info=exc,
            )
