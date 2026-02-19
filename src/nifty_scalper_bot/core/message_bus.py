"""Async message bus for decoupled component communication."""
import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

# --- Message Definitions ---
class MessageType(Enum):
    """Message types flowing through the bus."""
    TICK = "tick"              # Market data tick
    SIGNAL = "signal"          # Strategy signal/request
    ORDER_REQUEST = "order_request"  # Order to execute
    ORDER_UPDATE = "order_update"  # Execution confirmation (Fill/Cancel/Reject)
    POSITION_UPDATE = "position_update"  # Position change
    
T = TypeVar('T')

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
        self.max_queue_size = int(max_queue_size)
        # Separate queue per message type
        self.queues: dict[MessageType, asyncio.Queue] = {
            msg_type: asyncio.Queue(maxsize=self.max_queue_size)
            for msg_type in MessageType
        }
        # Subscribers store (message_type -> list of handler functions)
        self.subscribers: dict[MessageType, list[Callable[[Message], Awaitable[None]]]] = {
            msg_type: [] for msg_type in MessageType
        }
        self._running = False
        self._tasks: list[asyncio.Task] = []
        LOGGER.info("MessageBus initialized with max_queue_size=%s", max_queue_size)

    async def publish(self, message: Message) -> None:
        """Publish a message into the bus queue with deterministic backpressure."""
        queue = self.queues.get(message.type)
        if queue is None:
            raise KeyError(f'Unknown message type: {message.type}')
        depth = queue.qsize()
        if depth > int(self.max_queue_size * 0.8):
            LOGGER.warning(
                'MessageBus queue nearing capacity',
                extra={
                    'event': 'message_bus_queue_high_watermark',
                    'type': message.type.value,
                    'depth': depth,
                    'max_queue_size': self.max_queue_size,
                },
            )
        if self._running:
            active_dispatchers = [t for t in self._tasks if not t.done()]
            assert active_dispatchers, 'MessageBus dispatcher task is not alive'
        await queue.put(message)

    def subscribe(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Awaitable[None]]
    ) -> None:
        """Subscribe an async handler function to a message type."""
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(f"Handler for {message_type.value} must be an async function.")
        self.subscribers[message_type].append(handler)
        LOGGER.info("Component subscribed to %s", message_type.value)

    async def _dispatch_loop(self, message_type: MessageType) -> None:
        """Dispatch messages from a queue to its subscribers.

        Design invariants:
        - Each handler is isolated: one failure NEVER kills sibling handlers.
        - The loop NEVER exits on handler errors — only on CancelledError.
        - Every handler failure is logged with full context (observable + fail-fast).
        """
        queue = self.queues[message_type]
        handlers = self.subscribers[message_type]

        while self._running:
            try:
                message = await queue.get()
                queue.task_done()

                # ── Observable: pipeline stage marker ──
                if message.type == MessageType.TICK:
                    LOGGER.debug(
                        "BUS_DELIVER symbol=%s stage=MessageBus",
                        message.data.get("symbol"),
                        extra={
                            "event": "bus_deliver",
                            "symbol": message.data.get("symbol"),
                            "pipeline_stage": "BUS_DISPATCH",
                        },
                    )

                for handler in handlers:
                    h_name = getattr(handler, "__name__", repr(handler))
                    try:
                        await handler(message)
                    except Exception as exc:
                        # Fail-fast: log every error with structured context.
                        # Do NOT re-raise — sibling handlers must still execute.
                        # A re-raise here turns a single bad handler into a
                        # full message blackout for all subscribers.
                        LOGGER.error(
                            "MessageBus handler %s failed for %s: %s",
                            h_name,
                            message_type.value,
                            exc,
                            extra={
                                "event": "bus_handler_error",
                                "handler": h_name,
                                "message_type": message_type.value,
                                "pipeline_stage": "BUS_HANDLER_FAULT",
                            },
                            exc_info=exc,
                        )

            except asyncio.CancelledError:
                LOGGER.debug("%s dispatch loop cancelled.", message_type.value)
                raise
            except Exception as exc:
                # Queue.get() itself failed — log and keep the loop alive.
                LOGGER.error(
                    "MessageBus dispatch error in %s loop: %s",
                    message_type.value,
                    exc,
                    extra={"event": "bus_dispatch_error", "message_type": message_type.value},
                    exc_info=exc,
                )

    def start(self) -> None:
        """Start all dispatch loops."""
        if self._running:
            return
        self._running = True
        self._loop: asyncio.AbstractEventLoop | None = None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        for msg_type in MessageType:
            if self.subscribers[msg_type]:
                task = asyncio.create_task(
                    self._dispatch_loop(msg_type),
                    name=f"dispatch-{msg_type.value}"
                )
                self._tasks.append(task)

        LOGGER.info("Message bus started with %d active dispatchers.", len(self._tasks))

    def publish_from_thread(self, message: "Message") -> None:
        """Thread-safe publish: callable from any OS thread, not just asyncio tasks.

        Uses ``loop.call_soon_threadsafe`` + ``queue.put_nowait`` — the only
        documented-correct way to push items into an asyncio.Queue from a
        non-async thread (e.g. KiteConnect WS thread, PollingStreamer thread).

        Args:
            message: Message to publish.
        """
        loop = getattr(self, "_loop", None)
        if loop is None or not loop.is_running():
            # Loop not yet available — buffer via put_nowait directly
            # (safe only because we're single-producer here during startup)
            try:
                self.queues[message.type].put_nowait(message)
            except Exception:
                pass
            return
        try:
            self.queues[message.type].put_nowait  # validate key exists
            loop.call_soon_threadsafe(
                self.queues[message.type].put_nowait, message
            )
        except KeyError:
            LOGGER.error("publish_from_thread: unknown MessageType %s", message.type)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("publish_from_thread failed: %s", exc)

    async def stop(self) -> None:
        """Stop all dispatch loops."""
        if not self._running:
            return
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            
        with asyncio.TaskGroup() as tg:
            for task in self._tasks:
                tg.create_task(self._await_cancellation(task))
                
        self._tasks.clear()
        LOGGER.info("Message bus stopped.")

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
