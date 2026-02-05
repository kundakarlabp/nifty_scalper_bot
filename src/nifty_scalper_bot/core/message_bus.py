"""Async message bus for decoupled component communication."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar
import logging

LOGGER = logging.getLogger(__name__)

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
    
    def __init__(self, max_queue_size: int = 1000):
        # Separate queue per message type
        self.queues: dict[MessageType, asyncio.Queue] = {
            msg_type: asyncio.Queue(maxsize=max_queue_size)
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
        """Publish message to appropriate queue with pre-start buffering.
        
        ✅ FIX #2: Instead of dropping messages before start(), buffer TICK 
        messages so they're processed when the bus starts. This prevents
        losing initial market data during startup.
        """
        if not self._running:
            # Buffer tick messages for later dispatch
            if message.type == MessageType.TICK:
                try:
                    # Use put_nowait to avoid blocking - queue will buffer
                    self.queues[message.type].put_nowait(message)
                    LOGGER.debug(
                        f"⏳ Buffered pre-start tick for {message.data.get('symbol', 'unknown')}",
                        extra={"event": "tick_buffered", "type": message.type.value}
                    )
                except asyncio.QueueFull:
                    # Only log occasionally to avoid spam
                    if not hasattr(self, '_buffer_warn_count'):
                        self._buffer_warn_count = 0
                    self._buffer_warn_count += 1
                    if self._buffer_warn_count % 100 == 1:
                        LOGGER.warning(
                            f"⚠️ Pre-start tick buffer full (dropped {self._buffer_warn_count} ticks)",
                            extra={"event": "buffer_full"}
                        )
                return
            else:
                # Non-tick messages before start are logged but not dropped
                LOGGER.warning(
                    f"⚠️ MessageBus received '{message.type.value}' before start() - queuing",
                    extra={"event": "pre_start_message", "type": message.type.value}
                )
                try:
                    self.queues[message.type].put_nowait(message)
                except asyncio.QueueFull:
                    LOGGER.error(f"Queue full for pre-start {message.type.value}")
                return
        
        # Normal publish (bus is running)
        try:
            await self.queues[message.type].put(message)
        except asyncio.QueueFull:
            LOGGER.error(
                f"Queue full for {message.type.value} - dropping message.",
                extra={"event": "message_drop", "type": message.type.value}
            )
        except KeyError:
            LOGGER.error(f"Attempted to publish unknown message type: {message.type.value}")

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
        """Dispatch messages from a queue to its subscribers."""
        queue = self.queues[message_type]
        handlers = self.subscribers[message_type]
        
        while self._running:
            try:
                # Wait for message
                message = await queue.get()
                queue.task_done()
                
                # Dispatch to all handlers concurrently
                await asyncio.gather(
                    *[handler(message) for handler in handlers],
                    return_exceptions=True # Don't let one handler crash the whole loop
                )
                                            
            except asyncio.CancelledError:
                LOGGER.info("%s dispatch loop cancelled.", message_type.value)
                raise
            except Exception as exc:
                LOGGER.error(f"Critical dispatch error in {message_type.value} loop: {exc}", exc_info=True)

    def start(self) -> None:
        """Start all dispatch loops."""
        if self._running:
            return
        self._running = True
        
        for msg_type in MessageType:
            if self.subscribers[msg_type]:
                task = asyncio.create_task(
                    self._dispatch_loop(msg_type),
                    name=f"dispatch-{msg_type.value}"
                )
                self._tasks.append(task)
        
        LOGGER.info("Message bus started with %d active dispatchers.", len(self._tasks))

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

    async def _await_cancellation(self, task: asyncio.Task):
         with suppress(asyncio.CancelledError):
            await task
