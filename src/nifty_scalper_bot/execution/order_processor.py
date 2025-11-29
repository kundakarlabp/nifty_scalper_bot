"""Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle."""

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Mapping

import logging
from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.risk import RiskManager

LOGGER = logging.getLogger(__name__)

class OrderProcessor:
    """
    Dedicated asynchronous processor that consumes SIGNAL messages 
    and handles the entire order lifecycle (submission, monitoring, updates).
    """

    def __init__(
        self,
        message_bus: MessageBus,
        safe_order_manager: OrderManager, # The actual executor
        risk_manager: RiskManager,
    ):
        self.bus = message_bus
        self.executor = safe_order_manager
        self.risk_manager = risk_manager
        self._running = False
        self._task: asyncio.Task | None = None
        self.bus.subscribe(MessageType.SIGNAL, self.on_strategy_signal)
        LOGGER.info("OrderProcessor initialized and subscribed to SIGNALs.")

    async def on_strategy_signal(self, message: Message) -> None:
        """
        Receives a SIGNAL from the strategy runner and places the order.
        This is the entry point for all execution logic.
        """
        signal: dict[str, Any] = message.data
        
        symbol = signal.get("symbol")
        side = signal.get("side")
        quantity = signal.get("quantity")
        order_type = signal.get("order_type", OrderType.MARKET)
        price = signal.get("price")
        source = message.source

        if not all([symbol, side, quantity]):
            LOGGER.error("Invalid signal received: %s", signal)
            return

        LOGGER.info(
            "Executing signal from %s: %s %s %s",
            source, side, quantity, symbol,
            extra={"event": "signal_received", "signal": signal}
        )

        try:
            # 1. Place the order using the SafeOrderManager
            broker_order_id = self.executor.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            # 2. Publish initial ORDER_UPDATE to the MessageBus
            # The execution layer is now fully decoupled from the rest of the bot.
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "order_id": broker_order_id,
                        "symbol": symbol,
                        "status": "SUBMITTED",
                        "source_signal": signal,
                    },
                    source="order_processor"
                )
            )

        except Exception as exc:
            LOGGER.error(
                "Order submission failed for %s: %s",
                symbol, exc, exc_info=True,
                extra={"event": "order_submission_failed", "signal": signal}
            )
            
            # 3. Publish REJECTED/FAILED status back to the MessageBus
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "order_id": "N/A",
                        "symbol": symbol,
                        "status": "REJECTED",
                        "error": str(exc),
                    },
                    source="order_processor"
                )
            )
            # The executor itself logs the rejection via its internal hooks, 
            # but this message is needed for the rest of the application.
            
    def start(self) -> None:
        """Start the order processor (mainly for background monitoring tasks)."""
        if self.executor and hasattr(self.executor, 'start_monitoring'):
            # Delegate monitoring duties to the underlying manager (which runs necessary async tasks)
            with suppress(Exception):
                self.executor.start_monitoring()
        self._running = True
        
    async def stop(self) -> None:
        """Stop the order processor."""
        self._running = False
        if self.executor and hasattr(self.executor, 'stop_monitoring'):
            with suppress(Exception):
                self.executor.stop_monitoring()
        LOGGER.info("OrderProcessor stopped.")
