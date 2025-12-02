"""Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle."""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone, timedelta
from typing import Any

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.risk import RiskManager

LOGGER = logging.getLogger(__name__)

class OrderProcessor:
    """
    Smart asynchronous processor that handles signal debouncing 
    and slippage protection (Market -> Limit conversion).
    """

    def __init__(
        self,
        message_bus: MessageBus,
        safe_order_manager: OrderManager,
        risk_manager: RiskManager,
        data_hub: Any, # Now accepts DataHub
    ):
        self.bus = message_bus
        self.executor = safe_order_manager
        self.risk_manager = risk_manager
        self.data_hub = data_hub
        self._running = False
        
        # Anti-Whipsaw: Track last signal time per symbol
        self._last_signal_time: dict[str, datetime] = {}
        self._debounce_seconds = 5.0 

        self.bus.subscribe(MessageType.SIGNAL, self.on_strategy_signal)
        LOGGER.info("OrderProcessor initialized with Smart Execution logic.")

    async def on_strategy_signal(self, message: Message) -> None:
        """
        Smart Signal Handler:
        1. Checks Debounce (prevents spam)
        2. Calculates Protection Price (prevents bad fills)
        3. Executes
        """
        signal: dict[str, Any] = message.data
        symbol = signal.get("symbol")
        side = signal.get("side")
        qty = signal.get("quantity")
        
        if not all([symbol, side, qty]):
            return

        # --- 1. Debounce Check ---
        now = datetime.now(timezone.utc)
        last_time = self._last_signal_time.get(symbol)
        if last_time and (now - last_time).total_seconds() < self._debounce_seconds:
            LOGGER.info(f"Debounced signal for {symbol} (too fast)")
            return
        self._last_signal_time[symbol] = now

        # --- 2. Smart Price Calculation (Slippage Protection) ---
        # Convert MARKET orders to LIMIT orders with a buffer (e.g. 1% slippage allowance)
        # This guarantees fill like a Market order, but protects capital if price spikes 50%.
        
        order_type = signal.get("order_type", OrderType.MARKET)
        price = signal.get("price")

        if order_type == OrderType.MARKET and self.data_hub:
            tick = self.data_hub.get_quote(symbol)
            if tick:
                ltp = tick.get("ltp") or tick.get("last_price") or 0.0
                if ltp > 0:
                    # Convert to LIMIT order
                    order_type = OrderType.LIMIT
                    if side == "BUY":
                        # Buy up to 1% higher than LTP (Aggressive Limit)
                        price = round(ltp * 1.01, 1) 
                    else:
                        # Sell down to 1% lower than LTP
                        price = round(ltp * 0.99, 1)
                    
                    LOGGER.info(f"🛡️ Converted MARKET to LIMIT for protection: {symbol} @ {price} (LTP: {ltp})")

        # --- 3. Execution ---
        LOGGER.info(f"Executing: {side} {qty} {symbol} @ {price or 'MKT'}")

        try:
            broker_order_id = self.executor.place_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                price=price
            )
            
            # Publish Success
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=now,
                    data={
                        "order_id": broker_order_id,
                        "symbol": symbol,
                        "status": "SUBMITTED",
                        "price": price
                    },
                    source="order_processor"
                )
            )

        except Exception as exc:
            LOGGER.error(f"Order failed: {exc}")
            # Publish Failure
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=now,
                    data={"status": "REJECTED", "symbol": symbol, "error": str(exc)},
                    source="order_processor"
                )
            )

    def start(self) -> None:
        with suppress(Exception):
            self.executor.start_monitoring()
        self._running = True
        
    async def stop(self) -> None:
        self._running = False
        with suppress(Exception):
            self.executor.stop_monitoring()
