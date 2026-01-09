"""Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle."""

import asyncio
from nifty_scalper_bot.utils.logging import logger
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
        self._last_signal_time: dict[tuple[str, str], datetime] = {}
        self._active_trades: dict[str, str] = {}

        # Cooldown (seconds)
        self._debounce_seconds = 60.0

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

        key = (symbol, side)
        # 🔒 Intent lock (async-safe)
        if self._active_trades.get(key):
            LOGGER.warning(f"🚫 Intent already active: {symbol} {side}")
            return

        # --- 1. Debounce Check ---
        now = datetime.now(timezone.utc)
        last_time = self._last_signal_time.get(key)
        if last_time and (now - last_time).total_seconds() < self._debounce_seconds:
            LOGGER.info(
                f"⏳ Cooldown active: {symbol} {side} "
                f"({int((now - last_time).total_seconds())}s)"
            )
            return

        self._last_signal_time[key] = now
        self._active_trades[key] = "INTENT"


        # --- 2. Smart Price Calculation (Slippage Protection) ---
        # Convert MARKET orders to LIMIT orders with a protection buffer
        order_type = signal.get("order_type", OrderType.MARKET)
        price = signal.get("price")

        # Force Limit protection for all Options orders
        if self.data_hub:
            tick = self.data_hub.get_quote(symbol)
            if tick:
                ltp = tick.get("ltp") or tick.get("last_price") or 0.0
                
                # If user asked for MARKET, or we are in LIVE mode, convert to SAFE LIMIT
                if ltp > 0:
                    order_type = OrderType.LIMIT
                    
                    # 2% buffer is standard for Nifty scalping
                    buffer_pct = 1.02 
                    
                    if side == "BUY":
                        # Buy at LTP + 2% (Aggressive Limit)
                        raw_price = ltp * buffer_pct
                        price = round(raw_price / 0.05) * 0.05
                    else:
                        # Sell at LTP - 2%
                        price = round(ltp * (2 - buffer_pct), 1)
                        
                    LOGGER.info(f"🛡️ Safety Limit Applied: {symbol} {side} | LTP: {ltp} | Limit Price: {price}")

        # --- 3. Execution ---
        # Register active trade

        LOGGER.info(f"Executing: {side} {qty} {symbol} @ {price or 'MKT'}")

        try:
            broker_order_id = self.executor.place_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                price=price
            )
            # 🔒 Register active trade ONLY after broker ACK
            self._active_trades[key] = side
            
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
            # 🔓 Release lock on failure
            self._active_trades.pop(key, None)
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
