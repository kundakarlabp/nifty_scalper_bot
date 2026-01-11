"""
Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle.
Production-Grade: Handles Risk Checks, Thread Safety, Position Awareness, and Exit Priority.
"""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import RiskManager

# --- Order Execution States ---
INTENT = "INTENT"   # Signal accepted, execution pending

LOGGER = logging.getLogger(__name__)

class OrderProcessor:
    """
    Orchestrates order execution with strict Risk Management, Concurrency Control,
    and Position Awareness.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        safe_order_manager: OrderManager,
        risk_manager: RiskManager,
        position_manager: PositionManager, 
        data_hub: Any, 
    ):
        self.bus = message_bus
        self.executor = safe_order_manager
        self.risk_manager = risk_manager
        self.pos_manager = position_manager
        self.data_hub = data_hub
        self._running = False
        
        # 1. Thread Safety Lock
        self._lock = asyncio.Lock()
        
        # 2. State Tracking
        self._active_trades: Dict[str, str] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        
        # 3. Settings
        self._debounce_seconds = 60.0

        self.bus.subscribe(MessageType.SIGNAL, self.on_strategy_signal)
        LOGGER.info("✅ OrderProcessor initialized with Risk & Position Gating.")

    async def on_strategy_signal(self, message: Message) -> None:
        """
        Process signal -> Check Position/Risk -> Execute Order.
        """
        signal: dict[str, Any] = message.data
        symbol = signal.get("symbol")
        side = signal.get("side")      # "BUY" / "SELL"
        qty = signal.get("quantity")
        
        if not all([symbol, side, qty]):
            return

        # --- PHASE 1: INTELLIGENT GATING ---
        async with self._lock:
            key = symbol 
            
            # 1. Check Busy State
            if self._active_trades.get(key):
                LOGGER.warning(f"🚫 Execution busy for {symbol}")
                return

            # 2. Position Awareness & Exit Priority
            # Check if we hold a position in this symbol
            current_pos = None
            if self.pos_manager:
                # We iterate because PositionManager might key by different format
                all_pos = self.pos_manager.get_all_positions()
                for p in all_pos:
                    if p.symbol == symbol:
                        current_pos = p
                        break
            
            is_exit = False
            if current_pos and current_pos.quantity != 0:
                # If we are BUYing and have negative qty -> Closing
                # If we are SELLing and have positive qty -> Closing
                if (side == "BUY" and current_pos.quantity < 0) or \
                   (side == "SELL" and current_pos.quantity > 0):
                    is_exit = True

            # 3. Smart Debounce
            # If it's an EXIT, we SKIP the timer (Get out fast!)
            # If it's an ENTRY, we enforce the timer.
            if not is_exit:
                now = datetime.now(timezone.utc)
                last_time = self._last_signal_time.get(key)
                if last_time and (now - last_time).total_seconds() < self._debounce_seconds:
                    LOGGER.info(f"⏳ Cooldown active for Entry: {symbol}")
                    return
                
                # 4. Anti-Stacking
                # If we already have a position and this is NOT an exit, BLOCK IT.
                # This prevents "Double Exposure" if strategy misfires.
                if current_pos and abs(current_pos.quantity) > 0:
                    LOGGER.warning(f"🚫 Rejecting Stacked Entry for {symbol}. Position exists.")
                    return

            # 5. Risk Management Gate
            if self.risk_manager:
                # Unpack tuple (allowed, reason) from RiskManager.can_trade
                allowed, reason = self.risk_manager.can_trade(symbol, side, qty)
                if not allowed:
                    LOGGER.error(f"🛡️ Risk Rejection for {symbol}: {reason}")
                    return

            # Lock Resources
            self._last_signal_time[key] = datetime.now(timezone.utc)
            self._active_trades[key] = INTENT

        # --- PHASE 2: EXECUTION ---
        
        order_type = signal.get("order_type", "MARKET")
        price = signal.get("price", 0.0)
        
        # Limit Price Logic
        if order_type == "LIMIT" and price == 0.0 and self.data_hub:
            tick = self.data_hub.get_quote(symbol)
            if tick and tick.get("ltp"):
                ltp = float(tick["ltp"])
                # Apply 1% buffer to ensure limit fills
                buffer = 1.01 if side == "BUY" else 0.99
                price = round((ltp * buffer) / 0.05) * 0.05

        LOGGER.info(f"🚀 Executing: {side} {qty} {symbol} @ {price or 'MKT'} (Is Exit: {is_exit})")

        try:
            # Use asyncio.to_thread because place_order is blocking
            broker_order_id = await asyncio.to_thread(
                self.executor.place_order,
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                price=price,
                tag="strategy_auto"
            )
            
            # Unlock immediately after submission
            self._active_trades.pop(key, None)
            
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "order_id": broker_order_id,
                        "symbol": symbol,
                        "status": "SUBMITTED",
                        "price": price,
                        "side": side
                    },
                    source="order_processor"
                )
            )

        except Exception as exc:
            LOGGER.error(f"❌ Order Execution Failed: {exc}", exc_info=True)
            self._active_trades.pop(key, None)
            
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
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
