"""
Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle.
Production-Grade: Handles Risk Checks, Thread Safety, and Non-Blocking Execution.
"""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.risk.risk_manager import RiskManager

# --- Order Execution States ---
INTENT = "INTENT"   # Signal accepted, execution pending

LOGGER = logging.getLogger(__name__)

class OrderProcessor:
    """
    Orchestrates order execution with strict Risk Management and Concurrency Control.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        safe_order_manager: OrderManager,
        risk_manager: RiskManager,
        data_hub: Any, 
    ):
        self.bus = message_bus
        self.executor = safe_order_manager
        self.risk_manager = risk_manager
        self.data_hub = data_hub
        self._running = False
        
        # 1. Thread Safety Lock (Mandatory for async execution)
        self._lock = asyncio.Lock()
        
        # 2. State Tracking
        # Key = Symbol (NOT symbol+side). Prevents simultaneous Buy/Sell wars.
        self._active_trades: Dict[str, str] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        
        # 3. Settings
        self._debounce_seconds = 60.0

        self.bus.subscribe(MessageType.SIGNAL, self.on_strategy_signal)
        LOGGER.info("✅ OrderProcessor initialized with Risk-Gated Execution.")

    async def on_strategy_signal(self, message: Message) -> None:
        """
        Process signal -> Check Risk -> Execute Order (Non-Blocking).
        """
        signal: dict[str, Any] = message.data
        symbol = signal.get("symbol")
        side = signal.get("side")      # "BUY" / "SELL"
        qty = signal.get("quantity")
        
        # Basic Validation
        if not all([symbol, side, qty]):
            return

        # --- PHASE 1: ATOMIC CHECKS (Must be fast) ---
        async with self._lock:
            # 1. Conflict Prevention: One operation per symbol at a time
            key = symbol 
            if self._active_trades.get(key):
                LOGGER.warning(f"🚫 Execution busy for {symbol}")
                return

            # 2. Debounce: Prevent double-tap signals
            now = datetime.now(timezone.utc)
            last_time = self._last_signal_time.get(key)
            if last_time and (now - last_time).total_seconds() < self._debounce_seconds:
                LOGGER.info(f"⏳ Cooldown active: {symbol}")
                return

            # 3. Risk Management Gate (The Critical Safety Check)
            if self.risk_manager:
                # can_trade returns (allowed, reason)
                allowed, reason = self.risk_manager.can_trade(
                    symbol=symbol, 
                    side=side, 
                    quantity=qty
                )
                if not allowed:
                    LOGGER.error(f"🛡️ Risk Rejection for {symbol}: {reason}")
                    return

            # 4. Lock Resources
            self._last_signal_time[key] = now
            self._active_trades[key] = INTENT

        # --- PHASE 2: EXECUTION (Can be slow/blocking) ---
        
        # Price Logic (Simple Slippage Protection)
        order_type = signal.get("order_type", "MARKET")
        price = signal.get("price", 0.0)
        
        # If Limit order requested but no price, get LTP from DataHub
        if order_type == "LIMIT" and price == 0.0 and self.data_hub:
            tick = self.data_hub.get_quote(symbol)
            if tick and tick.get("ltp"):
                ltp = float(tick["ltp"])
                # Apply buffer (Buyer pays more, Seller asks less to ensure fill)
                # 1% buffer helps in fast Nifty moves
                buffer = 1.01 if side == "BUY" else 0.99
                price = round((ltp * buffer) / 0.05) * 0.05

        LOGGER.info(f"🚀 Executing: {side} {qty} {symbol} @ {price or 'MKT'}")

        try:
            # 5. Non-Blocking Execution
            # place_order is blocking, so we await it in a thread to keep the bot responsive.
            broker_order_id = await asyncio.to_thread(
                self.executor.place_order,
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=order_type,
                price=price,
                tag="strategy_auto"
            )
            
            # Release lock immediately on success.
            # (Position tracking is handled by PositionManager, not here)
            self._active_trades.pop(key, None)
            
            # 6. Publish Success
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=now,
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
            
            # Release lock on failure so we can try again later
            self._active_trades.pop(key, None)
            
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=now,
                    data={
                        "status": "REJECTED", 
                        "symbol": symbol, 
                        "error": str(exc)
                    },
                    source="order_processor"
                )
            )

    def start(self) -> None:
        """Start monitoring loop (delegated to executor)."""
        with suppress(Exception):
            self.executor.start_monitoring()
        self._running = True
        
    async def stop(self) -> None:
        """Stop processing."""
        self._running = False
        with suppress(Exception):
            self.executor.stop_monitoring()
