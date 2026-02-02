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
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
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
        self._last_signal_time: Dict[str, float] = {}

        # 3. Validation: Ensure OrderManager supports brackets
        # This prevents silent naked trading if the underlying manager is outdated
        if not hasattr(self.executor, "place_bracket_order"):
            LOGGER.critical("FATAL: OrderManager does not support 'place_bracket_order'. Brackets will fail.")
            # We don't raise here to allow startup, but this is a critical configuration error.

    async def start(self) -> None:
        """Start the order processor message listener."""
        LOGGER.info("OrderProcessor: Starting...")
        self._running = True
        await self.bus.subscribe(MessageType.STRATEGY_SIGNAL, self.on_strategy_signal)
        # Ensure executor monitoring is active
        with suppress(Exception):
            self.executor.start_monitoring()

    async def stop(self) -> None:
        """Stop the order processor."""
        LOGGER.info("OrderProcessor: Stopping...")
        self._running = False
        with suppress(Exception):
            self.executor.stop_monitoring()

    async def on_strategy_signal(self, message: Message) -> None:
        """
        Handle incoming strategy signals with enforced Bracket Logic.
        """
        if not self._running:
            return

        signal: dict[str, Any] = message.data
        symbol = signal.get("symbol")
        side = signal.get("side")
        qty = signal.get("quantity")
        price = signal.get("price", 0.0) or 0.0
        
        # ✅ FIX 1: Extract Protection Data & Strategy Name
        stop_loss = float(signal.get("stop_loss") or 0.0)
        take_profit = float(signal.get("target") or signal.get("take_profit") or 0.0)
        strategy_name = str(signal.get("strategy_name") or "strategy_auto")

        if not symbol or not side or not qty:
            LOGGER.error(f"Invalid Signal: {signal}")
            return

        # 1. Concurrency Check (Debounce)
        key = f"{symbol}"
        if key in self._active_trades:
            LOGGER.warning(f"⚠️ Skipping Signal {symbol}: Active Order in Process")
            return

        self._active_trades[key] = INTENT

        try:
            # 2. Position Awareness (Exit vs Entry)
            position = self.pos_manager.get_position(symbol)
            is_exit = False
            
            if position and position.quantity != 0:
                # Simple logic: if side differs, it's an exit/reduction
                if (position.side == "LONG" and side == "SELL") or \
                   (position.side == "SHORT" and side == "BUY"):
                    is_exit = True
                    LOGGER.info(f"🔻 Signal Identified as EXIT for {symbol}")

            # 3. Risk Check (Skip for exits to allow closing)
            if not is_exit:
                risk_ok, risk_msg = self.risk_manager.check_trade_risk(
                    symbol=symbol, 
                    side=side, 
                    quantity=qty, 
                    price=price
                )
                if not risk_ok:
                    LOGGER.warning(f"⛔ Risk Reject {symbol}: {risk_msg}")
                    # Release lock immediately on rejection
                    self._active_trades.pop(key, None)
                    return

            # 4. Determine Order Type
            order_type = OrderType.MARKET
            if price > 0:
                order_type = OrderType.LIMIT

            LOGGER.info(f"🚀 Executing: {side} {qty} {symbol} @ {price or 'MKT'} (Is Exit: {is_exit})")

            # 5. Execution Logic with FORCED BRACKETS
            broker_order_id = None

            # ✅ FIX 2: FORCE BRACKET EXECUTION FOR ENTRIES
            # If it's an ENTRY and we have valid Stop/Target, use place_bracket_order
            if not is_exit and stop_loss > 0 and take_profit > 0:
                
                # Double check capability to avoid crash
                if hasattr(self.executor, "place_bracket_order"):
                    LOGGER.info(
                        f"🛡️ BRACKET ORDER SUBMITTED | {symbol} | SL={stop_loss} | TP={take_profit}"
                    )
                    
                    broker_order_id = await asyncio.to_thread(
                        self.executor.place_bracket_order,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        entry_price=price if order_type == OrderType.LIMIT else None,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_atr_mult=1.5,  # Conservative default for auto-trail
                        tag=strategy_name,
                    )
                else:
                    # Critical fallback if OrderManager is outdated, but logs the orphan risk
                    LOGGER.error("❌ OrderManager missing 'place_bracket_order'. Placing ORPHAN trade.")
                    broker_order_id = await asyncio.to_thread(
                        self.executor.place_order,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=order_type,
                        price=price,
                        tag=strategy_name
                    )
            else:
                # Standard Execution for Exits or Naked Entries (if missing SL/TP)
                if not is_exit:
                    LOGGER.warning(f"⚠️ Executing NAKED ENTRY for {symbol} (Missing SL/TP in signal)")
                
                broker_order_id = await asyncio.to_thread(
                    self.executor.place_order,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type=order_type,
                    price=price,
                    tag=strategy_name
                )
            
            # ✅ FIX 3: Update message with strategy source
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "order_id": broker_order_id,
                        "symbol": symbol,
                        "status": "SUBMITTED",
                        "price": price,
                        "side": side,
                        "strategy": strategy_name
                    },
                    source="order_processor"
                )
            )

        except Exception as exc:
            LOGGER.error(f"❌ Order Execution Failed: {exc}", exc_info=True)
            
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={"status": "REJECTED", "symbol": symbol, "error": str(exc)},
                    source="order_processor"
                )
            )
        finally:
            # ✅ FIX 3 (Refined): Release lock AFTER protection logic completes.
            # This ensures we don't accept a new signal until the bracket is effectively registered
            # (since place_bracket_order is synchronous-inside-thread).
            self._active_trades.pop(key, None)
