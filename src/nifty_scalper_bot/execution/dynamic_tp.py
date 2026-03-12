"""Dynamic Take Profit Controller for maximizing trend capture."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

class DynamicTPController:
    """Adjusts Take Profit orders based on market momentum."""

    def __init__(
        self,
        *,
        tp_order_id: str,
        symbol: str,
        side: str,  # "BUY" or "SELL" (Exit Side)
        initial_price: float,
        get_indicators: Callable[[str], Dict[str, Any]],
        modify_order: Callable[[str, float], bool],
    ) -> None:
        self.tp_order_id = tp_order_id
        self.symbol = symbol
        self.side = side  # This is the side of the TP order (Opposite to Entry)
        self.current_tp_price = initial_price
        self.get_indicators = get_indicators
        self.modify_order = modify_order
        
        self._last_check_time = 0.0
        self._check_interval = 2.0  # Check every 2 seconds
        self._min_expansion = 5.0   # Minimum points to move TP
        
        LOGGER.debug(
            f"🚀 Dynamic TP Controller started for {symbol} (Order: {tp_order_id})",
            extra={"symbol": symbol, "initial_tp": initial_price}
        )

    def on_tick(self, tick: Dict[str, Any] | None = None) -> None:
        """Evaluate momentum and expand TP if conditions met."""
        now = time.time()
        if now - self._last_check_time < self._check_interval:
            return
        self._last_check_time = now

        # 1. Get Indicators (RSI, ADX, etc.)
        indicators = self.get_indicators(self.symbol)
        if not indicators:
            return

        rsi = indicators.get("rsi")
        adx = indicators.get("adx")
        
        if not isinstance(rsi, (int, float)): 
            return

        # 2. Evaluate Momentum
        # If we are exiting a LONG (Selling), we want strong UP momentum to expand TP
        should_expand = False
        expansion_amount = 0.0

        if self.side == "SELL":  # We are Long, TP is Sell
            # If RSI is Overbought (>70) AND ADX is Strong (>25), the trend is booming.
            # Don't cap profit! Move TP up.
            if rsi > 70 and (adx is None or adx > 25):
                should_expand = True
                expansion_amount = 10.0 # Move TP up by 10 points
                
        elif self.side == "BUY": # We are Short, TP is Buy
            # If RSI is Oversold (<30) AND ADX is Strong (>25), the crash is real.
            # Move TP down.
            if rsi < 30 and (adx is None or adx > 25):
                should_expand = True
                expansion_amount = 10.0 # Move TP down by 10 points

        # 3. Modify Order
        if should_expand:
            new_price = self._calculate_new_price(self.current_tp_price, expansion_amount)
            
            # Log the intent
            LOGGER.debug(
                f"🔥 Momentum Detected (RSI={rsi:.1f}). Expanding TP.",
                extra={
                    "symbol": self.symbol,
                    "old_tp": self.current_tp_price,
                    "new_tp": new_price
                }
            )
            
            success = self.modify_order(self.tp_order_id, new_price)
            if success:
                self.current_tp_price = new_price

    def _calculate_new_price(self, current: float, amount: float) -> float:
        if self.side == "SELL":
            return current + amount
        else:
            return max(0.05, current - amount)
