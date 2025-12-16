"""
Order Flow Imbalance Strategy.
World-Class implementation with Depth Imbalance, Large Orders, and Greeks Validation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class OrderFlowStrategy(EliteStrategy):
    """
    Trade based on L2 Market Depth (Level 2) Imbalances and Large Orders.
    """

    def __init__(self, config: OrderFlowStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._of_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Analyze Depth (Bid/Ask Imbalance).
        2. Detect Large Orders (Icebergs/Walls).
        3. Check VWAP Context.
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._of_config.symbol
        
        # 1. Fetch Indicators
        # We need Market Depth (L2), LTP, and Volume
        required_indicators = {
            "ltp", "market_depth", "vwap", 
            "volume", "avg_volume", "atr"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            depth = indicators.get("market_depth") or {}
            vwap = float(indicators.get("vwap") or 0)
            atr = float(indicators.get("atr") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0 or not depth:
            return None

        # 2. Order Flow Analysis
        # Calculate Imbalance Ratio from Top 5 levels
        # Imbalance = (Bid Qty - Ask Qty) / (Bid Qty + Ask Qty)
        
        bids = depth.get("bids", []) # List of [price, qty]
        asks = depth.get("asks", [])
        
        if not bids or not asks:
            return None
            
        # Sum top 5 quantities
        total_bid_qty = sum(q for _, q in bids[:5])
        total_ask_qty = sum(q for _, q in asks[:5])
        
        if (total_bid_qty + total_ask_qty) == 0:
            return None
            
        imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
        
        # Threshold: e.g., > 0.3 means Bids are 30% stronger than Asks
        threshold = 0.3 
        
        side: str | None = None
        
        # Bullish Imbalance + Price > VWAP
        if imbalance > threshold and ltp > vwap:
            side = "BUY"
            
        # Bearish Imbalance + Price < VWAP
        elif imbalance < -threshold and ltp < vwap:
            side = "SELL" # BaseStrategy handles PE mapping

        if not side:
            return None

        # 3. Large Order Confirmation
        # Check if there is a dominant wall supporting the move
        # e.g., for BUY, we want a huge Bid wall below LTP
        has_support = self._has_large_order(depth, side)
        if not has_support:
            return None # Imbalance might be fleeting

        # 4. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 5. Risk Management (Scalp Style)
        # Order flow edges are short-lived. Tight stops.
        if atr == 0: atr = ltp * 0.01
        
        stop_buffer = atr * 0.8
        
        if side == "BUY":
            stop_loss = ltp - stop_buffer
            tp1 = ltp + (stop_buffer * 1.5)
            tp2 = ltp + (stop_buffer * 3.0)
        else:
            stop_loss = ltp + stop_buffer
            tp1 = ltp - (stop_buffer * 1.5)
            tp2 = ltp - (stop_buffer * 3.0)

        # 6. Confidence Calculation
        confidence = 0.70
        if abs(imbalance) > 0.5: confidence += 0.15 # Massive imbalance
        
        # 7. Construct Signal
        LOGGER.info(
            f"🚀 Order Flow Signal: {symbol} {side} | Imbal: {imbalance:.2f} | Bids: {total_bid_qty} vs Asks: {total_ask_qty}",
            extra={
                "event": "order_flow_signal",
                "symbol": symbol,
                "imbalance": imbalance,
                "bid_qty": total_bid_qty,
                "ask_qty": total_ask_qty
            }
        )

        return EliteSignal(
            symbol=symbol,
            side=side,
            confidence=min(confidence, 0.99),
            entry_price=ltp,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            quantity=self._of_config.quantity or 1,
            strategy_name="Order_Flow_Pro",
            metadata={
                "imbalance": imbalance,
                "vwap": vwap
            }
        )

    def _has_large_order(self, depth: Mapping[str, Any], side: str) -> bool:
        """Check for single large orders (walls) on the supporting side."""
        try:
            # If Buying, we look for Large Bids (Support). If Selling, Large Asks (Resistance).
            book_key = "bids" if side == "BUY" else "asks"
            levels = depth.get(book_key, [])
            
            if not levels: return False
            
            # Calculate average size of top 5 orders
            avg_size = sum(q for _, q in levels[:5]) / len(levels[:5])
            
            # Is there any SINGLE order that is 3x the average?
            # Or matches the config threshold?
            threshold_pct = self._of_config.large_order_pct or 30.0 # e.g. 30% of total book
            total_vol = sum(q for _, q in levels[:5])
            
            for _, qty in levels[:5]:
                if (qty / total_vol) * 100 > threshold_pct:
                    return True
            return False
            
        except Exception:
            return False


__all__ = ["OrderFlowStrategy"]
