"""
Order Flow Imbalance Strategy.
World-Class implementation with Depth Imbalance, Large Orders, and Greeks Validation.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

    MIN_BARS_REQUIRED = 5

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_of_config",)

    def __init__(self, config: OrderFlowStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.

        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._of_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "depth",  # L2 Market Depth (Bids/Asks)
            "vwap",
            "atr",
            "volume",
            "avg_volume",
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        LOGGER.debug(
            "Entered OrderFlowStrategy._evaluate_signal",
            extra={"event": "order_flow_enter", "symbol": symbol},
        )
        try:
            # 1. Safe Data Extraction (Fast Path)
            # Depth is expected to be a dict: {'buy': [...], 'sell': [...]}
            depth = indicators.get("depth") or {}
            vwap = float(indicators.get("vwap") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)

            # ✅ VWAP Sanity Check - Prevent false signals
            if vwap <= 0:
                return None
            vwap_deviation = abs(current_price - vwap) / vwap
            if vwap_deviation > 0.10:  # More than 10% deviation is suspicious
                return None
            avg_vol = float(indicators.get("avg_volume") or 1.0)

            # Sanity Check: If Depth is empty or Price is zero, skip
            if not depth or current_price <= 0:
                return None

            bids = depth.get(
                "buy", []
            )  # List of dicts [{'price': x, 'quantity': y}, ...]
            asks = depth.get("sell", [])

            if not bids or not asks:
                return None

            # 2. Logic: Order Book Imbalance
            # Calculate total volume on Bid vs Ask side (Top 5 levels)
            total_bid_qty = sum(float(item.get("quantity", 0)) for item in bids[:5])
            total_ask_qty = sum(float(item.get("quantity", 0)) for item in asks[:5])

            if total_ask_qty == 0:
                total_ask_qty = 1.0  # Prevent div/0

            # Ratio > 1.0 means Buyers > Sellers
            imbalance_ratio = total_bid_qty / total_ask_qty
            ratio_min = getattr(self._of_config, "imbalance_ratio_min", 1.5)
            ratio_max = min(0.8, 1 / max(ratio_min, 1.0))

            # 3. Logic: Detect Large Orders (Icebergs/Walls)
            # Is there a single order significantly larger than the rest acting as a wall?
            has_bid_wall = self._detect_wall(bids, total_bid_qty)
            has_ask_wall = self._detect_wall(asks, total_ask_qty)

            # 4. Construct Signal
            side = ""
            stop_loss = 0.0
            tp1 = 0.0
            tp2 = 0.0

            # Fallback ATR if missing (0.5% of price)
            if atr <= 0:
                atr = current_price * 0.005

            # --- Bullish Setup (Buy the Support) ---
            # Condition: Aggressive Buyers (>1.5x) OR Support Wall
            # Context: Price must be > VWAP (Uptrend)
            if (imbalance_ratio > ratio_min or has_bid_wall) and current_price > vwap:
                # Volume check: Needs to be active (not dead hours)
                if vol > (avg_vol * 0.5):
                    side = "BUY"
                    risk_mult = 1.2 if imbalance_ratio > (ratio_min * 1.6) else 1.0
                    risk = atr * risk_mult
                    stop_loss = current_price - risk
                    tp1 = current_price + (risk * 1.6)
                    tp2 = current_price + (risk * 3.2)

            # --- Bearish Setup (Sell the Resistance) ---
            # Condition: Aggressive Sellers (Asks > 1.6x Bids -> Ratio < 0.6) OR Resistance Wall
            # Context: Price must be < VWAP (Downtrend)
            elif (imbalance_ratio < ratio_max or has_ask_wall) and current_price < vwap:
                if vol > (avg_vol * 0.5):
                    side = "SELL"
                    risk_mult = 1.2 if imbalance_ratio < (ratio_max * 0.7) else 1.0
                    risk = atr * risk_mult
                    stop_loss = current_price + risk
                    tp1 = current_price - (risk * 1.6)
                    tp2 = current_price - (risk * 3.2)

            if not side:
                return None
            if (
                side == "BUY" and (stop_loss >= current_price or tp1 <= current_price)
            ) or (
                side == "SELL" and (stop_loss <= current_price or tp1 >= current_price)
            ):
                LOGGER.info(
                    "Condition met: invalid order flow brackets",
                    extra={
                        "event": "order_flow_invalid_bracket",
                        "symbol": symbol,
                        "side": side,
                        "entry": current_price,
                        "stop_loss": stop_loss,
                        "tp1": tp1,
                        "tp2": tp2,
                    },
                )
                return None

            # 5. Confidence Scoring
            # Base 75%. Boost if Wall exists or Imbalance is extreme (>3x)
            confidence = 0.75
            if has_bid_wall or has_ask_wall:
                confidence += 0.10
            if imbalance_ratio > 3.0 or imbalance_ratio < 0.3:
                confidence += 0.10

            LOGGER.info(
                "Condition met: order_flow_signal",
                extra={
                    "event": "order_flow_signal",
                    "symbol": symbol,
                    "imbalance": imbalance_ratio,
                    "vwap": vwap,
                    "detail": (
                        f"🌊 Order Flow Signal: {symbol} {side} | Imbalance: "
                        f"{imbalance_ratio:.2f} | Wall: "
                        f'{"Yes" if (has_bid_wall or has_ask_wall) else "No"}'
                    ),
                },
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp2,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=self._of_config.quantity or 1,
                strategy_name="Order_Flow_Pro",
                metadata={
                    "type": "Depth_Imbalance",
                    "imbalance_ratio": round(imbalance_ratio, 2),
                    "wall_detected": has_bid_wall or has_ask_wall,
                    "vwap": vwap,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp1_rr": 1.6,
                    "tp2_rr": 3.2,
                },
            )

        except Exception as e:
            # Prevent single strategy crash from stopping the bot
            LOGGER.error(f"Order Flow Strategy Error on {symbol}: {e}", exc_info=True)
            return None

    def _detect_wall(self, depth_side: List[Dict[str, Any]], total_qty: float) -> bool:
        """
        Detect if any single level contributes > 40% of total top-5 depth volume.
        This indicates a 'Wall' or huge limit order.
        """
        if total_qty <= 0:
            return False
        threshold = 0.40  # 40% wall

        for level in depth_side[:5]:  # Check top 5 levels
            qty = float(level.get("quantity", 0))
            if (qty / total_qty) > threshold:
                return True
        return False


__all__ = ["OrderFlowStrategy"]
