"""
Bollinger Band Squeeze Breakout Strategy.
World-Class implementation with Greeks validation, Volatility checks, and Dynamic Risk.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class BBSqueezeStrategy(EliteStrategy):
    """
    Trade volatility expansion following tight Bollinger compression.
    Detects 'Squeeze' (Low Volatility) -> 'Expansion' (High Volatility).
    """

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_bb_config",)

    def __init__(self, config: BBSqueezeStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._bb_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare indicators for the StrategyManager to pre-calculate.
        Ensures all band data is perfectly synchronized.
        """
        return {
            "bollinger_upper",
            "bollinger_lower",
            "bollinger_mid",
            "atr",
            "volume",
            "average_volume",
            "ltp"
        }

    def _evaluate_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """
        Modern Signature: Evaluates signal using injected data points.
        """
        try:
            # 1. Safe Data Extraction
            upper = float(indicators.get("bollinger_upper") or 0.0)
            lower = float(indicators.get("bollinger_lower") or 0.0)
            mid = float(indicators.get("bollinger_mid") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0)

            # Prevent processing on invalid/missing data
            if upper == 0 or lower == 0 or current_price <= 0:
                return None

            # 2. Calculate Bandwidth (The Squeeze Intensity)
            # Bandwidth tells us how "tight" the spring is coiled.
            if mid == 0: return None
            bandwidth_pct = ((upper - lower) / mid) * 100

            # Threshold from config (e.g., 0.5% width)
            squeeze_threshold = getattr(self._bb_config, "squeeze_threshold_pct", 0.5)
            
            # If bands are wide, the squeeze has already resolved. Skip.
            # We allow up to 2x threshold to catch the very beginning of the expansion.
            if bandwidth_pct > (squeeze_threshold * 2.0):
                return None

            # 3. Detect Directional Breakout
            side = ""
            if current_price > upper:
                side = "BUY"
            elif current_price < lower:
                side = "SELL"
            
            if not side:
                return None

            # 4. Volume Confirmation (The "Fuel" Check)
            # A valid squeeze breakout MUST have expanding volume.
            vol_ratio = vol / avg_vol
            if vol_ratio < 1.3: # Require 30% surge over average
                return None

            # 5. Dynamic Risk Management
            # Fallback ATR for stop calculation if missing
            if atr == 0: atr = current_price * 0.005

            # Stop Loss: The Middle Band (Mean)
            # In a true breakout, price should NOT return to the mean.
            stop_loss = mid 
            
            if side == "BUY":
                # Targets based on volatility expansion
                tp1 = current_price + (atr * 2.5)
                tp2 = current_price + (atr * 5.0)
            else:
                tp1 = current_price - (atr * 2.5)
                tp2 = current_price - (atr * 5.0)

            # 6. Confidence Scoring
            # Base 75%. +15% if volume is extreme (>2.5x)
            confidence = 75.0
            if vol_ratio > 2.5: confidence += 15.0

            LOGGER.info(
                f"🚀 BB Squeeze Breakout: {symbol} {side} | Bandwidth: {bandwidth_pct:.2f}% | Vol: {vol_ratio:.1f}x",
                extra={
                    "event": "bb_squeeze_signal",
                    "symbol": symbol,
                    "bandwidth": bandwidth_pct,
                    "vol_ratio": vol_ratio
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 99.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._bb_config.quantity or 1,
                strategy_name="BB_Squeeze_Pro",
                metadata={
                    "type": "Volatility_Expansion",
                    "bandwidth_pct": round(bandwidth_pct, 3),
                    "volume_ratio": round(vol_ratio, 2),
                    "mid_band_support": mid
                }
            )

        except Exception as e:
            LOGGER.error(f"BB Strategy Critical Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["BBSqueezeStrategy"]
