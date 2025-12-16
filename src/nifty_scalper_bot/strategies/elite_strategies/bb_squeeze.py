"""
Bollinger Band Squeeze Breakout Strategy.
World-Class implementation with Greeks validation, Volatility checks, and Dynamic Risk.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

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

    def __init__(self, config: BBSqueezeStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Do not pass 'name' to super().__init__
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._bb_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Bandwidth (Is it tight?).
        2. Check Breakout (Did price close outside?).
        3. Check Volume (Is there power behind the move?).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._bb_config.symbol
        
        # 1. Fetch Indicators
        # We need BB Bands, RSI, Volume, and Greeks
        required_indicators = {
            "bb_upper", "bb_lower", "bb_middle", 
            "rsi", "volume", "avg_volume", "ltp"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        # Safe extraction with type casting
        try:
            upper = float(indicators.get("bb_upper") or 0)
            lower = float(indicators.get("bb_lower") or 0)
            mid = float(indicators.get("bb_middle") or 0)
            rsi = float(indicators.get("rsi") or 50)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            ltp = float(indicators.get("ltp") or 0)
        except (ValueError, TypeError):
            return None # Data not ready

        if ltp == 0 or mid == 0:
            return None

        # 2. Calculate Squeeze Metrics
        # Bandwidth: How tight are the bands? (Narrow = Squeeze)
        bandwidth = ((upper - lower) / mid) * 100
        squeeze_threshold = self._bb_config.squeeze_threshold  # e.g., 2.0%

        # 3. Detect Breakout Condition
        side: str | None = None
        
        # Bullish Breakout: Price > Upper Band
        if ltp > upper:
            # RSI Filter: Ensure momentum but not extreme overbought (>80 is risky)
            if 50 < rsi < 80: 
                side = "BUY"
        
        # Bearish Breakout: Price < Lower Band
        elif ltp < lower:
            # RSI Filter: Ensure momentum but not extreme oversold (<20 is risky)
            if 20 < rsi < 50:
                side = "SELL" # Note: For Options, this usually translates to buying PE via BaseStrategy

        if not side:
            return None

        # 4. Volume Confirmation (Energy)
        # We want Volume to be at least X% of Average (e.g., 120%)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        min_vol_ratio = self._bb_config.min_volume_ratio or 1.2
        
        if vol_ratio < min_vol_ratio:
            return None # False breakout (Low volume)

        # 5. Squeeze Validation (Was it tight before?)
        # Ideally, we check if bandwidth WAS low recently. 
        # For this atomic check, we ensure bandwidth isn't blown out yet.
        # If bandwidth is massive (>5%), the move already happened.
        if bandwidth > (squeeze_threshold * 3):
            return None # Too late, expansion already huge

        # 6. 🛡️ SAFETY GATE (Physics Check)
        # This uses the BaseElite method we added earlier
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 7. Risk Management (Technical Stop)
        # For BB Squeeze, the invalidation point is usually the Middle Band (Basis)
        # TP is projected expansion (Width * 2)
        width = upper - lower
        
        if side == "BUY":
            stop_loss = mid # Reverting to mean kills the trend
            tp1 = ltp + (width * 1.5)
            tp2 = ltp + (width * 3.0)
        else:
            stop_loss = mid
            tp1 = ltp - (width * 1.5)
            tp2 = ltp - (width * 3.0)

        # 8. Confidence Calculation
        # Higher confidence if volume is huge and squeeze was tight
        confidence = 0.70 
        if vol_ratio > 2.0: confidence += 0.10
        if bandwidth < squeeze_threshold: confidence += 0.10
        
        confidence = min(confidence, 0.99)

        # 9. Construct Signal
        LOGGER.info(
            f"🚀 BB Squeeze Signal: {symbol} {side} | Bandwidth: {bandwidth:.2f}% | Vol: {vol_ratio:.1f}x",
            extra={
                "event": "bb_squeeze_signal",
                "symbol": symbol,
                "bandwidth": bandwidth,
                "rsi": rsi
            }
        )

        return EliteSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            entry_price=ltp,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            quantity=self._bb_config.quantity or 1,
            strategy_name="BB_Squeeze_Pro",
            metadata={
                "bandwidth": bandwidth,
                "vol_ratio": vol_ratio,
                "rsi": rsi,
                "squeeze_threshold": squeeze_threshold
            }
        )


__all__ = ["BBSqueezeStrategy"]
