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

    MIN_BARS_REQUIRED = 25

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
            "bollinger_middle",
            "atr",
            "volume",
            "avg_volume",
            "ltp",
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
            "Entered BBSqueezeStrategy._evaluate_signal",
            extra={"event": "bb_squeeze_enter", "symbol": symbol},
        )
        try:
            # 1. Safe Data Extraction
            upper = float(indicators.get("bollinger_upper") or 0.0)
            lower = float(indicators.get("bollinger_lower") or 0.0)
            mid = float(indicators.get("bollinger_middle") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)

            # Prevent processing on invalid/missing data
            if upper == 0 or lower == 0 or current_price <= 0:
                return None

            # 2. Calculate Bandwidth (The Squeeze Intensity)
            # Bandwidth tells us how "tight" the spring is coiled.
            if mid == 0:
                return None
            bandwidth_pct = ((upper - lower) / mid) * 100

            # Threshold from config (e.g., 0.5% width)
            squeeze_threshold = getattr(self._bb_config, "squeeze_threshold_pct", 0.5)

            # If bands are wide, the squeeze has already resolved. Skip.
            # We allow up to 2x threshold to catch the very beginning of the expansion.
            if bandwidth_pct > (squeeze_threshold * 2.0):
                return None

            # 3. Detect Directional Breakout — BUY ONLY (long-only options buying)
            # Bearish breakout (price < lower band) cannot be traded via options buying
            # without margin; skip to prevent SELL signals conflicting with other strategies.
            side = ""
            if current_price > upper:
                side = "BUY"

            if not side:
                return None

            # 4. Volume Confirmation (The "Fuel" Check)
            # A valid squeeze breakout MUST have expanding volume.
            vol_ratio = vol / avg_vol
            if vol_ratio < 1.3:  # Require 30% surge over average
                return None

            # 5. Dynamic Risk Management
            # Fallback ATR for stop calculation if missing
            if atr <= 0:
                atr = current_price * 0.005

            # Stop Loss: The Middle Band (Mean)
            # In a true breakout, price should NOT return to the mean.
            stop_loss = mid if mid < current_price else current_price - (atr * 1.2)

            risk = abs(current_price - stop_loss)
            if risk <= 0:
                risk = atr * 1.2
                stop_loss = current_price - risk
            tp1 = current_price + (risk * 1.8)
            tp2 = current_price + (risk * 3.6)
            if stop_loss >= current_price or tp1 <= current_price:
                LOGGER.info(
                    "Condition met: invalid bb squeeze brackets",
                    extra={
                        "event": "bb_squeeze_invalid_bracket",
                        "symbol": symbol,
                        "side": side,
                        "entry": current_price,
                        "stop_loss": stop_loss,
                        "tp1": tp1,
                        "tp2": tp2,
                    },
                )
                return None

            # 6. Confidence Scoring
            # Base 75%. +15% if volume is extreme (>2.5x)
            confidence = 0.75
            if vol_ratio > 2.5:
                confidence += 0.15

            LOGGER.info(
                "Condition met: bb_squeeze_signal",
                extra={
                    "event": "bb_squeeze_signal",
                    "symbol": symbol,
                    "bandwidth": bandwidth_pct,
                    "vol_ratio": vol_ratio,
                    "detail": (
                        f"🚀 BB Squeeze Breakout: {symbol} {side} | Bandwidth: "
                        f"{bandwidth_pct:.2f}% | Vol: {vol_ratio:.1f}x"
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
                quantity=self._bb_config.quantity or 1,
                strategy_name="BB_Squeeze_Pro",
                metadata={
                    "type": "Volatility_Expansion",
                    "bandwidth_pct": round(bandwidth_pct, 3),
                    "volume_ratio": round(vol_ratio, 2),
                    "mid_band_support": mid,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp1_rr": 1.8,
                    "tp2_rr": 3.6,
                },
            )

        except Exception as e:
            LOGGER.error(f"BB Strategy Critical Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["BBSqueezeStrategy"]
