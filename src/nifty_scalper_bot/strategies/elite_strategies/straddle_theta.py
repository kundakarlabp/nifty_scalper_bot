"""
Straddle/Strangle Theta Decay Strategy.
World-Class implementation with ADX Range Filtering, Greeks Validation, and Delta Guards.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    StraddleThetaStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class StraddleThetaStrategy(EliteStrategy):
    """
    Delta-Neutral-ish / Theta-Positive strategy.
    Shorts ATM/OTM options when market is range-bound (Low ADX) and IV is decent.
    """

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_theta_config",)

    def __init__(self, config: StraddleThetaStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._theta_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        Note: We request Greeks here. ADX (Market Regime) is fetched from underlying.
        """
        return {
            "theta", 
            "delta", 
            "iv", 
            "ltp", 
            "volume", 
            "open_interest"
        }

    def _evaluate_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """
        Modern Signature: Evaluates signal using injected data.
        
        Args:
            symbol: Option Symbol (e.g., NIFTY24JAN22000CE).
            indicators: Dictionary containing pre-fetched Greeks.
            current_price: Latest Premium.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            theta = float(indicators.get("theta") or 0.0)
            delta = abs(float(indicators.get("delta") or 0.0)) # Absolute delta
            iv = float(indicators.get("iv") or 0.0)
            oi = float(indicators.get("open_interest") or 0.0)
            
            # Sanity Check: If Theta is positive (impossible for long options, but rare data error)
            # or if price is effectively zero, skip.
            if current_price < 2.0 or theta >= 0:
                return None

            # 2. Market Regime Check (Range Bound?)
            # We need the Underlying's ADX. Since 'indicators' dict contains data for the
            # OPTION symbol, we must look up the underlying via the engine safely.
            # This is a "Hybrid Access" - Push for Option, Pull for Underlying.
            
            # Get Underlying ADX (Default to 50 to block trade if missing)
            # We assume the StrategyRunner or Engine creates a mapping for this.
            # If unavailable, we can skip or use a strict default.
            adx = 0.0
            try:
                # Assuming standard NIFTY/BANKNIFTY underlying
                # You might need logic to detect the specific underlying based on symbol name
                underlying = "NSE:NIFTY 50" if "NIFTY" in symbol else "NSE:BANKNIFTY"
                adx = self._indicator_engine.compute_adx(underlying, period=14) or 50.0
            except Exception:
                # Fail safe: if we can't confirm market is ranging, don't short.
                return None

            # CRITICAL FILTER: Only trade when Market is Choppy (ADX < 25)
            # High ADX means trending -> Shorting options is dangerous.
            adx_threshold = getattr(self._theta_config, "adx_threshold", 25.0)
            if adx > adx_threshold:
                return None

            # 3. IV Filter (Is Premium Rich?)
            # We want to sell when IV is relatively high (Standard deviation logic or absolute)
            # For simplicity in this robust version, we check absolute IV vs config
            min_iv = getattr(self._theta_config, "min_iv", 12.0)
            if iv < min_iv:
                return None

            # 4. Delta Guard (Don't short Deep ITM)
            # We want to short OTM/ATM options where Theta is highest and risk is managed.
            # Delta > 0.6 means it's getting deep ITM.
            if delta > 0.60:
                return None

            # 5. Theta Efficiency Check
            # We want significantly negative theta (high decay).
            # e.g., We want to earn at least 2 points of theta per day.
            if theta > -2.0: 
                return None

            # 6. Risk Management (Short Premium)
            # Stop Loss: 20% of Premium (Tight stop for shorting to avoid gamma spikes)
            # Take Profit: 50% of Premium (Capture the decay)
            stop_loss = current_price * 1.25 
            tp1 = current_price * 0.50 
            tp2 = current_price * 0.10 # Runner

            # 7. Confidence Scoring
            # Base 75%. Boost if Theta is massive or ADX is dead flat.
            confidence = 75.0
            if adx < 15.0: confidence += 10.0
            if theta < -5.0: confidence += 10.0

            # 8. Construct Signal
            LOGGER.info(
                f"⚡ Theta Setup: Sell {symbol} | ADX: {adx:.1f} | Theta: {theta:.2f} | Delta: {delta:.2f}",
                extra={
                    "event": "straddle_theta_signal",
                    "symbol": symbol,
                    "adx": adx,
                    "theta": theta,
                    "iv": iv
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal="SELL",  # Shorting the Option
                confidence=min(confidence, 99.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._theta_config.quantity or 1,
                strategy_name="Straddle_Theta_Pro",
                metadata={
                    "type": "Theta_Decay_Short",
                    "adx_regime": "Ranging",
                    "theta": round(theta, 2),
                    "iv": round(iv, 2),
                    "delta": round(delta, 2)
                }
            )

        except Exception as e:
            # Prevent single strategy crash from stopping the bot
            LOGGER.error(f"Theta Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["StraddleThetaStrategy"]
