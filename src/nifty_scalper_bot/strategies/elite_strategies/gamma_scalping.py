"""
Gamma Scalping Strategy.
World-Class implementation with Greeks Validation (Gamma/Theta Efficiency) and Momentum.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    GammaScalpingStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class GammaScalpingStrategy(EliteStrategy):
    """
    Trade directional gamma edges.
    Captures explosive moves where Gamma (Acceleration) justifies the Theta (Decay) cost.
    Entry: High Momentum + Positive Gamma Environment.
    """
    MIN_BARS_REQUIRED = 3

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_gamma_config",)

    def __init__(self, config: GammaScalpingStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._gamma_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "gamma", 
            "theta", 
            "delta", 
            "ltp", 
            "volume", 
            "average_volume",
            "macd",         # Momentum Trigger
            "macd_signal",  # Signal Line
            "atr"           # Volatility for stops
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
            symbol: Ticker symbol.
            indicators: Dictionary containing pre-fetched indicators.
            current_price: Latest LTP.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            gamma = float(indicators.get("gamma") or 0.0)
            theta = float(indicators.get("theta") or 0.0)
            delta = float(indicators.get("delta") or 0.0)
            
            macd = float(indicators.get("macd") or 0.0)
            signal_line = float(indicators.get("macd_signal") or 0.0)
            
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0)

            # Sanity Checks
            if current_price <= 0:
                return None

            # 2. Logic: Gamma Filter (Acceleration)
            # We are looking for "Long Gamma" setups (Buying explosive moves).
            # Gamma must be positive and significant enough to drive price.
            # (Note: Gamma is usually small, e.g., 0.001 to 0.05)
            min_gamma = getattr(self._gamma_config, "min_gamma", 0.0005)
            if gamma < min_gamma:
                return None

            # 3. Logic: Theta Efficiency (Cost of Time)
            # Don't buy if Theta is burning too hard relative to the move.
            # Theta is usually negative for long options.
            # If Theta < -10 (burning fast) AND Gamma is not super high, skip.
            if theta < -15.0 and gamma < 0.002:
                # Too expensive to hold this position
                return None

            # 4. Logic: Momentum Trigger (MACD Crossover)
            # This is a Scalper: We want to enter EXACTLY when momentum shifts.
            # Bullish Crossover: MACD crosses above Signal
            bullish_momentum = (macd > signal_line) and (macd - signal_line) > 0.5
            
            # Bearish Crossover (for Shorting Options/Futures, or exiting)
            # If trading Options Long, we generally only care about Bullish Momentum of the option price.
            # However, if this strategy manages FUTURES, we can short.
            # Assuming Option Buying for Gamma Scalping here.
            
            if not bullish_momentum:
                return None

            # 5. Logic: Volume Confirmation
            # Acceleration needs fuel.
            vol_ratio = vol / avg_vol
            if vol_ratio < 1.0: # At least average volume
                return None

            # 6. Construct Signal (Buy Scalp)
            side = "BUY"
            
            # Fallback ATR
            if atr == 0: atr = current_price * 0.01

            # Tight Scalp Targets
            # Stop Loss: Recent volatility (ATR)
            stop_loss = current_price - (atr * 1.0)
            
            # Take Profit: Gamma moves are fast. 
            # Target 2x ATR or a fixed Gamma spike
            tp1 = current_price + (atr * 1.5)
            tp2 = current_price + (atr * 3.0)

            # 7. Confidence Scoring
            # Base 65% (Scalping is noisy).
            confidence = 0.65
            
            # Boost if Gamma is high (Acceleration is likely)
            if gamma > 0.002: confidence += 0.15
            
            # Boost if Volume is Absorbing (>2x)
            if vol_ratio > 2.0: confidence += 0.10

            LOGGER.info(
                f"⚡ Gamma Scalp: {symbol} {side} | Gamma: {gamma:.4f} | Theta: {theta:.2f} | MACD Diff: {(macd-signal_line):.2f}",
                extra={
                    "event": "gamma_scalping_signal",
                    "symbol": symbol,
                    "gamma": gamma,
                    "theta": theta,
                    "vol_ratio": vol_ratio
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._gamma_config.quantity or 1,
                strategy_name="Gamma_Scalp_Pro",
                metadata={
                    "type": "Momentum_Acceleration",
                    "gamma_efficiency": f"{gamma:.4f}/{theta:.1f}",
                    "momentum": "MACD_Bullish",
                    "vol_ratio": round(vol_ratio, 2)
                }
            )

        except Exception as e:
            LOGGER.error(f"Gamma Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["GammaScalpingStrategy"]
