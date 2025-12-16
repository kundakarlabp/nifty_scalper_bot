"""
Gamma Scalping Strategy.
World-Class implementation with Greeks validation and Theta awareness.
"""

from __future__ import annotations

from typing import Any, Mapping

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
    Trade directional gamma edges with delta and theta proxies.
    Captures moves where Gamma (Acceleration) > Theta (Decay).
    """

    def __init__(self, config: GammaScalpingStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._gamma_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Momentum (MACD / RSI).
        2. Check Volume (Is market active?).
        3. Validate Physics (Theta vs Gamma).
        4. Execute Scalp.
        """
        symbol = self._gamma_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "ltp", "macd", "volume", "avg_volume", 
            "minutes_since_open", "rsi",
            "delta", "gamma", "theta" # Greeks are vital here
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            macd = float(indicators.get("macd") or 0)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            rsi = float(indicators.get("rsi") or 50)
            mins_open = float(indicators.get("minutes_since_open") or 0)
            
            # Greeks (If available, else default to neutral)
            gamma = float(indicators.get("gamma") or 0)
            theta = float(indicators.get("theta") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0:
            return None

        # 2. Time Window Check
        # Gamma scalping needs movement. Avoid lunch lull (11:30 - 13:00).
        # Best times: 09:30-11:00 and 14:00-15:00
        # (Simplified logic: just check if market has been open for > 15 mins)
        if mins_open < 15:
            return None

        # 3. Volume Check
        # Need liquidity to scalp. Vol > 1.2x Avg
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.2:
            return None 

        # 4. Directional Logic
        side: str | None = None
        
        # Long Gamma (Buy Option): Price moving away from strike
        # We use MACD as a proxy for momentum acceleration
        if macd > 0 and rsi > 55 and rsi < 75:
            side = "BUY"
        elif macd < 0 and rsi < 45 and rsi > 25:
            side = "SELL" # BaseStrategy handles PE mapping

        if not side:
            return None

        # 5. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 6. Gamma/Theta Efficiency Check
        # We want High Gamma, Low Theta. 
        # If Theta burn is > 20 pts/day, the move must be violent to profit.
        if abs(theta) > 20.0 and vol_ratio < 2.0:
            LOGGER.info(f"⛔ Rejected {symbol}: High Theta ({theta}) vs Low Vol ({vol_ratio})")
            return None

        # 7. Risk Management (Scalp Settings)
        # Tight stops, quick targets.
        trigger_pts = self._gamma_config.hedge_trigger_points or 10.0
        
        if side == "BUY":
            stop_loss = ltp - (trigger_pts * 0.8) # Tight stop
            tp1 = ltp + trigger_pts
            tp2 = ltp + (trigger_pts * 2.0)
        else:
            stop_loss = ltp + (trigger_pts * 0.8)
            tp1 = ltp - trigger_pts
            tp2 = ltp - (trigger_pts * 2.0)

        # 8. Confidence Calculation
        # Scalping is high freq, lower confidence per trade usually
        confidence = 0.65
        if vol_ratio > 2.0: confidence += 0.15
        if abs(gamma) > 0.05: confidence += 0.10 # High gamma bonus

        # 9. Construct Signal
        LOGGER.info(
            f"🚀 Gamma Scalp: {symbol} {side} | Vol: {vol_ratio:.1f}x | MACD: {macd:.2f}",
            extra={
                "event": "gamma_scalping_signal",
                "symbol": symbol,
                "gamma": gamma,
                "theta": theta
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
            quantity=self._gamma_config.quantity or 1,
            strategy_name="Gamma_Scalping_Pro",
            metadata={
                "volume_ratio": vol_ratio,
                "macd": macd,
                "gamma": gamma,
                "theta": theta
            }
        )


__all__ = ["GammaScalpingStrategy"]
