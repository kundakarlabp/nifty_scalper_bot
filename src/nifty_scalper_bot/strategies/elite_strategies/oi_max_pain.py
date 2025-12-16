"""
OI Max Pain / VWAP Reversion Strategy.
World-Class implementation with Greeks validation and Mean Reversion Safety.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OIMaxPainStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class OIMaxPainStrategy(EliteStrategy):
    """
    Approximate Max Pain / Mean Reversion logic.
    Trades when price extends significantly from VWAP and shows momentum divergence.
    """

    def __init__(self, config: OIMaxPainStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature (removed 'name')
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._oi_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Extension (Distance from VWAP).
        2. Check Momentum (MACD Histogram reversal).
        3. Check Time (Avoid last 30 mins).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._oi_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "ltp", "vwap", "macd_hist", 
            "sma_200", "atr", "minutes_until_close",
            "volume", "avg_volume"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            vwap = float(indicators.get("vwap") or 0)
            macd_hist = float(indicators.get("macd_hist") or 0)
            sma = float(indicators.get("sma_200") or 0)
            atr = float(indicators.get("atr") or 0)
            mins_left = float(indicators.get("minutes_until_close") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0 or vwap == 0:
            return None

        # 2. Time Filter
        # Don't take mean reversion trades in the last 30 mins (Gamma risk)
        if mins_left < 30:
            return None

        # 3. Extension Logic (Deviation from VWAP)
        # Calculate % distance
        dist_pct = ((ltp - vwap) / vwap) * 100
        threshold = self._oi_config.deviation_threshold or 0.5 # e.g. 0.5% deviation
        
        side: str | None = None
        
        # Case A: Price is deeply below VWAP (Oversold) -> Expect bounce up
        # Confirmation: MACD Histogram turning positive (Momentum shift)
        if dist_pct < -threshold and macd_hist > 0:
            side = "BUY"
            
        # Case B: Price is deeply above VWAP (Overbought) -> Expect drop
        # Confirmation: MACD Histogram turning negative
        elif dist_pct > threshold and macd_hist < 0:
            side = "SELL" # BaseStrategy handles PE mapping

        if not side:
            return None

        # 4. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 5. Risk Management
        # Stop Loss: 1.5 ATR against us
        # Take Profit: Return to VWAP (Mean Reversion) or SMA
        if atr == 0: atr = ltp * 0.01
        
        stop_offset = atr * 1.5
        
        if side == "BUY":
            stop_loss = ltp - stop_offset
            tp1 = vwap # Primary target is always VWAP
            tp2 = sma if sma > ltp else (ltp + stop_offset * 3)
        else:
            stop_loss = ltp + stop_offset
            tp1 = vwap
            tp2 = sma if sma < ltp else (ltp - stop_offset * 3)

        # 6. Confidence Calculation
        # Higher confidence if we are reverting TOWARDS the 200 SMA
        confidence = 0.70
        if side == "BUY" and ltp < sma: confidence += 0.10 # Trend alignment
        if side == "SELL" and ltp > sma: confidence += 0.10
        
        # Boost if deviation is extreme (>1.0%)
        if abs(dist_pct) > 1.0: confidence += 0.10

        # 7. Construct Signal
        LOGGER.info(
            f"🚀 OI Max Pain / VWAP Revert: {symbol} {side} | Dist: {dist_pct:.2f}%",
            extra={
                "event": "oi_max_pain_signal",
                "symbol": symbol,
                "vwap": vwap,
                "macd_hist": macd_hist
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
            quantity=self._oi_config.quantity or 1,
            strategy_name="OI_Max_Pain_Pro",
            metadata={
                "vwap_dist": dist_pct,
                "macd_hist": macd_hist,
                "atr": atr
            }
        )


__all__ = ["OIMaxPainStrategy"]
