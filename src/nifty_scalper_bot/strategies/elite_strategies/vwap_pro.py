"""
VWAP Pro Strategy.
World-Class implementation with Trend Following Pullbacks and Greeks Validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class VWAPProStrategy(EliteStrategy):
    """
    Institutions trade at VWAP.
    We look for 'Trend Pullbacks': Strong Trend -> Retrace to VWAP -> Bounce.
    """

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature (removed 'name')
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Identify Trend (Price relative to EMA).
        2. Identify Trigger (Price touches/nears VWAP).
        3. Confirm Volume (Activity at value area).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._vwap_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "ltp", "vwap", "ema", "atr", 
            "volume", "avg_volume", "rsi",
            "minutes_since_open"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            vwap = float(indicators.get("vwap") or 0)
            ema = float(indicators.get("ema") or 0) # Usually EMA 20 or 50
            atr = float(indicators.get("atr") or 0)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            rsi = float(indicators.get("rsi") or 50)
            mins_open = float(indicators.get("minutes_since_open") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0 or vwap == 0:
            return None

        # 2. Time Filter
        # VWAP takes time to stabilize. Don't trade it in first 15 mins.
        if mins_open < 15:
            return None

        # 3. Setup Logic: Trend Pullback
        # We want to trade WITH the trend (defined by EMA)
        # Trigger is when price is close to VWAP (Value Area)
        
        # Distance to VWAP
        dist_to_vwap = abs(ltp - vwap)
        threshold = atr * 0.5 # Within 0.5 ATR of VWAP
        
        side: str | None = None
        
        # Bullish Setup:
        # 1. Trend is Up (VWAP > EMA is a common proxy, or Price generally > EMA)
        # 2. Price pulled back to near VWAP
        # 3. RSI is not overbought (room to go)
        if vwap > ema and dist_to_vwap < threshold and rsi < 60:
            # Check for bounce (LTP slightly above VWAP is safer than below)
            if ltp > vwap: 
                side = "BUY"
            
        # Bearish Setup:
        # 1. Trend is Down (VWAP < EMA)
        # 2. Price rallied to near VWAP
        # 3. RSI is not oversold
        elif vwap < ema and dist_to_vwap < threshold and rsi > 40:
            if ltp < vwap:
                side = "SELL" # BaseStrategy handles PE mapping

        if not side:
            return None

        # 4. Volume Confirmation
        # Institutional defense of VWAP requires volume.
        # Vol > 1.0x Avg
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.0:
            return None # No institutional interest

        # 5. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 6. Risk Management (ATR Based)
        # Stop Loss: A clear break of VWAP invalidates the thesis
        if atr == 0: atr = ltp * 0.01
        
        if side == "BUY":
            stop_loss = vwap - (atr * 0.5) # Tight stop below VWAP
            tp1 = ltp + (atr * 2.0)
            tp2 = ltp + (atr * 4.0)
        else:
            stop_loss = vwap + (atr * 0.5)
            tp1 = ltp - (atr * 2.0)
            tp2 = ltp - (atr * 4.0)

        # 7. Confidence Calculation
        # High confidence for trend-aligned VWAP bounces
        confidence = 0.85
        if vol_ratio > 1.5: confidence += 0.10 # Strong defense
        
        # 8. Construct Signal
        LOGGER.info(
            f"🚀 VWAP Pullback: {symbol} {side} | Trend: {'Bull' if vwap > ema else 'Bear'} | Vol: {vol_ratio:.1f}x",
            extra={
                "event": "vwap_pro_signal",
                "symbol": symbol,
                "vwap": vwap,
                "ema": ema
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
            quantity=self._vwap_config.quantity or 1,
            strategy_name="VWAP_Pro",
            metadata={
                "trend_filter": "Bullish" if vwap > ema else "Bearish",
                "dist_to_vwap": dist_to_vwap,
                "volume_ratio": vol_ratio,
                "atr": atr
            }
        )


__all__ = ["VWAPProStrategy"]
