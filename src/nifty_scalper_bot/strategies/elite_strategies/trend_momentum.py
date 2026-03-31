"""
Trend Momentum Strategy.
For when price is far from VWAP - follow the trend!
"""

from __future__ import annotations
from typing import Any, Dict
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TrendMomentumStrategy(EliteStrategy):
    """
    Simple trend-following strategy for options.
    Triggers when price moves significantly in one direction.
    """
    MIN_BARS_REQUIRED = 5

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)

    def get_required_indicators(self) -> set[str]:
        return {"ema", "atr", "volume", "average_volume", "rsi"}

    def _evaluate_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """
        Generate signal based on trend momentum.
        """
        try:
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or current_price * 0.01)
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("average_volume") or 1.0)
            rsi = float(indicators.get("rsi") or 50.0)

            if ema <= 0 or current_price <= 0:
                return None

            # Calculate trend strength
            trend_pct = ((current_price - ema) / ema) * 100
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0

            # Signal conditions
            # BUY: Price >1% above EMA, RSI not overbought, good volume
            if trend_pct > 1.0 and rsi < 70 and vol_ratio > 0.8:
                # Don't chase if too extended
                if trend_pct > 5.0:
                    return None
                    
                stop_loss = current_price - (atr * 2.0)
                target = current_price + (atr * 3.0)
                
                LOGGER.info(
                    f"🚀 Trend Momentum BUY: {symbol} | Trend={trend_pct:.2f}% | RSI={rsi:.1f}",
                    extra={"event": "trend_momentum_signal", "symbol": symbol}
                )
                
                return EliteSignal(
                    symbol=symbol,
                    signal="BUY",
                    confidence=min(70 + (vol_ratio * 10), 90),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    target=target,
                    quantity=1,
                    strategy_name="Trend_Momentum",
                    metadata={
                        "trend_pct": round(trend_pct, 2),
                        "vol_ratio": round(vol_ratio, 2),
                        "rsi": round(rsi, 1)
                    }
                )

            # Bearish trend: cannot short options without margin. Skip.
            # (trend_pct < -1.0 would have triggered SELL — blocked in long-only mode)

            return None

        except Exception as e:
            LOGGER.error(f"Trend Momentum error for {symbol}: {e}")
            return None
