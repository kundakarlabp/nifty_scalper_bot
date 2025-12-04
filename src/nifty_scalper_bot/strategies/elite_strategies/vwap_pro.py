"""VWAP Pro institutional-grade mean reversion strategy."""

from __future__ import annotations

from typing import Any, Mapping

# ✅ FIX: Inherit from EliteStrategy to ensure 'generate_signal' exists
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

class VWAPProStrategy(EliteStrategy):
    """
    VWAP Pro: An institutional-grade intraday strategy.
    
    Logic:
    1. Regime Filter: Trades in 'TRENDING' or 'VOLATILE' markets.
    2. Trend Filter: Longs only above EMA. Shorts only below EMA.
    3. Trigger: Price crosses/reverts to VWAP with volume confirmation.
    """

    def __init__(self, config: VWAPProStrategyConfig) -> None:
        """Initialise strategy with configuration."""
        # Pass config to parent EliteStrategy to handle basic setup
        super().__init__(name="VWAP Pro", config=config)
        self._vwap_config = config

    def get_required_indicators(self) -> list[str]:
        """Return indicator keys required for VWAP evaluation."""
        return [
            "vwap",
            "ema",
            "volume",
            "avg_volume",
            "atr",
            "rsi",
        ]

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Generate signal when price interacts with VWAP in direction of trend."""

        self._logger.debug(
            "Entered VWAPProStrategy._evaluate_signal",
            extra={"event": "vwap_pro_evaluate", "symbol": symbol},
        )
        try:
            # 1. Extract & Validate Indicators
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)
            rsi = float(indicators.get("rsi") or 50.0)

            if vwap <= 0 or ema <= 0 or atr <= 0:
                return None

            # 2. Volume Filter: Ignore moves with very low volume
            if avg_volume > 0 and volume < (avg_volume * 0.6):
                return None

            # 3. Determine Trend Bias (EMA Filter)
            trend_bias = "BULLISH" if current_price > ema else "BEARISH"

            # 4. Detect VWAP Interaction (Mean Reversion / Pullback)
            dist_to_vwap = current_price - vwap
            # Price must be within 0.5 ATR of VWAP to be considered a valid interaction
            is_near_vwap = abs(dist_to_vwap) < (atr * 0.5)

            side = ""
            if trend_bias == "BULLISH" and is_near_vwap:
                # Buy Condition: Uptrend + Pullback to VWAP + RSI not overbought
                if 40 <= rsi <= 65:
                    side = "BUY"
            
            elif trend_bias == "BEARISH" and is_near_vwap:
                # Sell Condition: Downtrend + Rally to VWAP + RSI not oversold
                if 35 <= rsi <= 60:
                    side = "SELL"

            if not side:
                return None

            # 5. Position Filter
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None

            # 6. Signal Construction
            vol_ratio = (volume / avg_volume) if avg_volume > 0 else 1.0
            confidence = self._vwap_config.min_confidence
            confidence += min(15.0, (vol_ratio - 1.0) * 10.0)
            
            stop_buffer = atr * 0.5 
            
            if side == "BUY":
                anchor = min(vwap, current_price)
                stop_loss = anchor - stop_buffer
                risk = current_price - stop_loss
                tp1 = current_price + (risk * 2.0)
                tp2 = current_price + (risk * 3.0)
            else:
                anchor = max(vwap, current_price)
                stop_loss = anchor + stop_buffer
                risk = stop_loss - current_price
                tp1 = current_price - (risk * 2.0)
                tp2 = current_price - (risk * 3.0)

            self._logger.info(
                "Condition met: vwap_pro_signal",
                extra={
                    "event": "vwap_pro_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "trend": trend_bias
                },
            )

            return EliteSignal(
                symbol=symbol,
                side=side,
                confidence=min(confidence, 100.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=1,
                strategy_name=self.name,
                metadata={
                    "vwap": vwap,
                    "ema": ema,
                    "atr": atr,
                    "volume_ratio": vol_ratio,
                    "trend_bias": trend_bias
                },
            )

        except Exception as exc:
            self._logger.error(
                "Failure in VWAPProStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "vwap_pro_evaluate_error", "symbol": symbol},
            )
            return None
