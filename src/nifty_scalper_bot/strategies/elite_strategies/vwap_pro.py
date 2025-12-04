"""VWAP Pro institutional-grade mean reversion strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)


class VWAPProStrategy(EliteStrategy):
    """
    VWAP Pro: An institutional-grade intraday strategy.
    
    Logic:
    1. Regime Filter: Only trades in 'TRENDING' or 'VOLATILE' markets (handled by manager).
    2. Trend Filter: Longs only above EMA. Shorts only below EMA.
    3. Trigger: Price crosses/reverts to VWAP with volume confirmation.
    """

    def __init__(self, config: VWAPProStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.
        """
        super().__init__(name="VWAP Pro", config=config)
        self._vwap_config = config
        self._last_state: dict[str, str] = {}  # Track state for crossover detection

    def get_required_indicators(self) -> list[str]:
        """Return indicator keys required for VWAP evaluation.
        
        We request 'ema' (standard 20-period) as a trend proxy.
        """
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
        """Generate signal when price interacts with VWAP in direction of trend.

        Args:
            symbol: Trading symbol evaluated.
            indicators: Indicator snapshot for symbol.
            current_price: Latest traded price.
            position: Existing open position when present.

        Returns:
            EliteSignal | None: Signal when setup detected else ``None``.
        """

        self._logger.debug(
            "Entered VWAPProStrategy._evaluate_signal",
            extra={"event": "vwap_pro_evaluate", "symbol": symbol},
        )
        try:
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)
            rsi = float(indicators.get("rsi") or 50.0)

            # Data validation
            if vwap <= 0 or ema <= 0 or atr <= 0:
                return None

            # 1. Volume Filter
            # We need volume to be somewhat relevant (e.g., > 80% of avg) to avoid ghost moves
            if avg_volume > 0 and volume < (avg_volume * 0.8):
                return None

            # 2. Determine Trend Bias using EMA
            # Price > EMA => Bullish Bias
            # Price < EMA => Bearish Bias
            trend_bias = "BULLISH" if current_price > ema else "BEARISH"

            # 3. Detect VWAP Interaction
            # We look for price being close to VWAP (Mean Reversion Entry)
            dist_to_vwap = current_price - vwap
            dist_ratio = abs(dist_to_vwap) / atr

            # Threshold: Price must be within 0.5 ATR of VWAP to consider it a "test" or "cross"
            # If it's too far, we missed the move.
            is_near_vwap = dist_ratio < 0.5

            side = ""
            if trend_bias == "BULLISH" and is_near_vwap:
                # Long Condition: Uptrend + Pullback to VWAP or Crossing Up
                # RSI check to ensure momentum isn't dead but not overbought
                if 40 <= rsi <= 65:
                    side = "BUY"
            
            elif trend_bias == "BEARISH" and is_near_vwap:
                # Short Condition: Downtrend + Rally to VWAP or Crossing Down
                if 35 <= rsi <= 60:
                    side = "SELL"

            if not side:
                return None

            # 4. Filter out if we already have a position in this direction
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None

            # 5. Calculate Confidence & Targets
            # Higher volume spike = Higher confidence
            vol_ratio = (volume / avg_volume) if avg_volume > 0 else 1.0
            
            confidence = self._vwap_config.min_confidence
            confidence += min(15.0, (vol_ratio - 1.0) * 10.0) # Bonus for volume
            
            # Risk Management
            # Stop Loss: On the other side of VWAP + buffer
            # Target: Trend continuation
            stop_buffer = atr * 0.5 
            
            if side == "BUY":
                stop_loss = vwap - stop_buffer
                # If price is already below VWAP (failed break), enter cautiously or use tighter stop
                if current_price < vwap:
                     stop_loss = current_price - stop_buffer
                
                risk = current_price - stop_loss
                tp1 = current_price + (risk * 2.0)
                tp2 = current_price + (risk * 3.0)
            else:
                stop_loss = vwap + stop_buffer
                if current_price > vwap:
                    stop_loss = current_price + stop_buffer

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
                    "dist_to_vwap": dist_to_vwap,
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

        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in VWAPProStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "vwap_pro_evaluate_error", "symbol": symbol},
            )
            return None
