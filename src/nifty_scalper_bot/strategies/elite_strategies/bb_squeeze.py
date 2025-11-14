"""Bollinger Band squeeze breakout strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
)


class BBSqueezeStrategy(EliteStrategy):
    """Trade volatility expansion following tight Bollinger compression."""

    def __init__(self, config: BBSqueezeStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="BB Squeeze", config=config)
        self._bb_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return signal when compression breaks with volume and RSI support.

        Args:
            symbol: Trading symbol evaluated.
            indicators: Indicator snapshot for symbol.
            current_price: Latest traded price.
            position: Existing open position when present.

        Returns:
            EliteSignal | None: Signal when setup detected else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered BBSqueezeStrategy._evaluate_signal",
            extra={"event": "bb_squeeze_evaluate", "symbol": symbol},
        )
        try:
            upper = float(indicators.get("bollinger_upper") or 0.0)
            lower = float(indicators.get("bollinger_lower") or 0.0)
            middle = float(indicators.get("bollinger_middle") or 0.0)
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)
            rsi = float(indicators.get("rsi") or 50.0)
            if upper <= 0 or lower <= 0 or middle <= 0 or avg_volume <= 0:
                return None
            bandwidth = (upper - lower) / middle * 100.0
            if bandwidth > self._bb_config.bandwidth_threshold:
                return None

            volume_required = avg_volume * self._bb_config.volume_breakout
            if volume < volume_required:
                return None

            side = ""
            if current_price > upper and rsi > 55.0:
                side = "BUY"
            elif current_price < lower and rsi < 45.0:
                side = "SELL"
            if not side:
                return None

            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None

            compression_bonus = max(
                self._bb_config.bandwidth_threshold - bandwidth,
                0.0,
            )
            confidence = self._bb_config.min_confidence
            confidence += min(20.0, compression_bonus * 50.0)

            range_width = upper - lower
            if range_width <= 0:
                return None
            stop_loss = middle
            if side == "BUY":
                tp1 = current_price + range_width * 2.0
                tp2 = current_price + range_width * 3.0
            else:
                tp1 = current_price - range_width * 2.0
                tp2 = current_price - range_width * 3.0

            self._logger.info(
                "Condition met: bb_squeeze_signal",
                extra={
                    "event": "bb_squeeze_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "bandwidth": bandwidth,
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
                    "bandwidth": bandwidth,
                    "rsi": rsi,
                    "volume": volume,
                    "volume_required": volume_required,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in BBSqueezeStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "bb_squeeze_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["BBSqueezeStrategy"]
