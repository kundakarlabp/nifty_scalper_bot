"""Gamma scalping hedge aware strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    GammaScalpingStrategyConfig,
)


class GammaScalpingStrategy(EliteStrategy):
    """Trade directional gamma edges with delta and theta proxies."""

    def __init__(self, config: GammaScalpingStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="Gamma Scalping", config=config)
        self._gamma_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return signal when implied volatility proxies favour gamma scalps.

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
            "Entered GammaScalpingStrategy._evaluate_signal",
            extra={"event": "gamma_evaluate", "symbol": symbol},
        )
        try:
            minutes_since_open = float(indicators.get("minutes_since_open") or 0.0)
            minutes_until_close = float(indicators.get("minutes_until_close") or 0.0)
            if (
                minutes_since_open < 60
                or minutes_until_close <= self._gamma_config.exit_time_minutes
            ):
                return None
            atr = float(indicators.get("atr") or 0.0)
            bar_range = float(indicators.get("bar_range") or 0.0)
            volume_ratio = float(indicators.get("volume_spike_ratio") or 0.0)
            macd = float(indicators.get("macd") or 0.0)
            if atr <= 0 or bar_range <= 0:
                return None
            if volume_ratio < 1.0:
                return None
            momentum_bias = macd / max(abs(macd), 1.0)
            side = (
                "BUY"
                if momentum_bias >= self._gamma_config.delta_threshold
                else (
                    "SELL"
                    if momentum_bias <= -self._gamma_config.delta_threshold
                    else ""
                )
            )
            if not side:
                return None
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None
            confidence = self._gamma_config.min_confidence
            confidence += min(15.0, volume_ratio * 3.0)
            stop_offset = max(atr, bar_range)
            if side == "BUY":
                stop_loss = current_price - stop_offset
                tp1 = current_price + self._gamma_config.hedge_trigger_points
                tp2 = current_price + self._gamma_config.hedge_trigger_points * 1.5
            else:
                stop_loss = current_price + stop_offset
                tp1 = current_price - self._gamma_config.hedge_trigger_points
                tp2 = current_price - self._gamma_config.hedge_trigger_points * 1.5
            self._logger.info(
                "Condition met: gamma_scalping_signal",
                extra={
                    "event": "gamma_scalping_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
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
                    "minutes_since_open": minutes_since_open,
                    "volume_ratio": volume_ratio,
                    "macd": macd,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in GammaScalpingStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "gamma_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["GammaScalpingStrategy"]
