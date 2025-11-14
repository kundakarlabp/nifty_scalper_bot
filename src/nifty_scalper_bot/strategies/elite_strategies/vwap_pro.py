"""VWAP mean reversion pro strategy."""

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
    """Exploit VWAP deviations with RSI confirmation and volume filters."""

    def __init__(self, config: VWAPProStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="VWAP Mean Reversion Pro", config=config)
        self._vwap_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return signal when price stretches from VWAP with supportive RSI.

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
            "Entered VWAPProStrategy._evaluate_signal",
            extra={"event": "vwap_evaluate", "symbol": symbol},
        )
        try:
            vwap = float(indicators.get("vwap") or 0.0)
            rsi = float(indicators.get("rsi") or 50.0)
            atr = float(indicators.get("atr") or 0.0)
            volume_ratio = float(indicators.get("volume_spike_ratio") or 0.0)
            if vwap <= 0 or atr <= 0:
                return None
            deviation = current_price - vwap
            deviation_pct = abs(deviation) / vwap * 100 if vwap else 0.0
            if volume_ratio < self._vwap_config.volume_spike:
                return None
            side = ""
            confidence = self._vwap_config.min_confidence
            if (
                deviation < -atr * self._vwap_config.stddev_mult
                and rsi <= self._vwap_config.rsi_oversold
            ):
                side = "BUY"
                confidence += min(20.0, deviation_pct)
            elif (
                deviation > atr * self._vwap_config.stddev_mult
                and rsi >= self._vwap_config.rsi_overbought
            ):
                side = "SELL"
                confidence += min(20.0, deviation_pct)
            if not side:
                return None
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None
            tp_offset = atr * 1.2
            if side == "BUY":
                stop_loss = current_price - atr
                tp1 = current_price + tp_offset
                tp2 = vwap
            else:
                stop_loss = current_price + atr
                tp1 = current_price - tp_offset
                tp2 = vwap
            self._logger.info(
                "Condition met: vwap_pro_signal",
                extra={
                    "event": "vwap_pro_signal",
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
                    "deviation_pct": deviation_pct,
                    "rsi": rsi,
                    "volume_ratio": volume_ratio,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in VWAPProStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "vwap_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["VWAPProStrategy"]
