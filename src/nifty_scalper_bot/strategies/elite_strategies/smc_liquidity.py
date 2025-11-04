"""Smart Money Concepts liquidity sweep strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)


class SMCStrategy(EliteStrategy):
    """Detect potential liquidity grabs using volatility and volume clues."""

    def __init__(self, config: SMCStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="SMC Liquidity", config=config)
        self._smc_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return elite signal when price sweeps extremes with volume confirmation.

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
            "Entered SMCStrategy._evaluate_signal",
            extra={"event": "smc_evaluate", "symbol": symbol},
        )
        try:
            atr = float(indicators.get("atr") or 0.0)
            volume_ratio = float(indicators.get("volume_spike_ratio") or 0.0)
            lower_band = float(indicators.get("bollinger_lower") or 0.0)
            upper_band = float(indicators.get("bollinger_upper") or 0.0)
            vwap = float(indicators.get("vwap") or 0.0)
            if atr <= 0 or lower_band <= 0 or upper_band <= 0 or vwap <= 0:
                return None
            tolerance = atr * (self._smc_config.equal_level_tolerance_pct / 100.0)
            if volume_ratio < self._smc_config.volume_spike_mult:
                return None
            side = ""
            confidence = self._smc_config.min_confidence
            if current_price < (lower_band - tolerance):
                side = "BUY"
                confidence += min(15.0, volume_ratio * 5.0)
            elif current_price > (upper_band + tolerance):
                side = "SELL"
                confidence += min(15.0, volume_ratio * 5.0)
            if not side:
                return None
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None
            stop_buffer = self._smc_config.sweep_distance_points
            if side == "BUY":
                stop_loss = current_price - max(atr, stop_buffer)
                tp1 = current_price + atr * 1.5
                tp2 = current_price + atr * 2.5
            else:
                stop_loss = current_price + max(atr, stop_buffer)
                tp1 = current_price - atr * 1.5
                tp2 = current_price - atr * 2.5
            self._logger.info(
                "Condition met: smc_liquidity_signal",
                extra={
                    "event": "smc_liquidity_signal",
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
                    "volume_ratio": volume_ratio,
                    "vwap": vwap,
                    "band_extreme": lower_band if side == "BUY" else upper_band,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in SMCStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "smc_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["SMCStrategy"]
