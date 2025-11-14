"""OI max pain inspired directional strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OIMaxPainStrategyConfig,
)


class OIMaxPainStrategy(EliteStrategy):
    """Approximate OI max pain using momentum proxies and proximity to VWAP."""

    def __init__(self, config: OIMaxPainStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="OI Max Pain", config=config)
        self._oi_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Generate signal when price drifts far from VWAP with MACD confirmation.

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
            "Entered OIMaxPainStrategy._evaluate_signal",
            extra={"event": "oi_evaluate", "symbol": symbol},
        )
        try:
            minutes_until_close = float(indicators.get("minutes_until_close") or 0.0)
            if minutes_until_close <= self._oi_config.exit_time_minutes:
                return None
            vwap = float(indicators.get("vwap") or 0.0)
            sma = float(indicators.get("sma") or 0.0)
            macd_hist = float(indicators.get("macd_histogram") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            if vwap <= 0 or atr <= 0 or sma <= 0:
                return None
            distance = current_price - vwap
            if abs(distance) < self._oi_config.min_distance_points:
                return None
            side = (
                "BUY"
                if distance < 0 and macd_hist > 0
                else "SELL" if distance > 0 and macd_hist < 0 else ""
            )
            if not side:
                return None
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None
            confidence = self._oi_config.min_confidence
            confidence += min(20.0, abs(distance) / max(atr, 1.0) * 5.0)
            stop_offset = atr * 1.2
            if side == "BUY":
                stop_loss = current_price - stop_offset
                tp1 = current_price + atr * 1.5
                tp2 = sma
            else:
                stop_loss = current_price + stop_offset
                tp1 = current_price - atr * 1.5
                tp2 = sma
            self._logger.info(
                "Condition met: oi_max_pain_signal",
                extra={
                    "event": "oi_max_pain_signal",
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
                    "distance": distance,
                    "macd_hist": macd_hist,
                    "minutes_until_close": minutes_until_close,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OIMaxPainStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "oi_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["OIMaxPainStrategy"]
