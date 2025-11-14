"""ATM straddle theta strategy."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    StraddleThetaStrategyConfig,
)


class StraddleThetaStrategy(EliteStrategy):
    """Deploy premium decay plays when volatility compresses post open."""

    def __init__(self, config: StraddleThetaStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="ATM Straddle Theta", config=config)
        self._straddle_config = config
        self._last_entry_date: dict[str, date] = {}

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return signal when expiry day theta setup aligns with volatility limits.

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
            "Entered StraddleThetaStrategy._evaluate_signal",
            extra={"event": "straddle_theta_evaluate", "symbol": symbol},
        )
        try:
            now = datetime.now(timezone.utc)
            if now.weekday() != 1:  # Tuesday expiry focus
                return None

            entry_minute = int(self._straddle_config.entry_time_minutes)
            current_minute = now.hour * 60 + now.minute
            if current_minute != entry_minute:
                return None

            last_entry = self._last_entry_date.get(symbol)
            if last_entry == now.date():
                return None

            vix_value = indicators.get("vix")
            vix = float(vix_value) if vix_value is not None else None
            if vix is None or vix > self._straddle_config.vix_threshold:
                return None

            bar_range = float(indicators.get("bar_range") or 0.0)
            if bar_range > 200.0:
                return None

            confidence = self._straddle_config.min_confidence
            stop_loss = current_price + self._straddle_config.max_spot_move_points
            tp1 = current_price
            tp2 = current_price

            self._last_entry_date[symbol] = now.date()
            self._logger.info(
                "Condition met: straddle_theta_signal",
                extra={
                    "event": "straddle_theta_signal",
                    "symbol": symbol,
                    "side": "SELL",
                    "confidence": confidence,
                    "vix": vix,
                },
            )
            return EliteSignal(
                symbol=symbol,
                side="SELL",
                confidence=min(confidence, 100.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=1,
                strategy_name=self.name,
                metadata={
                    "vix": vix,
                    "bar_range": bar_range,
                    "entry_minute": entry_minute,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StraddleThetaStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "straddle_theta_evaluate_error", "symbol": symbol},
            )
            return None

    def get_required_indicators(self) -> list[str]:
        """Return indicator requirements including VIX snapshot.

        Args:
            None.

        Returns:
            list[str]: Indicator names required for evaluation.

        Raises:
            None.
        """

        base_indicators = super().get_required_indicators()
        for required in ("vix", "bar_range"):
            if required not in base_indicators:
                base_indicators.append(required)
        return base_indicators


__all__ = ["StraddleThetaStrategy"]
