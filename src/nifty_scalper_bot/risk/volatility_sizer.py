"""Volatility-aware position sizing utilities."""

from __future__ import annotations

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class VolatilitySizer:
    """Adjust position size using relative ATR volatility."""

    def __init__(self, normal_atr: float = 0.015) -> None:
        """Initialise the sizer with a baseline ATR ratio.

        Args:
            normal_atr: Reference ATR percentage defining "normal" volatility.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered VolatilitySizer.__init__",
            extra={"event": "vol_sizer_init", "normal_atr": normal_atr},
        )
        try:
            self.normal_atr = max(float(normal_atr), 0.001)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in VolatilitySizer.__init__: %s",
                exc,
                extra={"event": "vol_sizer_init_error"},
                exc_info=exc,
            )
            self.normal_atr = 0.015

    def get_volatility_multiplier(self, atr_pct: float) -> float:
        """Return multiplier scaled inversely to volatility regime.

        Args:
            atr_pct: Current ATR expressed as a decimal percentage.

        Returns:
            float: Recommended size multiplier capped between 0.4 and 1.2.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered VolatilitySizer.get_volatility_multiplier",
            extra={"event": "vol_sizer_enter", "atr_pct": atr_pct},
        )
        try:
            atr_pct = max(float(atr_pct), 0.0)
            baseline = self.normal_atr or 0.015
            if baseline <= 0:
                baseline = 0.015
            vol_ratio = atr_pct / baseline

            if vol_ratio < 0.5:
                multiplier = 1.2
            elif vol_ratio < 0.8:
                multiplier = 1.1
            elif vol_ratio < 1.2:
                multiplier = 1.0
            elif vol_ratio < 1.5:
                multiplier = 0.8
            elif vol_ratio < 2.0:
                multiplier = 0.6
            else:
                multiplier = 0.4

            LOGGER.info(
                "Condition met: volatility_multiplier_resolved",
                extra={
                    "event": "volatility_multiplier_resolved",
                    "vol_ratio": round(vol_ratio, 4),
                    "multiplier": multiplier,
                },
            )
            return multiplier
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in VolatilitySizer.get_volatility_multiplier: %s",
                exc,
                extra={"event": "vol_sizer_error"},
                exc_info=exc,
            )
            return 0.4
