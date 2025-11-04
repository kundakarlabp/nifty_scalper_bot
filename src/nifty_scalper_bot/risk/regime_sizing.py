"""Regime-aware position sizing with multiplier caps."""

from __future__ import annotations

from enum import Enum

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.sampled_logging import RegimeSampler

LOGGER = get_logger(__name__)
_REGIME_SAMPLER = RegimeSampler(period_s=15.0)


class RegimeType(str, Enum):
    """Market regime classification labels."""

    TREND = "TREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"


class RegimeSizer:
    """Apply regime-based sizing multipliers with safety caps."""

    MAX_MULTIPLIER = 1.5
    MIN_MULTIPLIER = 0.5

    def __init__(
        self,
        trend_mult: float = 1.15,
        range_mult: float = 0.85,
        volatile_mult: float = 0.7,
        event_mult: float = 0.6,
    ) -> None:
        """Initialize a regime sizer with configured multipliers.

        Args:
            trend_mult: Multiplier applied in trending regimes.
            range_mult: Multiplier applied in ranging regimes.
            volatile_mult: Multiplier applied in volatile regimes.
            event_mult: Multiplier applied in event-driven regimes.

        Returns:
            None.

        Raises:
            None.
        """

        self._multipliers = {
            RegimeType.TREND: trend_mult,
            RegimeType.RANGE: range_mult,
            RegimeType.VOLATILE: volatile_mult,
            RegimeType.EVENT: event_mult,
            RegimeType.UNKNOWN: 1.0,
        }
        LOGGER.debug(
            "Entered RegimeSizer.__init__",
            extra={
                "event": "regime_sizer_init",
                "trend_mult": trend_mult,
                "range_mult": range_mult,
                "volatile_mult": volatile_mult,
                "event_mult": event_mult,
            },
        )

    def apply_regime_adjustment(
        self,
        base_size: float,
        regime: RegimeType,
        additional_mult: float = 1.0,
    ) -> float:
        """Return adjusted size after applying regime multipliers.

        Args:
            base_size: Base position size before regime adjustments.
            regime: Regime classification for the adjustment.
            additional_mult: Additional multiplier applied before clamping.

        Returns:
            float: Adjusted position size.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered RegimeSizer.apply_regime_adjustment",
            extra={
                "event": "regime_sizer_apply_enter",
                "base_size": base_size,
                "regime": regime.value,
                "additional_mult": additional_mult,
            },
        )
        regime_mult = self._multipliers.get(regime, 1.0)
        combined_mult = regime_mult * additional_mult
        capped_mult = max(self.MIN_MULTIPLIER, min(combined_mult, self.MAX_MULTIPLIER))
        adjusted_size = base_size * capped_mult
        if capped_mult != combined_mult:
            LOGGER.warning(
                "Condition met: regime_multiplier_capped",
                extra={
                    "event": "regime_multiplier_capped",
                    "regime": regime.value,
                    "combined_mult": combined_mult,
                    "capped_mult": capped_mult,
                },
            )

        try:
            _REGIME_SAMPLER.maybe_log(
                LOGGER,
                regime=regime.value,
                multiplier=capped_mult,
                extra={
                    "regime": regime.value,
                    "multiplier": capped_mult,
                    "combined_mult": combined_mult,
                },
                level="info",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in RegimeSizer.apply_regime_adjustment sampler: %s",
                exc,
                extra={"event": "regime_sampler_error", "regime": regime.value},
                exc_info=exc,
            )

        return adjusted_size


__all__ = ["RegimeSizer", "RegimeType"]
