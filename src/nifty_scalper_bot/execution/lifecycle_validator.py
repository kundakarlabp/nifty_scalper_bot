"""Lifecycle configuration validation for TP/SL parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from nifty_scalper_bot.config.settings import Settings

LOGGER = get_logger(__name__)


class LifecycleConfigError(ValueError):
    """Raised when lifecycle configuration violates constraints."""


def validate_lifecycle_config(settings: "Settings") -> None:
    """Validate lifecycle configuration at startup.

    Args:
        settings: Runtime settings to validate.

    Returns:
        None.

    Raises:
        LifecycleConfigError: If validation fails.
    """

    LOGGER.debug(
        "Entered validate_lifecycle_config",
        extra={"event": "lifecycle_validator_enter"},
    )

    try:
        lifecycle = settings.orders.lifecycle

        if lifecycle.tp2_R_trend <= lifecycle.tp1_R:
            raise LifecycleConfigError(
                "LIFECYCLE_TP2_R_TREND must be greater than LIFECYCLE_TP1_R",
            )

        if lifecycle.tp2_R_range <= lifecycle.tp1_R:
            raise LifecycleConfigError(
                "LIFECYCLE_TP2_R_RANGE must be greater than LIFECYCLE_TP1_R",
            )

        if not (0 < lifecycle.tp1_partial < 1):
            raise LifecycleConfigError(
                "LIFECYCLE_TP1_PARTIAL must be between 0 and 1",
            )

        if lifecycle.time_stop_min <= 0:
            raise LifecycleConfigError(
                "LIFECYCLE_TIME_STOP_MIN must be positive",
            )

        if lifecycle.trail_atr_mult <= 0:
            raise LifecycleConfigError(
                "LIFECYCLE_TRAIL_ATR_MULT must be positive",
            )

        if getattr(lifecycle, "initial_stop_R", 1.0) <= 0:
            raise LifecycleConfigError(
                "LIFECYCLE_INITIAL_STOP_R must be positive",
            )

        LOGGER.info(
            "Lifecycle config validated",
            extra={
                "event": "lifecycle_validated",
                "tp1_R": lifecycle.tp1_R,
                "tp1_partial": lifecycle.tp1_partial,
                "tp2_R_trend": lifecycle.tp2_R_trend,
                "tp2_R_range": lifecycle.tp2_R_range,
                "trail_atr_mult": lifecycle.trail_atr_mult,
                "time_stop_min": lifecycle.time_stop_min,
                "initial_stop_R": getattr(lifecycle, "initial_stop_R", None),
            },
        )
    except LifecycleConfigError:
        LOGGER.error(
            "Lifecycle configuration validation failed",
            extra={"event": "lifecycle_validation_failed"},
        )
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Unexpected error in lifecycle validation: %s",
            exc,
            extra={"event": "lifecycle_validation_error"},
            exc_info=exc,
        )
        raise LifecycleConfigError("Lifecycle validation error") from exc


__all__ = ["LifecycleConfigError", "validate_lifecycle_config"]
