"""Configuration validation helpers for execution components."""

from __future__ import annotations

import os

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def validate_execution_config() -> list[str]:
    """Validate critical execution environment configuration.

    Args:
        None.

    Returns:
        List[str]: Collection of validation error messages.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered validate_execution_config",
        extra={"event": "config_validation_enter"},
    )
    errors: list[str] = []
    try:
        tp1_raw = (os.getenv("LIFECYCLE_TP1_R") or "").strip()
        if tp1_raw:
            try:
                if float(tp1_raw) <= 0:
                    errors.append("LIFECYCLE_TP1_R must be positive")
            except ValueError:
                errors.append("LIFECYCLE_TP1_R must be a number")

        tp2_trend_raw = (os.getenv("LIFECYCLE_TP2_R_TREND") or "").strip()
        if tp2_trend_raw:
            try:
                if float(tp2_trend_raw) <= 0:
                    errors.append("LIFECYCLE_TP2_R_TREND must be positive")
            except ValueError:
                errors.append("LIFECYCLE_TP2_R_TREND must be a number")

        mode = (os.getenv("EXECUTION_MODE") or "SHADOW").strip().upper() or "SHADOW"
        if mode not in {"LIVE", "SIMULATION", "SHADOW", "PAPER"}:
            errors.append(f"Invalid EXECUTION_MODE: {mode}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in validate_execution_config: %s",
            exc,
            extra={"event": "config_validation_error"},
            exc_info=exc,
        )
        errors.append("config_validation_internal_error")
    if errors:
        LOGGER.warning(
            "Configuration validation reported errors",
            extra={"event": "config_validation_failed", "errors": errors},
        )
    return errors


__all__ = ["validate_execution_config"]
