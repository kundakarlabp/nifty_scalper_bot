"""
Elite strategies configuration validation.
Production-Grade: Defensive attribute access and robust error shielding.
"""

from __future__ import annotations

from typing import Any
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

class EliteConfigError(ValueError):
    """Raised when elite strategy configuration is invalid."""

def _resolve_numeric(obj: object, *names: str) -> float | None:
    """Return numeric attribute value for the first matching name."""
    for name in names:
        if not hasattr(obj, name):
            continue
        value: Any = getattr(obj, name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise EliteConfigError(
                f"Invalid numeric value for {name!s}: {value!r}"
            )
    return None

def validate_elite_config(elite: EliteStrategiesSettings) -> None:
    """
    Validate elite strategies configuration.
    Defensive implementation to prevent AttributeError during boot.
    """
    # ✅ FIX 1: Defensive check for the 'enabled' attribute
    # Some older versions of the config might not have this field during build transitions
    is_enabled = getattr(elite, "enabled", True) 

    LOGGER.debug(
        'Entered validate_elite_config',
        extra={'event': 'elite_validator_enter', 'enabled': is_enabled},
    )

    if not is_enabled:
        LOGGER.info(
            'Elite strategies disabled, skipping validation',
            extra={'event': 'elite_disabled'},
        )
        return

    try:
        # ✅ FIX 2: Use getattr for all strategy lookups to prevent crashes
        oi_settings = getattr(elite, 'oi_max_pain', None)
        if oi_settings:
            bullish = _resolve_numeric(oi_settings, 'pcr_bullish', 'oi_pcr_bullish')
            bearish = _resolve_numeric(oi_settings, 'pcr_bearish', 'oi_pcr_bearish')
            if bullish is not None and bearish is not None and bullish <= bearish:
                raise EliteConfigError(
                    f'OI_PCR_BULLISH ({bullish}) must be > OI_PCR_BEARISH ({bearish})'
                )

        gamma_settings = getattr(elite, 'gamma_scalping', None)
        if gamma_settings:
            delta_threshold = _resolve_numeric(
                gamma_settings, 'delta_threshold', 'gamma_delta_threshold'
            )
            if delta_threshold is not None and not (0 < delta_threshold < 1):
                raise EliteConfigError(
                    f'GAMMA_DELTA_THRESHOLD ({delta_threshold}) must be between 0 and 1'
                )

        cpr_settings = getattr(elite, 'cpr', None)
        if cpr_settings:
            window_start = _resolve_numeric(cpr_settings, 'breakout_window_start')
            window_end = _resolve_numeric(cpr_settings, 'breakout_window_end')
            if window_start is not None and window_end is not None and window_end <= window_start:
                raise EliteConfigError('CPR Breakout Window: End must be > Start')

        # ✅ FIX 3: Safety check for max_concurrent_strategies
        max_concurrent = getattr(elite, 'max_concurrent_strategies', 3)
        if max_concurrent < 1:
             raise EliteConfigError(f'MAX_CONCURRENT_STRATEGIES ({max_concurrent}) must be at least 1')

        LOGGER.info(
            'Elite config validated successfully',
            extra={
                'event': 'elite_validated',
                'max_concurrent': max_concurrent,
            },
        )

    except EliteConfigError as e:
        LOGGER.error(f'Elite configuration validation failed: {e}')
        raise
    except Exception as exc:
        LOGGER.error(f'Unexpected error in elite validation: {exc}', exc_info=True)
        # We raise a ConfigError but allow the trace to show the root cause
        raise EliteConfigError(f'Elite validation unexpected error: {exc}')
