"""Elite strategies configuration validation."""

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
    """Return numeric attribute value for the first matching name.

    Args:
        obj: Object providing configuration attributes.
        *names: Candidate attribute names to probe.

    Returns:
        float | None: Converted numeric value or ``None`` when absent.

    Raises:
        EliteConfigError: When the attribute exists but cannot be coerced.
    """

    for name in names:
        if not hasattr(obj, name):
            continue
        value: Any = getattr(obj, name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise EliteConfigError(
                f"Invalid numeric value for {name!s}: {value!r}"
            ) from exc
    return None


def validate_elite_config(elite: EliteStrategiesSettings) -> None:
    """Validate elite strategies configuration.

    Args:
        elite: Elite strategies settings.

    Returns:
        None.

    Raises:
        EliteConfigError: If validation fails.
    """

    LOGGER.debug(
        'Entered validate_elite_config',
        extra={'event': 'elite_validator_enter', 'enabled': elite.enabled},
    )

    if not elite.enabled:
        LOGGER.info(
            'Elite strategies disabled, skipping validation',
            extra={'event': 'elite_disabled'},
        )
        return

    try:
        oi_settings = elite.oi_max_pain
        bullish = _resolve_numeric(oi_settings, 'pcr_bullish', 'oi_pcr_bullish')
        bearish = _resolve_numeric(oi_settings, 'pcr_bearish', 'oi_pcr_bearish')
        if bullish is not None and bearish is not None and bullish <= bearish:
            raise EliteConfigError(
                f'OI_PCR_BULLISH ({bullish}) must be > OI_PCR_BEARISH ({bearish})'
            )

        gamma_settings = elite.gamma
        delta_threshold = _resolve_numeric(
            gamma_settings, 'delta_threshold', 'gamma_delta_threshold'
        )
        if delta_threshold is not None and not (0 < delta_threshold < 1):
            raise EliteConfigError(
                f'GAMMA_DELTA_THRESHOLD ({delta_threshold}) must be between 0 and 1'
            )

        cpr_settings = elite.cpr
        window_start = _resolve_numeric(
            cpr_settings, 'breakout_window_start', 'cpr_breakout_window_start'
        )
        window_end = _resolve_numeric(
            cpr_settings, 'breakout_window_end', 'cpr_breakout_window_end'
        )
        if (
            window_start is not None
            and window_end is not None
            and window_end <= window_start
        ):
            raise EliteConfigError(
                'CPR_BREAKOUT_WINDOW_END must be greater than CPR_BREAKOUT_WINDOW_START'
            )

        smc_settings = elite.smc
        tolerance = _resolve_numeric(
            smc_settings,
            'equal_level_tolerance_pct',
            'smc_equal_level_tolerance_pct',
        )
        if tolerance is not None and tolerance > 5.0:
            LOGGER.warning(
                'SMC tolerance high',
                extra={
                    'event': 'smc_tolerance_high',
                    'smc_equal_level_tolerance_pct': float(tolerance),
                    'recommendation': 'Consider reducing to < 5% for tighter levels',
                },
            )

        order_flow_settings = elite.order_flow
        imbalance_sell = _resolve_numeric(
            order_flow_settings,
            'imbalance_ratio_sell',
            'order_flow_imbalance_ratio_sell',
        )
        imbalance_buy = _resolve_numeric(
            order_flow_settings,
            'imbalance_ratio_buy',
            'order_flow_imbalance_ratio_buy',
        )
        if imbalance_sell is not None and imbalance_sell >= 1.0:
            raise EliteConfigError(
                'ORDER_FLOW_IMBALANCE_RATIO_SELL must be < 1.0'
            )
        if imbalance_buy is not None and imbalance_buy <= 1.0:
            raise EliteConfigError(
                'ORDER_FLOW_IMBALANCE_RATIO_BUY must be > 1.0'
            )

        if elite.max_concurrent_strategies < 1:
            raise EliteConfigError(
                (
                    'MAX_CONCURRENT_STRATEGIES '
                    f'({elite.max_concurrent_strategies}) must be at least 1'
                )
            )

        pcr_range = None
        if bullish is not None and bearish is not None:
            pcr_range = f'{bearish}-{bullish}'

        cpr_window = None
        if window_start is not None and window_end is not None:
            cpr_window = f'{window_start}-{window_end}'

        LOGGER.info(
            'Elite config validated',
            extra={
                'event': 'elite_validated',
                'enabled_strategies': elite.max_concurrent_strategies,
                'oi_pcr_range': pcr_range,
                'gamma_delta_threshold': delta_threshold,
                'cpr_window': cpr_window,
            },
        )

    except EliteConfigError:
        LOGGER.error(
            'Elite configuration validation failed',
            extra={'event': 'elite_validation_failed'},
        )
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            'Unexpected error in elite validation: %s',
            exc,
            extra={'event': 'elite_validation_error'},
            exc_info=exc,
        )
        raise EliteConfigError(f'Elite validation error: {exc}') from exc


__all__ = ['EliteConfigError', 'validate_elite_config']
