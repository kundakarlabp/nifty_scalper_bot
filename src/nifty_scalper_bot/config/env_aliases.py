"""Environment variable alias resolution helpers."""

from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

ENV_ALIASES: dict[str, tuple[str, ...]] = {
    'ENABLE_LIVE_TRADING': ('ENABLE_LIVE', 'ENABLE_TRADING', 'ALLOW_LIVE_ORDERS'),
    'RISK__CAPITAL': ('RISK_CAPITAL', 'BACKTEST__CAPITAL'),
    'RISK__PER_TRADE_RISK_PCT': ('RISK_PER_TRADE_PCT', 'PER_TRADE_RISK_PCT'),
    'RISK__MAX_CAPITAL_PER_ORDER_PCT': (
        'RISK_PER_TRADE_CAP_PCT',
        'MAX_CAPITAL_PER_ORDER_PCT',
    ),
    'RISK__MAX_DRAWDOWN_PCT_DAY': (
        'RISK_MAX_DRAWDOWN_PCT',
        'DAILY_PNL_CAP_PCT',
    ),
    'TELEGRAM__BOT_TOKEN': ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_TOKEN'),
    'TELEGRAM__CHAT_ID': ('TELEGRAM_CHAT_ID',),
    'WEBSOCKET__DISABLED': ('WEBSOCKET_DISABLED', 'DISABLE_WEBSOCKET'),
    'SESSION_ALLOW_OUT_OF_HOURS': (
        'SESSION__ALLOW_OUT_OF_HOURS',
        'ALLOW_OFFHOURS_TESTING',
    ),
}


def resolve_env(canonical: str, default: Any | None = None) -> Any | None:
    """Return canonical environment value respecting legacy aliases.

    Args:
        canonical: Canonical environment variable name.
        default: Value returned when neither canonical nor alias is set.

    Returns:
        Any | None: Resolved environment value.

    Raises:
        None.
    """

    LOGGER.debug(
        'Entered resolve_env',
        extra={'event': 'env_alias_resolve_enter', 'canonical': canonical},
    )
    try:
        value = os.getenv(canonical)
        if value is not None:
            LOGGER.debug(
                'Condition met: resolve_env_canonical_hit',
                extra={'event': 'env_alias_canonical_hit', 'canonical': canonical},
            )
            return value
        aliases = ENV_ALIASES.get(canonical, ())
        for alias in aliases:
            alias_value = os.getenv(alias)
            if alias_value is not None:
                LOGGER.debug(
                    'Condition met: resolve_env_alias_hit',
                    extra={
                        'event': 'env_alias_alias_hit',
                        'canonical': canonical,
                        'alias': alias,
                    },
                )
                return alias_value
        LOGGER.debug(
            'Condition met: resolve_env_default',
            extra={'event': 'env_alias_default', 'canonical': canonical},
        )
        return default
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            'Failure in resolve_env: %s',
            exc,
            extra={'event': 'env_alias_resolve_error', 'canonical': canonical},
            exc_info=exc,
        )
        return default


def normalize_env_on_load() -> None:
    """Populate canonical environment keys when only aliases are provided.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        'Entered normalize_env_on_load',
        extra={'event': 'env_alias_normalize_enter'},
    )
    try:
        for canonical, aliases in ENV_ALIASES.items():
            if os.getenv(canonical) is not None:
                continue
            for alias in aliases:
                value = os.getenv(alias)
                if value is None:
                    continue
                os.environ[canonical] = value
                LOGGER.debug(
                    'Condition met: env_alias_promoted',
                    extra={
                        'event': 'env_alias_promoted',
                        'canonical': canonical,
                        'alias': alias,
                    },
                )
                break
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            'Failure in normalize_env_on_load: %s',
            exc,
            extra={'event': 'env_alias_normalize_error'},
            exc_info=exc,
        )
        raise


__all__ = ['ENV_ALIASES', 'resolve_env', 'normalize_env_on_load']
