"""Startup hydration helpers delegated to MarketDataManager."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("nifty_scalper_bot.core.hydration")


def assert_valid_token(token: Any) -> int:
    """Validate token. Args: token. Returns: int token. Raises: RuntimeError."""
    if token is None:
        raise RuntimeError('token is None')
    try:
        parsed = int(token)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'invalid token: {token!r}') from exc
    if parsed <= 0:
        raise RuntimeError(f'invalid token: {parsed}')
    return parsed


def hydrate_contracts(*args: Any, **kwargs: Any) -> dict[int, list[dict[str, Any]]]:
    """No-op legacy compatibility. Args: any. Returns: empty map. Raises: none."""
    LOGGER.info('hydrate_contracts_legacy_noop delegated_to_market_data_manager')
    return {}
