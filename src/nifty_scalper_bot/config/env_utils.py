"""Environment normalization helpers for live/paper defaults."""

from __future__ import annotations

import os


def truthy(value: str | None) -> bool:
    """Parse truthy env flags. Args: value. Returns: bool. Raises: none."""
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def setdefault_env(key: str, value: str) -> None:
    """Set env key only when missing. Args: key, value. Returns: None. Raises: none."""
    if os.getenv(key) is None:
        os.environ[key] = value


def normalise_live_env_defaults() -> None:
    """Derive live/paper env defaults. Args: none. Returns: None. Raises: none."""
    enable_live = truthy(os.getenv('ENABLE_LIVE'))
    execution_mode = (os.getenv('EXECUTION_MODE') or '').strip().upper()
    live_requested = enable_live or execution_mode == 'LIVE'

    if live_requested:
        defaults = {
            'ENABLE_LIVE': 'true',
            'ENABLE_LIVE_TRADING': 'true',
            'EXECUTION_MODE': 'LIVE',
            'ORDERS__ENABLE_LIVE': 'true',
            'PAPER__ENABLED': 'false',
            'PAPER_MODE': 'false',
            'SHADOW_MODE': 'false',
        }
    else:
        defaults = {
            'ENABLE_LIVE': 'false',
            'ENABLE_LIVE_TRADING': 'false',
            'EXECUTION_MODE': 'PAPER',
            'ORDERS__ENABLE_LIVE': 'false',
            'PAPER__ENABLED': 'true',
            'PAPER_MODE': 'true',
            'SHADOW_MODE': 'true',
        }

    for key, value in defaults.items():
        setdefault_env(key, value)
