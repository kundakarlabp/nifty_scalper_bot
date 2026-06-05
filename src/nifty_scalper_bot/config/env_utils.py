"""Environment normalization helpers for live/paper defaults."""

from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)


def _strip_inline_comment(text: str) -> str:
    """Remove an inline ``# comment`` and surrounding whitespace.

    Env files sometimes contain ``KEY=30.0   # note`` and the value read back
    includes the comment, which breaks ``float()``/``int()``.
    """
    # Only treat ``#`` as a comment when it is clearly trailing (preceded by a
    # space) or starts the value; this avoids mangling values that legitimately
    # contain ``#``. For numeric config this is safe.
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.strip().strip('"').strip("'").strip()


def parse_float_env(value: object, default: float) -> float:
    """Safely parse a float from a config/env value.

    Accepts an int/float directly, or a string that may contain surrounding
    whitespace, quotes, or an inline ``# comment``. Returns *default* for
    None/blank/invalid input (logging a warning on invalid), never raising.
    """
    if value is None:
        return default
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = _strip_inline_comment(str(value))
    if cleaned == "":
        return default
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        LOGGER.warning("parse_float_env: invalid value %r, using default %s", value, default)
        return default


def parse_int_env(value: object, default: int) -> int:
    """Safely parse an int (via float to tolerate '30.0'). See parse_float_env."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = _strip_inline_comment(str(value))
    if cleaned == "":
        return default
    try:
        return int(float(cleaned))
    except (TypeError, ValueError):
        LOGGER.warning("parse_int_env: invalid value %r, using default %s", value, default)
        return default


def parse_bool_env(value: object, default: bool = False) -> bool:
    """Safely parse a bool from a config/env value (inline comments stripped)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    cleaned = _strip_inline_comment(str(value)).lower()
    if cleaned == "":
        return default
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    LOGGER.warning("parse_bool_env: invalid value %r, using default %s", value, default)
    return default


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
