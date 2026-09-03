"""Environment normalization helpers for live/paper defaults."""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)
LIVE_PER_TRADE_RISK_PCT = "7.0"
# Each live entry is capped to the remaining daily-loss budget before broker
# submission. Keep that budget coherent with the canonical LIVE per-trade risk
# so an indivisible NIFTY lot is not constrained by contradictory percentages.
LIVE_DAILY_LOSS_PCT = LIVE_PER_TRADE_RISK_PCT
PRODUCTION_LIVE_DEFAULT_INITIALIZED = "PRODUCTION_LIVE_DEFAULT_INITIALIZED"


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


def _is_lightsail_production() -> bool:
    return (os.getenv('DEPLOYMENT_PLATFORM') or '').strip().lower() == 'aws_lightsail'


def _production_live_default_enabled() -> bool:
    """Return whether this Lightsail host needs its one-time LIVE migration."""
    if not _is_lightsail_production():
        return False
    if truthy(os.getenv(PRODUCTION_LIVE_DEFAULT_INITIALIZED)):
        return False
    preference = os.getenv('PRODUCTION_DEFAULT_LIVE')
    if preference is not None and preference.strip() and not truthy(preference):
        return False
    return True


def _persist_production_live_defaults(defaults: dict[str, str]) -> None:
    """Persist the one-time Lightsail LIVE migration without touching secrets."""
    env_path = (os.getenv('BOT_ENV_FILE') or '').strip()
    if not env_path:
        return
    path = Path(env_path).expanduser()
    if not path.exists() or not path.is_file():
        return

    updates = dict(defaults)
    updates[PRODUCTION_LIVE_DEFAULT_INITIALIZED] = 'true'
    existing = path.read_text(encoding='utf-8').splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            out.append(line)
            continue
        key = stripped.split('=', 1)[0].strip()
        if key in updates:
            if key not in seen:
                out.append(f'{key}={updates[key]}')
                seen.add(key)
            continue
        out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f'{key}={value}')

    tmp = path.with_name(f'.{path.name}.live-default.tmp')
    try:
        tmp.write_text('\n'.join(out).rstrip() + '\n', encoding='utf-8')
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.environ[PRODUCTION_LIVE_DEFAULT_INITIALIZED] = 'true'
    except OSError as exc:
        LOGGER.warning('Could not persist Lightsail LIVE default migration: %s', exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def normalise_live_env_defaults() -> None:
    """Derive live/paper env defaults. Args: none. Returns: None. Raises: none."""
    enable_live = truthy(os.getenv('ENABLE_LIVE'))
    execution_mode = (os.getenv('EXECUTION_MODE') or '').strip().upper()
    production_default_live = _production_live_default_enabled()
    live_requested = production_default_live or enable_live or execution_mode == 'LIVE'

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
        non_live_mode = execution_mode if execution_mode in {'SHADOW', 'PAPER'} else 'PAPER'
        defaults = {
            'ENABLE_LIVE': 'false',
            'ENABLE_LIVE_TRADING': 'false',
            'EXECUTION_MODE': non_live_mode,
            'ORDERS__ENABLE_LIVE': 'false',
            'PAPER__ENABLED': 'true',
            'PAPER_MODE': 'true',
            'SHADOW_MODE': 'true',
        }

    production_initialized = _is_lightsail_production() and truthy(
        os.getenv(PRODUCTION_LIVE_DEFAULT_INITIALIZED)
    )
    for key, value in defaults.items():
        if production_default_live or production_initialized:
            # On the first upgraded boot, migrate the legacy SHADOW env to LIVE.
            # Thereafter ENABLE_LIVE + EXECUTION_MODE are canonical and the
            # derived aliases are synchronized in-process, so an explicit admin
            # switch back to SHADOW remains authoritative across restarts.
            os.environ[key] = value
        else:
            setdefault_env(key, value)

    if production_default_live:
        _persist_production_live_defaults(defaults)

    if live_requested:
        # One canonical live risk envelope. Keep accepted aliases aligned so
        # legacy deployment values cannot silently make the 7% per-trade policy
        # unattainable behind a lower daily-loss ceiling. Existing remaining-day
        # sizing clamps and final RiskManager breakers remain unchanged.
        os.environ['RISK__PER_TRADE_RISK_PCT'] = LIVE_PER_TRADE_RISK_PCT
        os.environ['RISK_PER_TRADE_PCT'] = LIVE_PER_TRADE_RISK_PCT
        os.environ['RISK_DAILY_LOSS_PCT'] = LIVE_DAILY_LOSS_PCT
        os.environ['RISK_DAILY_PNL_CAP_PCT'] = LIVE_DAILY_LOSS_PCT
        os.environ['RISK_MAX_DAILY_LOSS_PCT'] = LIVE_DAILY_LOSS_PCT
        os.environ['DAILY_PNL_CAP_PCT'] = LIVE_DAILY_LOSS_PCT


def resolve_build_sha() -> str:
    """Canonical build/commit SHA for the running deployment.

    Single source of truth for main.py startup banner, runner status blocks
    and the Telegram /status command — previously three divergent env
    fallback chains. Railway sets RAILWAY_GIT_COMMIT_SHA.
    """
    import os

    return (
        os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("GIT_COMMIT_SHA")
        or os.getenv("SOURCE_VERSION")
        or os.getenv("GIT_SHA")
        or "unknown"
    )