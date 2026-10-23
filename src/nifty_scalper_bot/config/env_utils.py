"""Environment normalization helpers for live/paper defaults."""

from __future__ import annotations

import logging
import os

from nifty_scalper_bot.config import defaults

LOGGER = logging.getLogger(__name__)
LIVE_PER_TRADE_RISK_PCT = defaults.DEFAULT_LIVE_PER_TRADE_RISK_PCT
# Daily loss is an independent portfolio-level circuit breaker. An indivisible
# lot that cannot fit the per-trade risk budget is skipped rather than widening
# either limit at runtime.
LIVE_DAILY_LOSS_PCT = defaults.DEFAULT_LIVE_DAILY_LOSS_PCT
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
        LOGGER.warning(
            "parse_float_env: invalid value %r, using default %s", value, default
        )
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
        LOGGER.warning(
            "parse_int_env: invalid value %r, using default %s", value, default
        )
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
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setdefault_env(key: str, value: str) -> None:
    """Set env key only when missing. Args: key, value. Returns: None. Raises: none."""
    if os.getenv(key) is None:
        os.environ[key] = value


def normalise_live_env_defaults() -> None:
    """Resolve a coherent trading mode from explicit operator settings.

    LIVE requires both canonical controls. Missing/conflicting controls fail
    closed, including on a fresh AWS instance. No environment file is written.
    """
    enable_live = truthy(os.getenv("ENABLE_LIVE"))
    mode = (os.getenv("EXECUTION_MODE") or "").strip().upper()
    live_requested = enable_live and mode == "LIVE"

    if live_requested:
        values = {
            "ENABLE_LIVE": "true",
            "ENABLE_LIVE_TRADING": "true",
            "EXECUTION_MODE": "LIVE",
            "ORDERS__ENABLE_LIVE": "true",
            "PAPER__ENABLED": "false",
            "PAPER_MODE": "false",
            "SHADOW_MODE": "false",
        }
    else:
        # Conflicting ENABLE_LIVE / EXECUTION_MODE settings always disable
        # live orders; retain an explicit paper or shadow request where valid.
        nonlive = (
            mode
            if mode in {"SHADOW", "PAPER"}
            else ("SHADOW" if mode or enable_live else "PAPER")
        )
        values = {
            "ENABLE_LIVE": "false",
            "ENABLE_LIVE_TRADING": "false",
            "EXECUTION_MODE": nonlive,
            "ORDERS__ENABLE_LIVE": "false",
            "PAPER__ENABLED": "true",
            "PAPER_MODE": "true",
            "SHADOW_MODE": "true",
        }
        if enable_live or mode == "LIVE":
            LOGGER.error(
                "LIVE_MODE_CONFLICT_FAIL_CLOSED enable_live=%s execution_mode=%s",
                enable_live,
                mode,
            )

    # This is one canonical, in-process normalization owner; legacy aliases
    # cannot re-arm a supposedly disarmed environment.
    for key, value in values.items():
        os.environ[key] = value

    if live_requested:
        # Operator-approved bounded entry paths; each explicit false wins.
        setdefault_env("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED", "true")
        setdefault_env("VWAP_TREND_RETEST_EXTENSION_ENABLED", "true")
        setdefault_env("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED", "true")
        LOGGER.warning(
            "SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=%s mode=LIVE",
            truthy(os.getenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED")),
        )
        LOGGER.warning(
            "LIVE_ENTRY_QUALITY_CONFIG vwap_retest=%s adaptive_ofi=%s mode=LIVE",
            truthy(os.getenv("VWAP_TREND_RETEST_EXTENSION_ENABLED")),
            truthy(os.getenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED")),
        )

        # Preserve the existing LIVE risk profile. The canonical values live
        # in config/defaults.py; aliases are normalized for older consumers.
        os.environ["RISK__PER_TRADE_RISK_PCT"] = LIVE_PER_TRADE_RISK_PCT
        os.environ["RISK_PER_TRADE_PCT"] = LIVE_PER_TRADE_RISK_PCT
        os.environ["RISK_DAILY_LOSS_PCT"] = LIVE_DAILY_LOSS_PCT
        os.environ["RISK_DAILY_PNL_CAP_PCT"] = LIVE_DAILY_LOSS_PCT
        os.environ["RISK_MAX_DAILY_LOSS_PCT"] = LIVE_DAILY_LOSS_PCT
        os.environ["DAILY_PNL_CAP_PCT"] = LIVE_DAILY_LOSS_PCT


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
