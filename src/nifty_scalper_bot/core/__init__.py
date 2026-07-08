"""Core orchestration utilities for the trading bot."""

from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger
# TODO: remove compatibility shim once downstream modules are updated.
from nifty_scalper_bot.utils.pricing import canonical_price_source  # compat

_CORE_TRUTHY = {"1", "true", "yes", "y", "on", "live"}


def _core_env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _CORE_TRUTHY


def _real_live_mode_requested() -> bool:
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    live_enabled = _core_env_true("ENABLE_LIVE") or _core_env_true("ENABLE_LIVE_TRADING")
    paper_shadow = _core_env_true("PAPER_MODE") or _core_env_true("PAPER__ENABLED") or _core_env_true("SHADOW_MODE")
    return mode == "LIVE" and live_enabled and not paper_shadow


try:
    from nifty_scalper_bot.core.strategy_live_safety import apply_patches as _apply_strategy_live_safety

    _apply_strategy_live_safety()
except Exception as exc:  # noqa: BLE001 - non-live tooling imports should remain usable
    get_logger(__name__).error(
        "STRATEGY_LIVE_SAFETY_PATCH_FAILED error=%s",
        exc,
        extra={"event": "STRATEGY_LIVE_SAFETY_PATCH_FAILED", "error_type": type(exc).__name__},
    )
    if _real_live_mode_requested():
        raise RuntimeError("strategy_live_safety_patch_failed") from exc

try:
    from nifty_scalper_bot.core.strategy_exit_score_diagnostics import apply_patches as _apply_strategy_exit_score_diagnostics

    _apply_strategy_exit_score_diagnostics()
except Exception as exc:  # noqa: BLE001 - diagnostics must not disable tooling imports
    get_logger(__name__).error(
        "STRATEGY_EXIT_SCORE_DIAGNOSTIC_PATCH_FAILED error=%s",
        exc,
        extra={"event": "STRATEGY_EXIT_SCORE_DIAGNOSTIC_PATCH_FAILED", "error_type": type(exc).__name__},
    )

try:
    from nifty_scalper_bot.core.boot_log_safety import apply_filters as _apply_boot_log_rate_controls

    _apply_boot_log_rate_controls()
except Exception as exc:  # noqa: BLE001 - core package import must not crash tooling
    get_logger(__name__).error(
        "BOOT_LOG_RATE_CONTROL_FAILED error=%s",
        exc,
        extra={"event": "BOOT_LOG_RATE_CONTROL_FAILED", "error_type": type(exc).__name__},
    )

try:
    from nifty_scalper_bot.core.market_data_hardening_bootstrap import (
        install_market_data_hardening_or_raise as _install_market_data_hardening,
    )

    _install_market_data_hardening(get_logger(__name__))
except Exception as exc:  # noqa: BLE001 - fail closed only for real live mode
    get_logger(__name__).error(
        "MARKET_DATA_HARDENING_BOOTSTRAP_FAILED error=%s",
        exc,
        extra={
            "event": "MARKET_DATA_HARDENING_BOOTSTRAP_FAILED",
            "error_type": type(exc).__name__,
        },
    )
    if _real_live_mode_requested():
        raise RuntimeError("market_data_hardening_bootstrap_failed") from exc

__all__ = ["NiftyScalperApp", "canonical_price_source"]

_LOGGER = get_logger(__name__)


def _apply_app_runtime_patches(app_module: Any) -> None:
    from nifty_scalper_bot.core.boot_readiness_safety import apply_app_patch as _ready_adapter
    from nifty_scalper_bot.core.polling_failover_runtime import apply_app_patch as _polling_adapter

    _ready_adapter(app_module)
    _polling_adapter(app_module)


def __getattr__(name: str) -> Any:
    """Return lazily imported core entry points.

    Args:
        name: Attribute requested from the package.

    Returns:
        Any: Resolved attribute when exported by the package.

    Raises:
        AttributeError: If the requested attribute is not available.
    """

    _LOGGER.debug(
        "Entered core.__getattr__",
        extra={"event": "core_init_getattr_enter", "attribute": name},
    )
    if name in {"NiftyScalperApp", "app"}:
        try:
            from . import app as _app_module

            _apply_app_runtime_patches(_app_module)
            if name == "app":
                return _app_module
            _App = _app_module.NiftyScalperApp
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Failure in core.__getattr__: %s",
                exc,
                extra={"event": "core_init_getattr_error", "attribute": name},
            )
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        _LOGGER.info(
            "Condition met: resolved NiftyScalperApp",
            extra={"event": "core_init_getattr_resolved", "attribute": name},
        )
        return _App
    _LOGGER.info(
        "Condition met: attribute not exported",
        extra={"event": "core_init_getattr_missing", "attribute": name},
    )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
