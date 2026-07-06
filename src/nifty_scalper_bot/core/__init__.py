"""Core orchestration utilities for the trading bot."""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.utils.logging import get_logger
# TODO: remove compatibility shim once downstream modules are updated.
from nifty_scalper_bot.utils.pricing import canonical_price_source  # compat

try:
    from nifty_scalper_bot.core.strategy_live_safety import apply_patches as _apply_strategy_live_safety

    _apply_strategy_live_safety()
except Exception as exc:  # noqa: BLE001 - core package import must not crash tooling
    get_logger(__name__).error(
        "STRATEGY_LIVE_SAFETY_PATCH_FAILED error=%s",
        exc,
        extra={"event": "STRATEGY_LIVE_SAFETY_PATCH_FAILED", "error_type": type(exc).__name__},
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

__all__ = ["NiftyScalperApp", "canonical_price_source"]

_LOGGER = get_logger(__name__)


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
    if name == "NiftyScalperApp":
        try:
            from .app import NiftyScalperApp as _App
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
