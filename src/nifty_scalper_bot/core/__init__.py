"""Core orchestration utilities for the trading bot."""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.utils.logging import get_logger

__all__ = ["NiftyScalperApp"]

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
