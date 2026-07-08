"""Lightweight install helper for polling failover runtime patches."""

from __future__ import annotations

from typing import Any


def install_on_app_module(app_module: Any) -> bool:
    """Install polling failover runtime patch on a loaded ``core.app`` module."""
    from nifty_scalper_bot.core.polling_failover_runtime import apply_app_patch

    return apply_app_patch(app_module)


__all__ = ["install_on_app_module"]
