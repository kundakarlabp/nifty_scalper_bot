"""Explicit runtime order manager composition."""

from __future__ import annotations

from nifty_scalper_bot.execution import order_manager_legacy as _legacy


class RuntimeOrderManager(_legacy.OrderManager):
    """Production order manager assembled through normal inheritance."""


__all__ = ["RuntimeOrderManager"]
