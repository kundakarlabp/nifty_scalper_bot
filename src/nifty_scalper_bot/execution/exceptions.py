"""Execution exception hierarchy for granular error handling.

Canonical execution/broker base classes live in ``utils.errors`` so imports from
``nifty_scalper_bot.execution.exceptions`` and ``nifty_scalper_bot.utils.errors``
refer to the same runtime class objects.
"""

from __future__ import annotations

from nifty_scalper_bot.utils.errors import (
    BrokerError,
    OrderExecutionError,
    OrderPlacementError,
)


class MarginCheckError(OrderPlacementError):
    """Insufficient margin to place order."""


class OrderModificationError(OrderExecutionError):
    """Order modification request failed."""


class RiskBlockError(OrderPlacementError):
    """Order blocked by risk manager gates."""


__all__ = [
    "OrderExecutionError",
    "BrokerError",
    "OrderPlacementError",
    "MarginCheckError",
    "OrderModificationError",
    "RiskBlockError",
]
