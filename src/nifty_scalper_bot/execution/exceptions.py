"""Execution exception hierarchy for granular error handling."""

from __future__ import annotations


class OrderExecutionError(Exception):
    """Base exception for all order execution failures."""


class BrokerError(OrderExecutionError):
    """Broker API errors (network, auth, rate limit, reject)."""


class OrderPlacementError(OrderExecutionError):
    """Order placement validation failed before reaching broker."""


class MarginCheckError(OrderPlacementError):
    """Insufficient margin to place order."""


class OrderModificationError(OrderExecutionError):
    """Order modification request failed."""


class RiskBlockError(OrderPlacementError):
    """Order blocked by risk manager gates."""
