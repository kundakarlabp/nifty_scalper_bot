"""Typed exception hierarchy for the Nifty scalper bot."""

from __future__ import annotations


class BotError(Exception):
    """Base exception for all bot related errors."""


class ConfigurationError(BotError):
    """Raised when configuration is invalid or incomplete."""


class RateLimitError(BotError):
    """Raised when rate limit thresholds are exceeded."""


class BrokerError(BotError):
    """Raised when broker integrations fail."""


class OrderPlacementError(BrokerError):
    """Raised when order placement fails after retries or validation."""


class DataStaleError(BotError):
    """Raised when market data is considered stale."""


class WebSocketError(BotError):
    """Raised when websocket connectivity fails."""


__all__ = [
    "BotError",
    "ConfigurationError",
    "RateLimitError",
    "BrokerError",
    "OrderPlacementError",
    "DataStaleError",
    "WebSocketError",
]
