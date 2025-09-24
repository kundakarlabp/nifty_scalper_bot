from __future__ import annotations

"""Utilities for classifying broker error messages."""

from typing import Any

AUTH = "AUTH"
THROTTLE = "THROTTLE"
SUBSCRIPTION = "SUBSCRIPTION"
MAINTENANCE = "MAINTENANCE"
NETWORK = "NETWORK"
UNKNOWN = "UNKNOWN"

_KEY_AUTH_PHRASES = [
    "invalid session",
    "session expired",
    "not a valid access token",
]

_KEY_THROTTLE = [
    "throttle",
    "rate limit",
    "too many requests",
    "429",
]

_KEY_SUBSCRIPTION = [
    "subscription",
    "permission",
]

_KEY_MAINTENANCE = [
    "maintenance",
]

_KEY_NETWORK = [
    "timeout",
    "timed out",
    "connection",
    "network",
]


def classify_broker_error(exc_or_msg: Any, status: int | None = None) -> str:
    """Return a coarse classification for broker errors.

    Parameters
    ----------
    exc_or_msg:
        Exception instance or error message string.
    status:
        Optional HTTP status code.
    """
    msg = str(exc_or_msg)
    stat = status
    if stat is None:
        stat = getattr(exc_or_msg, "status", None)
        if stat is None:
            resp = getattr(exc_or_msg, "response", None)
            stat = getattr(resp, "status", getattr(resp, "status_code", None))
    m = msg.lower()
    if "incorrect 'api_key' or 'access_token'" in m and stat not in (401, 403):
        return UNKNOWN
    if stat in (401, 403) or any(p in m for p in _KEY_AUTH_PHRASES):
        return AUTH
    if any(k in m for k in _KEY_THROTTLE) or stat == 429:
        return THROTTLE
    if any(k in m for k in _KEY_SUBSCRIPTION):
        return SUBSCRIPTION
    if any(k in m for k in _KEY_MAINTENANCE):
        return MAINTENANCE
    if any(k in m for k in _KEY_NETWORK):
        return NETWORK
    return UNKNOWN


__all__ = [
    "classify_broker_error",
    "AUTH",
    "THROTTLE",
    "SUBSCRIPTION",
    "MAINTENANCE",
    "NETWORK",
    "UNKNOWN",
]
