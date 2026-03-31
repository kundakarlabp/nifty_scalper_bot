"""Broker error normalization helpers."""

from __future__ import annotations

from enum import Enum


class BrokerReject(Enum):
    """Enumerate normalized broker rejection reasons."""

    MIS_CUTOFF = "MIS_CUTOFF"
    MARKET_CLOSED = "MARKET_CLOSED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    THROTTLED = "THROTTLED"
    UNKNOWN = "UNKNOWN"


def parse_broker_error(message: str | Exception | None) -> BrokerReject:
    """Return normalized rejection code for Zerodha-style errors.

    Args:
        message: Raw broker error payload or exception.

    Returns:
        BrokerReject: Parsed rejection category.
    """

    token = str(message or "").strip().lower()
    if not token:
        return BrokerReject.UNKNOWN
    if (
        "intraday orders (mis) are allowed only till" in token
        or "mis only till" in token
    ):
        return BrokerReject.MIS_CUTOFF
    if "markets are closed right now" in token or ("amo" in token and "post" in token):
        return BrokerReject.MARKET_CLOSED
    if "insufficient funds" in token or "required margin" in token:
        return BrokerReject.INSUFFICIENT_FUNDS
    if "rate limit" in token or "too many requests" in token or "throttle" in token:
        return BrokerReject.THROTTLED
    return BrokerReject.UNKNOWN


__all__ = ["BrokerReject", "parse_broker_error"]
