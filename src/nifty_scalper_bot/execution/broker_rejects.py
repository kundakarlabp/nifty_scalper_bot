"""Broker error normalization and deterministic recovery hints.

The broker frequently returns human-readable error text.  This module converts
that text into a stable internal reason code so the live order path can decide
whether it is safe to retry, refresh a quote, resize an order, or stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BrokerReject(Enum):
    """Normalized broker rejection reasons used by entry and exit workflows."""

    MIS_CUTOFF = "MIS_CUTOFF"
    MARKET_CLOSED = "MARKET_CLOSED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    PRICE_OUT_OF_RANGE = "PRICE_OUT_OF_RANGE"
    TRIGGER_PRICE_INVALID = "TRIGGER_PRICE_INVALID"
    QUANTITY_INVALID = "QUANTITY_INVALID"
    FREEZE_LIMIT = "FREEZE_LIMIT"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    ORDER_STATE_INVALID = "ORDER_STATE_INVALID"
    THROTTLED = "THROTTLED"
    TRANSIENT_BROKER = "TRANSIENT_BROKER"
    AUTHENTICATION = "AUTHENTICATION"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(Enum):
    """Safe, high-level recovery action for a normalized rejection."""

    REFRESH_QUOTE_AND_REPRICE = "REFRESH_QUOTE_AND_REPRICE"
    RESIZE_TO_AVAILABLE_MARGIN = "RESIZE_TO_AVAILABLE_MARGIN"
    SPLIT_QUANTITY = "SPLIT_QUANTITY"
    RECONCILE_ORDERBOOK = "RECONCILE_ORDERBOOK"
    RETRY_WITH_BACKOFF = "RETRY_WITH_BACKOFF"
    SWITCH_PRODUCT_IF_VALID = "SWITCH_PRODUCT_IF_VALID"
    STOP_AND_ALERT = "STOP_AND_ALERT"


@dataclass(frozen=True, slots=True)
class BrokerRejectDecision:
    """Normalized rejection plus the only permitted automatic response."""

    reason: BrokerReject
    retryable: bool
    action: RecoveryAction


def parse_broker_error(message: str | Exception | None) -> BrokerReject:
    """Return a normalized rejection code for Zerodha-style error payloads."""

    token = " ".join(str(message or "").strip().lower().split())
    if not token:
        return BrokerReject.UNKNOWN

    if (
        "intraday orders (mis) are allowed only till" in token
        or "mis only till" in token
        or "mis order is not allowed" in token
    ):
        return BrokerReject.MIS_CUTOFF

    if (
        "markets are closed right now" in token
        or "market is closed" in token
        or ("amo" in token and ("post" in token or "outside market" in token))
    ):
        return BrokerReject.MARKET_CLOSED

    if any(
        phrase in token
        for phrase in (
            "insufficient funds",
            "required margin",
            "not enough cash",
            "not enough balance",
            "margin required",
            "available margin",
        )
    ):
        return BrokerReject.INSUFFICIENT_FUNDS

    if any(
        phrase in token
        for phrase in (
            "price is outside the allowed range",
            "price out of range",
            "price bands",
            "circuit limit",
            "limit price",
            "difference between limit price and trigger price",
        )
    ):
        return BrokerReject.PRICE_OUT_OF_RANGE

    if any(
        phrase in token
        for phrase in (
            "trigger price",
            "stoploss trigger",
            "stop loss trigger",
            "sl trigger",
        )
    ) and any(
        phrase in token
        for phrase in ("invalid", "should be", "must be", "higher", "lower")
    ):
        return BrokerReject.TRIGGER_PRICE_INVALID

    if any(
        phrase in token
        for phrase in (
            "freeze quantity",
            "freeze limit",
            "maximum allowed quantity",
            "maximum order quantity",
        )
    ):
        return BrokerReject.FREEZE_LIMIT

    if any(
        phrase in token
        for phrase in (
            "invalid quantity",
            "quantity should be",
            "lot size",
            "quantity is not a multiple",
            "minimum quantity",
        )
    ):
        return BrokerReject.QUANTITY_INVALID

    if any(
        phrase in token
        for phrase in (
            "invalid instrument",
            "instrument token",
            "unknown tradingsymbol",
            "invalid tradingsymbol",
            "contract is not enabled",
            "instrument is not tradable",
            "expired contract",
        )
    ):
        return BrokerReject.INSTRUMENT_INVALID

    if any(
        phrase in token
        for phrase in (
            "duplicate order",
            "duplicate request",
            "request already processed",
            "order already placed",
        )
    ):
        return BrokerReject.DUPLICATE_ORDER

    if any(
        phrase in token
        for phrase in ("order not found", "invalid order id", "unknown order")
    ):
        return BrokerReject.ORDER_NOT_FOUND

    if any(
        phrase in token
        for phrase in (
            "order cannot be modified",
            "order cannot be cancelled",
            "order is complete",
            "order already complete",
            "invalid order status",
        )
    ):
        return BrokerReject.ORDER_STATE_INVALID

    if any(
        phrase in token
        for phrase in ("rate limit", "too many requests", "throttle", "429")
    ):
        return BrokerReject.THROTTLED

    if any(
        phrase in token
        for phrase in (
            "gateway timeout",
            "service unavailable",
            "temporarily unavailable",
            "connection reset",
            "connection timed out",
            "read timed out",
            "502",
            "503",
            "504",
        )
    ):
        return BrokerReject.TRANSIENT_BROKER

    if any(
        phrase in token
        for phrase in (
            "token is invalid",
            "invalid access token",
            "session expired",
            "authentication failed",
            "permission denied",
            "403",
        )
    ):
        return BrokerReject.AUTHENTICATION

    return BrokerReject.UNKNOWN


def recovery_decision(message: str | Exception | None) -> BrokerRejectDecision:
    """Return the permitted automatic recovery for a broker error.

    Ambiguous responses are deliberately non-retryable.  An API timeout can mean
    the broker accepted the order but the client missed the acknowledgement;
    those cases must reconcile the broker orderbook before any resubmission.
    """

    reason = parse_broker_error(message)
    mapping = {
        BrokerReject.PRICE_OUT_OF_RANGE: (True, RecoveryAction.REFRESH_QUOTE_AND_REPRICE),
        BrokerReject.TRIGGER_PRICE_INVALID: (True, RecoveryAction.REFRESH_QUOTE_AND_REPRICE),
        BrokerReject.INSUFFICIENT_FUNDS: (True, RecoveryAction.RESIZE_TO_AVAILABLE_MARGIN),
        BrokerReject.FREEZE_LIMIT: (True, RecoveryAction.SPLIT_QUANTITY),
        BrokerReject.DUPLICATE_ORDER: (True, RecoveryAction.RECONCILE_ORDERBOOK),
        BrokerReject.ORDER_NOT_FOUND: (True, RecoveryAction.RECONCILE_ORDERBOOK),
        BrokerReject.ORDER_STATE_INVALID: (True, RecoveryAction.RECONCILE_ORDERBOOK),
        BrokerReject.TRANSIENT_BROKER: (True, RecoveryAction.RECONCILE_ORDERBOOK),
        BrokerReject.THROTTLED: (True, RecoveryAction.RETRY_WITH_BACKOFF),
        BrokerReject.MIS_CUTOFF: (True, RecoveryAction.SWITCH_PRODUCT_IF_VALID),
    }
    retryable, action = mapping.get(
        reason,
        (False, RecoveryAction.STOP_AND_ALERT),
    )
    return BrokerRejectDecision(reason=reason, retryable=retryable, action=action)


__all__ = [
    "BrokerReject",
    "BrokerRejectDecision",
    "RecoveryAction",
    "parse_broker_error",
    "recovery_decision",
]
