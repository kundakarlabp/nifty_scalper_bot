"""Deterministic broker rejection classification and permitted recovery action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BrokerFailure(Enum):
    MIS_CUTOFF = "MIS_CUTOFF"
    MARKET_CLOSED = "MARKET_CLOSED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    PRICE_INVALID = "PRICE_INVALID"
    TRIGGER_INVALID = "TRIGGER_INVALID"
    QUANTITY_INVALID = "QUANTITY_INVALID"
    FREEZE_LIMIT = "FREEZE_LIMIT"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    DUPLICATE_OR_AMBIGUOUS = "DUPLICATE_OR_AMBIGUOUS"
    ORDER_STATE_UNKNOWN = "ORDER_STATE_UNKNOWN"
    THROTTLED = "THROTTLED"
    TRANSIENT = "TRANSIENT"
    AUTHENTICATION = "AUTHENTICATION"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(Enum):
    REFRESH_AND_REVALIDATE = "REFRESH_AND_REVALIDATE"
    RESIZE_AND_REVALIDATE = "RESIZE_AND_REVALIDATE"
    CAP_QUANTITY_AND_REVALIDATE = "CAP_QUANTITY_AND_REVALIDATE"
    RECONCILE_BEFORE_RETRY = "RECONCILE_BEFORE_RETRY"
    BACKOFF_AND_REVALIDATE = "BACKOFF_AND_REVALIDATE"
    STOP_AND_ALERT = "STOP_AND_ALERT"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    failure: BrokerFailure
    action: RecoveryAction
    retryable: bool
    reconcile_first: bool = False


def classify_broker_failure(message: object) -> BrokerFailure:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return BrokerFailure.UNKNOWN
    if any(subject in text for subject in (
        "trigger price", "stop loss trigger", "stoploss trigger", "sl trigger"
    )) and any(word in text for word in (
        "invalid", "should be", "must be", "higher", "lower"
    )):
        return BrokerFailure.TRIGGER_INVALID
    ordered = (
        (BrokerFailure.MIS_CUTOFF, ("mis only till", "mis order is not allowed", "intraday orders (mis) are allowed only till")),
        (BrokerFailure.MARKET_CLOSED, ("market is closed", "markets are closed", "outside market hours")),
        (BrokerFailure.INSUFFICIENT_FUNDS, ("insufficient funds", "required margin", "margin required", "not enough balance")),
        (BrokerFailure.FREEZE_LIMIT, ("freeze quantity", "freeze limit", "maximum order quantity")),
        (BrokerFailure.QUANTITY_INVALID, ("invalid quantity", "quantity is not a multiple", "minimum quantity")),
        (BrokerFailure.INSTRUMENT_INVALID, ("invalid instrument", "invalid tradingsymbol", "expired contract", "instrument is not tradable")),
        (BrokerFailure.DUPLICATE_OR_AMBIGUOUS, ("duplicate order", "duplicate request", "request already processed", "order already placed")),
        (BrokerFailure.ORDER_STATE_UNKNOWN, ("order not found", "invalid order id", "unknown order", "invalid order status")),
        (BrokerFailure.THROTTLED, ("rate limit", "too many requests", "throttle", "http 429", "status 429")),
        (BrokerFailure.TRANSIENT, ("gateway timeout", "service unavailable", "temporarily unavailable", "connection reset", "connection timed out", "read timed out", "http 502", "http 503", "http 504")),
        (BrokerFailure.AUTHENTICATION, ("invalid access token", "session expired", "authentication failed", "permission denied", "http 403")),
        (BrokerFailure.PRICE_INVALID, ("price out of range", "price bands", "circuit limit", "invalid limit price", "limit price is outside")),
    )
    for failure, phrases in ordered:
        if any(phrase in text for phrase in phrases):
            return failure
    if "amo" in text and ("post" in text or "outside market" in text):
        return BrokerFailure.MARKET_CLOSED
    return BrokerFailure.UNKNOWN


def decide_recovery(message: object) -> RecoveryDecision:
    failure = classify_broker_failure(message)
    mapping = {
        BrokerFailure.PRICE_INVALID: RecoveryDecision(failure, RecoveryAction.REFRESH_AND_REVALIDATE, True),
        BrokerFailure.TRIGGER_INVALID: RecoveryDecision(failure, RecoveryAction.REFRESH_AND_REVALIDATE, True),
        BrokerFailure.INSUFFICIENT_FUNDS: RecoveryDecision(failure, RecoveryAction.RESIZE_AND_REVALIDATE, True),
        BrokerFailure.FREEZE_LIMIT: RecoveryDecision(failure, RecoveryAction.CAP_QUANTITY_AND_REVALIDATE, True),
        BrokerFailure.DUPLICATE_OR_AMBIGUOUS: RecoveryDecision(failure, RecoveryAction.RECONCILE_BEFORE_RETRY, True, True),
        BrokerFailure.ORDER_STATE_UNKNOWN: RecoveryDecision(failure, RecoveryAction.RECONCILE_BEFORE_RETRY, True, True),
        BrokerFailure.TRANSIENT: RecoveryDecision(failure, RecoveryAction.RECONCILE_BEFORE_RETRY, True, True),
        BrokerFailure.THROTTLED: RecoveryDecision(failure, RecoveryAction.BACKOFF_AND_REVALIDATE, True),
    }
    return mapping.get(
        failure,
        RecoveryDecision(failure, RecoveryAction.STOP_AND_ALERT, False),
    )


__all__ = [
    "BrokerFailure",
    "RecoveryAction",
    "RecoveryDecision",
    "classify_broker_failure",
    "decide_recovery",
]
