"""Normalize broker failures into deterministic, context-safe recovery decisions.

This module does not execute recovery.  It describes the only recovery class a
caller may attempt.  The live order state machine must still refresh broker and
market state, re-run risk checks, and persist each transition before retrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class BrokerReject(Enum):
    """Stable broker-rejection categories used by entry and exit workflows."""

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
    """High-level action permitted after a normalized rejection."""

    REFRESH_MARKET_AND_REVALIDATE = "REFRESH_MARKET_AND_REVALIDATE"
    RESIZE_AND_REVALIDATE = "RESIZE_AND_REVALIDATE"
    SPLIT_AND_RECONCILE = "SPLIT_AND_RECONCILE"
    RECONCILE_ORDERBOOK = "RECONCILE_ORDERBOOK"
    RETRY_WITH_BACKOFF = "RETRY_WITH_BACKOFF"
    STOP_AND_ALERT = "STOP_AND_ALERT"


@dataclass(frozen=True, slots=True)
class BrokerRejectDecision:
    """Normalized reason and constraints for a possible recovery attempt."""

    reason: BrokerReject
    retryable: bool
    action: RecoveryAction
    reconcile_before_retry: bool = False
    count_toward_kill_switch: bool = False


# Rules are intentionally ordered.  Specific trigger-price failures must be
# evaluated before generic limit-price text, and deterministic user/order
# validation failures must not be mistaken for transient broker outages.
_TRIGGER_SUBJECTS: Final[tuple[str, ...]] = (
    "trigger price",
    "stoploss trigger",
    "stop loss trigger",
    "sl trigger",
)
_TRIGGER_QUALIFIERS: Final[tuple[str, ...]] = (
    "invalid",
    "should be",
    "must be",
    "higher",
    "lower",
    "difference between",
)

_PHRASE_RULES: Final[tuple[tuple[BrokerReject, tuple[str, ...]], ...]] = (
    (
        BrokerReject.MIS_CUTOFF,
        (
            "intraday orders (mis) are allowed only till",
            "mis only till",
            "mis order is not allowed",
        ),
    ),
    (
        BrokerReject.MARKET_CLOSED,
        ("markets are closed right now", "market is closed"),
    ),
    (
        BrokerReject.INSUFFICIENT_FUNDS,
        (
            "insufficient funds",
            "required margin",
            "not enough cash",
            "not enough balance",
            "margin required",
            "available margin",
        ),
    ),
    (
        BrokerReject.FREEZE_LIMIT,
        (
            "freeze quantity",
            "freeze limit",
            "maximum allowed quantity",
            "maximum order quantity",
        ),
    ),
    (
        BrokerReject.QUANTITY_INVALID,
        (
            "invalid quantity",
            "quantity should be",
            "lot size",
            "quantity is not a multiple",
            "minimum quantity",
        ),
    ),
    (
        BrokerReject.INSTRUMENT_INVALID,
        (
            "invalid instrument",
            "instrument token",
            "unknown tradingsymbol",
            "invalid tradingsymbol",
            "contract is not enabled",
            "instrument is not tradable",
            "expired contract",
        ),
    ),
    (
        BrokerReject.DUPLICATE_ORDER,
        (
            "duplicate order",
            "duplicate request",
            "request already processed",
            "order already placed",
        ),
    ),
    (
        BrokerReject.ORDER_NOT_FOUND,
        ("order not found", "invalid order id", "unknown order"),
    ),
    (
        BrokerReject.ORDER_STATE_INVALID,
        (
            "order cannot be modified",
            "order cannot be cancelled",
            "order is complete",
            "order already complete",
            "invalid order status",
        ),
    ),
    (
        BrokerReject.THROTTLED,
        ("rate limit", "too many requests", "throttle", "http 429", "status 429"),
    ),
    (
        BrokerReject.TRANSIENT_BROKER,
        (
            "gateway timeout",
            "service unavailable",
            "temporarily unavailable",
            "connection reset",
            "connection timed out",
            "read timed out",
            "http 502",
            "http 503",
            "http 504",
            "status 502",
            "status 503",
            "status 504",
        ),
    ),
    (
        BrokerReject.AUTHENTICATION,
        (
            "token is invalid",
            "invalid access token",
            "session expired",
            "authentication failed",
            "permission denied",
            "http 403",
            "status 403",
        ),
    ),
    (
        BrokerReject.PRICE_OUT_OF_RANGE,
        (
            "price is outside the allowed range",
            "price out of range",
            "price bands",
            "circuit limit",
            "invalid limit price",
            "limit price is outside",
        ),
    ),
)

_RECOVERY_RULES: Final[
    dict[BrokerReject, tuple[bool, RecoveryAction, bool, bool]]
] = {
    BrokerReject.PRICE_OUT_OF_RANGE: (
        True,
        RecoveryAction.REFRESH_MARKET_AND_REVALIDATE,
        False,
        False,
    ),
    BrokerReject.TRIGGER_PRICE_INVALID: (
        True,
        RecoveryAction.REFRESH_MARKET_AND_REVALIDATE,
        False,
        False,
    ),
    BrokerReject.INSUFFICIENT_FUNDS: (
        True,
        RecoveryAction.RESIZE_AND_REVALIDATE,
        False,
        False,
    ),
    BrokerReject.FREEZE_LIMIT: (
        True,
        RecoveryAction.SPLIT_AND_RECONCILE,
        False,
        False,
    ),
    BrokerReject.DUPLICATE_ORDER: (
        True,
        RecoveryAction.RECONCILE_ORDERBOOK,
        True,
        False,
    ),
    BrokerReject.ORDER_NOT_FOUND: (
        True,
        RecoveryAction.RECONCILE_ORDERBOOK,
        True,
        False,
    ),
    BrokerReject.ORDER_STATE_INVALID: (
        True,
        RecoveryAction.RECONCILE_ORDERBOOK,
        True,
        False,
    ),
    BrokerReject.TRANSIENT_BROKER: (
        True,
        RecoveryAction.RECONCILE_ORDERBOOK,
        True,
        True,
    ),
    BrokerReject.THROTTLED: (
        True,
        RecoveryAction.RETRY_WITH_BACKOFF,
        False,
        False,
    ),
}


def parse_broker_error(message: str | Exception | None) -> BrokerReject:
    """Return a normalized rejection code for Zerodha-style error payloads."""

    token = " ".join(str(message or "").strip().lower().split())
    if not token:
        return BrokerReject.UNKNOWN

    if any(subject in token for subject in _TRIGGER_SUBJECTS) and any(
        qualifier in token for qualifier in _TRIGGER_QUALIFIERS
    ):
        return BrokerReject.TRIGGER_PRICE_INVALID

    if "amo" in token and ("post" in token or "outside market" in token):
        return BrokerReject.MARKET_CLOSED

    for reason, phrases in _PHRASE_RULES:
        if any(phrase in token for phrase in phrases):
            return reason

    # Bare status codes are accepted only as complete tokens.  This avoids
    # classifying order IDs or prices containing the same digits.
    words = set(token.replace(":", " ").replace("/", " ").split())
    if "429" in words:
        return BrokerReject.THROTTLED
    if words.intersection({"502", "503", "504"}):
        return BrokerReject.TRANSIENT_BROKER
    if "403" in words:
        return BrokerReject.AUTHENTICATION

    return BrokerReject.UNKNOWN


def recovery_decision(message: str | Exception | None) -> BrokerRejectDecision:
    """Return the only context-free recovery class permitted for an error.

    `retryable=True` does not authorize immediate resubmission.  The caller must
    perform the action, re-run preflight/risk checks, and use a bounded attempt
    budget.  Context-dependent product conversion is deliberately excluded.
    """

    reason = parse_broker_error(message)
    retryable, action, reconcile, count_failure = _RECOVERY_RULES.get(
        reason,
        (False, RecoveryAction.STOP_AND_ALERT, False, reason is BrokerReject.UNKNOWN),
    )
    return BrokerRejectDecision(
        reason=reason,
        retryable=retryable,
        action=action,
        reconcile_before_retry=reconcile,
        count_toward_kill_switch=count_failure,
    )


__all__ = [
    "BrokerReject",
    "BrokerRejectDecision",
    "RecoveryAction",
    "parse_broker_error",
    "recovery_decision",
]
