"""Canonical broker order-status mapping for execution reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DomainOrderState(Enum):
    SUBMISSION_PENDING = "submission_pending"
    OPEN = "open"
    TRIGGER_PENDING = "trigger_pending"
    CANCEL_PENDING = "cancel_pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


_STATUS_MAP = {
    "PUT ORDER REQ RECEIVED": DomainOrderState.SUBMISSION_PENDING,
    "PUT ORDER REQUEST RECEIVED": DomainOrderState.SUBMISSION_PENDING,
    "VALIDATION PENDING": DomainOrderState.SUBMISSION_PENDING,
    "OPEN PENDING": DomainOrderState.SUBMISSION_PENDING,
    "OPEN": DomainOrderState.OPEN,
    "TRIGGER PENDING": DomainOrderState.TRIGGER_PENDING,
    "MODIFY VALIDATION PENDING": DomainOrderState.OPEN,
    "MODIFY PENDING": DomainOrderState.OPEN,
    "CANCEL PENDING": DomainOrderState.CANCEL_PENDING,
    "COMPLETE": DomainOrderState.FILLED,
    "COMPLETED": DomainOrderState.FILLED,
    "FILLED": DomainOrderState.FILLED,
    "REJECTED": DomainOrderState.REJECTED,
    "CANCELLED": DomainOrderState.CANCELLED,
    "CANCELED": DomainOrderState.CANCELLED,
}


@dataclass(frozen=True)
class BrokerOrderStatusEvidence:
    state: DomainOrderState
    raw_status: str


def map_broker_order_status(status: object) -> BrokerOrderStatusEvidence:
    raw = str(status or "").strip().upper()
    return BrokerOrderStatusEvidence(
        _STATUS_MAP.get(raw, DomainOrderState.UNKNOWN), raw
    )
