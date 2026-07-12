"""Typed broker-position evidence for fail-closed reconciliation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.utils.symbols import normalize_symbol


class BrokerPositionState(Enum):
    FLAT_CONFIRMED = "flat_confirmed"
    NON_FLAT_CONFIRMED = "non_flat_confirmed"
    UNKNOWN = "unknown"
    STALE = "stale"
    API_ERROR = "api_error"
    SYMBOL_UNRESOLVED = "symbol_unresolved"


@dataclass(frozen=True)
class BrokerPositionEvidence:
    state: BrokerPositionState
    symbol: str
    net_quantity: int | None
    fetched_at: datetime
    age_seconds: float
    source: str
    error: str | None = None


def _row_symbol(row: Mapping[str, Any]) -> str:
    return str(
        row.get("tradingsymbol") or row.get("symbol") or row.get("instrument") or ""
    )


def normalize_authoritative_quantity(value: object) -> int | None:
    """Return an integer quantity only when broker evidence is unambiguous."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = Decimal(text)
        except InvalidOperation:
            return None
        if not parsed.is_finite() or parsed != parsed.to_integral_value():
            return None
        return int(parsed)
    return None


def _row_qty(row: Mapping[str, Any]) -> int | None:
    value = row.get("quantity", row.get("net_quantity", row.get("net", 0)))
    return normalize_authoritative_quantity(value)


def evidence_from_positions(
    symbol: str, rows: Iterable[Mapping[str, Any]], *, source: str = "broker.positions"
) -> BrokerPositionEvidence:
    now = datetime.now(timezone.utc)
    wanted = normalize_symbol(symbol)
    if not wanted:
        return BrokerPositionEvidence(
            BrokerPositionState.SYMBOL_UNRESOLVED,
            symbol,
            None,
            now,
            0.0,
            source,
            "empty_symbol",
        )
    try:
        position_rows = list(rows)
    except Exception as exc:  # noqa: BLE001 - broker payload materialization boundary
        return BrokerPositionEvidence(
            BrokerPositionState.API_ERROR,
            wanted,
            None,
            now,
            0.0,
            source,
            f"positions_iter_failed:{type(exc).__name__}",
        )
    if not position_rows:
        return BrokerPositionEvidence(
            BrokerPositionState.FLAT_CONFIRMED, wanted, 0, now, 0.0, source
        )

    net = 0
    matched = False
    try:
        for row in position_rows:
            if not isinstance(row, Mapping):
                raise TypeError("position row is not a mapping")
            raw_symbol = _row_symbol(row)
            row_symbol = normalize_symbol(raw_symbol)
            if row_symbol != wanted:
                # Match Kite no-exchange tradingsymbols to NFO-prefixed locals.
                if normalize_symbol(f"NFO:{raw_symbol}") != wanted:
                    continue
            parsed_qty = _row_qty(row)
            if parsed_qty is None:
                raise ValueError("position quantity is not an integer")
            matched = True
            net += parsed_qty
    except (TypeError, ValueError) as exc:
        return BrokerPositionEvidence(
            BrokerPositionState.UNKNOWN,
            wanted,
            None,
            now,
            0.0,
            source,
            f"malformed_positions:{type(exc).__name__}",
        )
    if not matched:
        return BrokerPositionEvidence(
            BrokerPositionState.SYMBOL_UNRESOLVED,
            wanted,
            None,
            now,
            0.0,
            source,
            "symbol_not_found_in_non_empty_positions",
        )
    state = (
        BrokerPositionState.FLAT_CONFIRMED
        if net == 0
        else BrokerPositionState.NON_FLAT_CONFIRMED
    )
    return BrokerPositionEvidence(state, wanted, net, now, 0.0, source)
