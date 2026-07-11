"""Typed broker-position evidence for fail-closed reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


def _row_qty(row: Mapping[str, Any]) -> int:
    value = row.get("quantity", row.get("net_quantity", row.get("net", 0)))
    return int(float(value or 0))


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
    net = 0
    matched = False
    try:
        for row in rows:
            row_symbol = normalize_symbol(_row_symbol(row))
            if row_symbol != wanted:
                # Match Kite no-exchange tradingsymbols to NFO-prefixed locals.
                if normalize_symbol(f"NFO:{_row_symbol(row)}") != wanted:
                    continue
            matched = True
            net += _row_qty(row)
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
            BrokerPositionState.FLAT_CONFIRMED, wanted, 0, now, 0.0, source
        )
    state = (
        BrokerPositionState.FLAT_CONFIRMED
        if net == 0
        else BrokerPositionState.NON_FLAT_CONFIRMED
    )
    return BrokerPositionEvidence(state, wanted, net, now, 0.0, source)
