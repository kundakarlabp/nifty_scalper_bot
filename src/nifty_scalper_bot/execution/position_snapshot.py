"""File purpose:
    Decode broker position responses into one authoritative exposure snapshot.

Key responsibilities:
    - Reject missing, malformed, partial, cached, or ambiguous exposure data.
    - Normalize symbols and net quantities once for every execution subsystem.
    - Provide deterministic symbol and account-flat queries.

Operational constraints:
    - Only a valid complete snapshot may prove flatness.
    - ``day`` rows and stale cache payloads are never exposure authority.
    - Malformed rows invalidate the complete snapshot rather than being skipped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

from nifty_scalper_bot.utils.symbols import normalize_symbol


class PositionSnapshotError(ValueError):
    """Raised when broker exposure cannot be interpreted authoritatively."""


@dataclass(frozen=True, slots=True)
class PositionSnapshotRow:
    """One validated broker net-position row."""

    symbol: str
    quantity: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Validated complete broker exposure snapshot."""

    rows: tuple[PositionSnapshotRow, ...]
    source: str
    fetched_at: float

    def quantity_for(self, symbol: str) -> int:
        """Return signed net quantity for ``symbol``."""
        target = normalize_symbol(symbol)
        for row in self.rows:
            if row.symbol == target:
                return row.quantity
        return 0

    @property
    def all_flat(self) -> bool:
        """Return whether every validated row is flat."""
        return all(row.quantity == 0 for row in self.rows)

    def raw_rows(self) -> list[dict[str, Any]]:
        """Return detached rows for existing reconciliation code."""
        return [dict(row.raw) for row in self.rows]


_SYMBOL_KEYS = ("tradingsymbol", "symbol", "instrument")
_QUANTITY_KEYS = ("quantity", "net_quantity", "net_qty", "netQuantity", "net")


def _extract_rows(payload: object) -> tuple[list[object], str]:
    if payload is None:
        raise PositionSnapshotError("position snapshot is missing")

    if isinstance(payload, Mapping):
        if "net" in payload:
            rows = payload.get("net")
            source = "net"
        elif "positions" in payload:
            rows = payload.get("positions")
            source = "positions"
        elif any(key in payload for key in _SYMBOL_KEYS):
            rows = [payload]
            source = "single_row"
        else:
            raise PositionSnapshotError(
                "position snapshot has no authoritative net collection"
            )
    else:
        rows = payload
        source = "sequence"

    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise PositionSnapshotError("position snapshot collection is malformed")
    if not isinstance(rows, Iterable):
        raise PositionSnapshotError("position snapshot collection is not iterable")
    try:
        return list(rows), source
    except TypeError as exc:
        raise PositionSnapshotError(
            "position snapshot collection is not iterable"
        ) from exc


def _parse_quantity(row: Mapping[str, Any], index: int) -> int:
    key = next((name for name in _QUANTITY_KEYS if name in row), None)
    if key is None:
        raise PositionSnapshotError(f"position row {index} has no net quantity")
    value = row.get(key)
    if value is None or isinstance(value, bool):
        raise PositionSnapshotError(f"position row {index} has invalid net quantity")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PositionSnapshotError(
            f"position row {index} has invalid net quantity {value!r}"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise PositionSnapshotError(
            f"position row {index} has non-integral net quantity {value!r}"
        )
    return int(numeric)


def decode_position_snapshot(payload: object) -> PositionSnapshot:
    """Validate and normalize a complete broker net-position snapshot."""
    raw_rows, source = _extract_rows(payload)
    validated: list[PositionSnapshotRow] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_rows):
        if not isinstance(item, Mapping):
            raise PositionSnapshotError(f"position row {index} is not an object")
        raw_symbol = next(
            (item.get(key) for key in _SYMBOL_KEYS if item.get(key)), None
        )
        if raw_symbol is None:
            raise PositionSnapshotError(f"position row {index} has no symbol")
        symbol = normalize_symbol(str(raw_symbol))
        if not symbol:
            raise PositionSnapshotError(f"position row {index} has an empty symbol")
        if symbol in seen:
            raise PositionSnapshotError(f"duplicate broker position row for {symbol}")
        seen.add(symbol)
        validated.append(
            PositionSnapshotRow(
                symbol=symbol,
                quantity=_parse_quantity(item, index),
                raw=dict(item),
            )
        )
    return PositionSnapshot(
        rows=tuple(validated),
        source=source,
        fetched_at=time.time(),
    )


__all__ = [
    "PositionSnapshot",
    "PositionSnapshotError",
    "PositionSnapshotRow",
    "decode_position_snapshot",
]
