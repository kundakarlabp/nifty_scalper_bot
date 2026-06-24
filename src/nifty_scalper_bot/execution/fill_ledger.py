"""Restart-safe, idempotent fill ledger for the canonical BO lifecycle.

This module is deliberately independent from broker submission and bracket
state mutation.  It stores confirmed broker fills only and provides exact,
quantity-weighted realized P&L.  Runtime wiring is a separate staged change so
that persistence and arithmetic can be proven before they influence live exits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Literal, Mapping, Sequence


FillKind = Literal["ENTRY", "EXIT"]
FillSide = Literal["BUY", "SELL"]


class FillLedgerError(RuntimeError):
    """Base error for fill-ledger persistence and integrity failures."""


class FillConflictError(FillLedgerError):
    """Raised when one broker fill identity is reused with different economics."""


class FillValidationError(FillLedgerError):
    """Raised when a fill is incomplete or economically invalid."""


@dataclass(frozen=True, slots=True)
class FillLeg:
    """One immutable confirmed broker fill or broker-confirmed aggregate fill."""

    fill_id: str
    bracket_id: str
    order_id: str
    kind: FillKind
    side: FillSide
    quantity: int
    price: float
    target: str | None = None
    reason: str | None = None
    fees: float = 0.0
    recorded_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fill_id = str(self.fill_id or "").strip()
        bracket_id = str(self.bracket_id or "").strip()
        order_id = str(self.order_id or "").strip()
        kind = str(self.kind or "").strip().upper()
        side = str(self.side or "").strip().upper()
        quantity = int(self.quantity or 0)
        price = float(self.price or 0.0)
        fees = float(self.fees or 0.0)
        recorded_at = float(self.recorded_at or 0.0)

        if not fill_id or not bracket_id or not order_id:
            raise FillValidationError("fill_id, bracket_id and order_id are required")
        if kind not in {"ENTRY", "EXIT"}:
            raise FillValidationError(f"invalid fill kind: {self.kind!r}")
        if side not in {"BUY", "SELL"}:
            raise FillValidationError(f"invalid fill side: {self.side!r}")
        if quantity <= 0:
            raise FillValidationError("fill quantity must be positive")
        if not math.isfinite(price) or price <= 0:
            raise FillValidationError("fill price must be finite and positive")
        if not math.isfinite(fees) or fees < 0:
            raise FillValidationError("fill fees must be finite and non-negative")
        if not math.isfinite(recorded_at) or recorded_at <= 0:
            raise FillValidationError("recorded_at must be finite and positive")

        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "bracket_id", bracket_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fees", fees)
        object.__setattr__(self, "recorded_at", recorded_at)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def economics(self) -> tuple[object, ...]:
        """Return immutable economic fields used for conflict detection."""

        return (
            self.fill_id,
            self.bracket_id,
            self.order_id,
            self.kind,
            self.side,
            self.quantity,
            round(self.price, 10),
            self.target,
            self.reason,
            round(self.fees, 10),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FillLeg":
        return cls(
            fill_id=str(payload.get("fill_id") or ""),
            bracket_id=str(payload.get("bracket_id") or ""),
            order_id=str(payload.get("order_id") or ""),
            kind=str(payload.get("kind") or "").upper(),  # type: ignore[arg-type]
            side=str(payload.get("side") or "").upper(),  # type: ignore[arg-type]
            quantity=int(payload.get("quantity") or 0),
            price=float(payload.get("price") or 0.0),
            target=(str(payload.get("target")) if payload.get("target") is not None else None),
            reason=(str(payload.get("reason")) if payload.get("reason") is not None else None),
            fees=float(payload.get("fees") or 0.0),
            recorded_at=float(payload.get("recorded_at") or time.time()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RealizedPnl:
    """Quantity-weighted realized P&L derived only from confirmed fills."""

    entry_side: FillSide
    entry_quantity: int
    exit_quantity: int
    entry_vwap: float
    exit_vwap: float | None
    gross_pnl: float
    fees: float
    net_pnl: float
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weighted_vwap(legs: Sequence[FillLeg]) -> float | None:
    quantity = sum(leg.quantity for leg in legs)
    if quantity <= 0:
        return None
    notional = sum(leg.quantity * leg.price for leg in legs)
    return notional / quantity


def calculate_realized_pnl(legs: Iterable[FillLeg]) -> RealizedPnl:
    """Calculate P&L from confirmed fills with strict quantity/side validation.

    Entry fills must share one side. Exit fills must be the opposite side. A
    partial exit returns ``complete=False``. Over-exits are rejected rather than
    silently creating synthetic P&L.
    """

    all_legs = list(legs)
    entries = [leg for leg in all_legs if leg.kind == "ENTRY"]
    exits = [leg for leg in all_legs if leg.kind == "EXIT"]
    if not entries:
        raise FillValidationError("at least one confirmed entry fill is required")

    entry_sides = {leg.side for leg in entries}
    if len(entry_sides) != 1:
        raise FillValidationError("entry fills must have one consistent side")
    entry_side = next(iter(entry_sides))
    exit_side = "SELL" if entry_side == "BUY" else "BUY"
    if any(leg.side != exit_side for leg in exits):
        raise FillValidationError("exit fill side does not reduce the entry exposure")

    entry_quantity = sum(leg.quantity for leg in entries)
    exit_quantity = sum(leg.quantity for leg in exits)
    if exit_quantity > entry_quantity:
        raise FillValidationError("confirmed exit quantity exceeds confirmed entry quantity")

    entry_vwap = _weighted_vwap(entries)
    if entry_vwap is None:
        raise FillValidationError("entry VWAP is unavailable")
    exit_vwap = _weighted_vwap(exits)

    if entry_side == "BUY":
        gross = sum((leg.price - entry_vwap) * leg.quantity for leg in exits)
    else:
        gross = sum((entry_vwap - leg.price) * leg.quantity for leg in exits)
    fees = sum(leg.fees for leg in all_legs)
    net = gross - fees

    return RealizedPnl(
        entry_side=entry_side,
        entry_quantity=entry_quantity,
        exit_quantity=exit_quantity,
        entry_vwap=round(entry_vwap, 8),
        exit_vwap=round(exit_vwap, 8) if exit_vwap is not None else None,
        gross_pnl=round(gross, 2),
        fees=round(fees, 2),
        net_pnl=round(net, 2),
        complete=exit_quantity == entry_quantity,
    )


class BracketFillLedgerStore:
    """SQLite-backed immutable fill ledger with idempotent inserts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bracket_fill_legs (
                        fill_id TEXT PRIMARY KEY,
                        bracket_id TEXT NOT NULL,
                        order_id TEXT NOT NULL,
                        kind TEXT NOT NULL CHECK(kind IN ('ENTRY', 'EXIT')),
                        side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                        quantity INTEGER NOT NULL CHECK(quantity > 0),
                        price REAL NOT NULL CHECK(price > 0),
                        target TEXT,
                        reason TEXT,
                        fees REAL NOT NULL DEFAULT 0 CHECK(fees >= 0),
                        recorded_at REAL NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_bracket_fill_legs_bracket
                    ON bracket_fill_legs(bracket_id, recorded_at, fill_id)
                    """
                )
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to initialize fill ledger: {exc}") from exc

    @staticmethod
    def _row_to_leg(row: sqlite3.Row) -> FillLeg:
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError:
            metadata = {}
        return FillLeg(
            fill_id=str(row["fill_id"]),
            bracket_id=str(row["bracket_id"]),
            order_id=str(row["order_id"]),
            kind=str(row["kind"]),  # type: ignore[arg-type]
            side=str(row["side"]),  # type: ignore[arg-type]
            quantity=int(row["quantity"]),
            price=float(row["price"]),
            target=str(row["target"]) if row["target"] is not None else None,
            reason=str(row["reason"]) if row["reason"] is not None else None,
            fees=float(row["fees"]),
            recorded_at=float(row["recorded_at"]),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def get_fill(self, fill_id: str) -> FillLeg | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM bracket_fill_legs WHERE fill_id = ?",
                    (str(fill_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to read fill {fill_id}: {exc}") from exc
        return self._row_to_leg(row) if row is not None else None

    def record_fill(self, leg: FillLeg) -> bool:
        """Insert one fill.

        Returns ``True`` for a new row and ``False`` for an identical replay.
        Reusing a fill ID with different economics raises ``FillConflictError``.
        """

        existing = self.get_fill(leg.fill_id)
        if existing is not None:
            if existing.economics() != leg.economics():
                raise FillConflictError(
                    f"fill_id {leg.fill_id!r} already exists with different economics"
                )
            return False

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO bracket_fill_legs (
                        fill_id, bracket_id, order_id, kind, side, quantity,
                        price, target, reason, fees, recorded_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        leg.fill_id,
                        leg.bracket_id,
                        leg.order_id,
                        leg.kind,
                        leg.side,
                        leg.quantity,
                        leg.price,
                        leg.target,
                        leg.reason,
                        leg.fees,
                        leg.recorded_at,
                        json.dumps(dict(leg.metadata), sort_keys=True, default=str),
                    ),
                )
        except sqlite3.IntegrityError:
            # Another thread/process may have inserted the same fill between the
            # read and insert. Re-read and apply the same immutable conflict rule.
            replay = self.get_fill(leg.fill_id)
            if replay is not None and replay.economics() == leg.economics():
                return False
            raise FillConflictError(
                f"fill_id {leg.fill_id!r} was concurrently written differently"
            )
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to persist fill {leg.fill_id}: {exc}") from exc
        return True

    def load_fills(self, bracket_id: str) -> list[FillLeg]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM bracket_fill_legs
                    WHERE bracket_id = ?
                    ORDER BY recorded_at ASC, fill_id ASC
                    """,
                    (str(bracket_id),),
                ).fetchall()
        except sqlite3.Error as exc:
            raise FillLedgerError(
                f"unable to load fills for bracket {bracket_id}: {exc}"
            ) from exc
        return [self._row_to_leg(row) for row in rows]

    def realized_pnl(self, bracket_id: str) -> RealizedPnl:
        return calculate_realized_pnl(self.load_fills(bracket_id))


__all__ = [
    "BracketFillLedgerStore",
    "FillConflictError",
    "FillKind",
    "FillLedgerError",
    "FillLeg",
    "FillSide",
    "FillValidationError",
    "RealizedPnl",
    "calculate_realized_pnl",
]
