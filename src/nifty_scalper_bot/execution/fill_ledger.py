"""Restart-safe, idempotent fill ledger for the canonical BO lifecycle."""

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
    """Base persistence or integrity error."""


class FillConflictError(FillLedgerError):
    """A broker fill identity was reused with different economics."""


class FillValidationError(FillLedgerError):
    """A fill or fill set is economically invalid."""


@dataclass(frozen=True, slots=True)
class FillLeg:
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
            raise FillValidationError("fees must be finite and non-negative")
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


@dataclass(frozen=True, slots=True)
class RealizedPnl:
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


def _vwap(legs: Sequence[FillLeg]) -> float | None:
    quantity = sum(leg.quantity for leg in legs)
    if quantity <= 0:
        return None
    return sum(leg.quantity * leg.price for leg in legs) / quantity


def calculate_realized_pnl(legs: Iterable[FillLeg]) -> RealizedPnl:
    all_legs = list(legs)
    entries = [leg for leg in all_legs if leg.kind == "ENTRY"]
    exits = [leg for leg in all_legs if leg.kind == "EXIT"]
    if not entries:
        raise FillValidationError("at least one confirmed entry fill is required")
    entry_sides = {leg.side for leg in entries}
    if len(entry_sides) != 1:
        raise FillValidationError("entry fills must have one consistent side")
    entry_side = next(iter(entry_sides))
    reducing_side = "SELL" if entry_side == "BUY" else "BUY"
    if any(leg.side != reducing_side for leg in exits):
        raise FillValidationError("exit side does not reduce the entry exposure")
    entry_quantity = sum(leg.quantity for leg in entries)
    exit_quantity = sum(leg.quantity for leg in exits)
    if exit_quantity > entry_quantity:
        raise FillValidationError("confirmed exit quantity exceeds confirmed entry quantity")
    entry_vwap = _vwap(entries)
    if entry_vwap is None:
        raise FillValidationError("entry VWAP unavailable")
    exit_vwap = _vwap(exits)
    if entry_side == "BUY":
        gross = sum((leg.price - entry_vwap) * leg.quantity for leg in exits)
    else:
        gross = sum((entry_vwap - leg.price) * leg.quantity for leg in exits)
    fees = sum(leg.fees for leg in all_legs)
    return RealizedPnl(
        entry_side=entry_side,
        entry_quantity=entry_quantity,
        exit_quantity=exit_quantity,
        entry_vwap=round(entry_vwap, 8),
        exit_vwap=round(exit_vwap, 8) if exit_vwap is not None else None,
        gross_pnl=round(gross, 2),
        fees=round(fees, 2),
        net_pnl=round(gross - fees, 2),
        complete=exit_quantity == entry_quantity,
    )


class BracketFillLedgerStore:
    """SQLite-backed immutable fill ledger with replay protection."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
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
                        kind TEXT NOT NULL CHECK(kind IN ('ENTRY','EXIT')),
                        side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
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
                    "CREATE INDEX IF NOT EXISTS idx_fill_bracket ON bracket_fill_legs(bracket_id, recorded_at, fill_id)"
                )
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to initialize fill ledger: {exc}") from exc

    @staticmethod
    def _from_row(row: sqlite3.Row) -> FillLeg:
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
                    "SELECT * FROM bracket_fill_legs WHERE fill_id = ?", (str(fill_id),)
                ).fetchone()
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to read fill {fill_id}: {exc}") from exc
        return self._from_row(row) if row is not None else None

    def record_fill(self, leg: FillLeg) -> bool:
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
            replay = self.get_fill(leg.fill_id)
            if replay is not None and replay.economics() == leg.economics():
                return False
            raise FillConflictError(f"fill_id {leg.fill_id!r} was concurrently changed")
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to persist fill {leg.fill_id}: {exc}") from exc
        return True

    def load_fills(self, bracket_id: str) -> list[FillLeg]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM bracket_fill_legs WHERE bracket_id = ? ORDER BY recorded_at, fill_id",
                    (str(bracket_id),),
                ).fetchall()
        except sqlite3.Error as exc:
            raise FillLedgerError(f"unable to load fills for {bracket_id}: {exc}") from exc
        return [self._from_row(row) for row in rows]

    def realized_pnl(self, bracket_id: str) -> RealizedPnl:
        return calculate_realized_pnl(self.load_fills(bracket_id))


__all__ = [
    "BracketFillLedgerStore",
    "FillConflictError",
    "FillLedgerError",
    "FillLeg",
    "FillValidationError",
    "RealizedPnl",
    "calculate_realized_pnl",
]
