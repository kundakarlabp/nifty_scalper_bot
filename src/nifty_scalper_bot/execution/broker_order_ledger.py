"""Durable broker-order ledger for deterministic reconciliation.

The ledger is intentionally broker-order centric. It records broker order IDs
seen during reconciliation before they are applied to PositionManager state, so
restarts and repeated broker snapshots cannot turn the same unknown order into a
new warning or a fresh accounting attempt on every cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping
import uuid

from nifty_scalper_bot.utils.symbols import normalize_symbol

TERMINAL_STATUSES = {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
ACTIVE_STATUSES = {"PENDING", "OPEN", "PARTIALLY_FILLED"}
BLOCKING_RECONCILIATION_STATES = {
    "active_unknown_broker_order",
    "broker_state_unverified",
    "quarantined_broker_position",
}

_STATUS_MAP = {
    "SUBMITTED": "PENDING",
    "VALIDATION PENDING": "PENDING",
    "PUT ORDER REQ RECEIVED": "PENDING",
    "PUT ORDER REQUEST RECEIVED": "PENDING",
    "PENDING": "PENDING",
    "OPEN": "OPEN",
    "OPEN PENDING": "OPEN",
    "TRIGGER PENDING": "OPEN",
    "PARTIAL": "PARTIALLY_FILLED",
    "PARTIALLY FILLED": "PARTIALLY_FILLED",
    "PARTIALLY_FILLED": "PARTIALLY_FILLED",
    "COMPLETE": "FILLED",
    "COMPLETED": "FILLED",
    "FILLED": "FILLED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
}

_ORDER_ID_FIELDS = (
    "order_id",
    "broker_order_id",
    "exchange_order_id",
    "parent_order_id",
    "guid",
)
_SYMBOL_FIELDS = ("symbol", "tradingsymbol", "trading_symbol", "instrument")
_SIDE_FIELDS = ("transaction_type", "side", "order_side")
_QTY_FIELDS = (
    "quantity",
    "qty",
    "filled_quantity",
    "filled_qty",
    "filled",
    "pending_quantity",
)
_FILLED_QTY_FIELDS = ("filled_quantity", "filled_qty", "filled")
_PRICE_FIELDS = ("average_price", "avg_price", "fill_price", "price")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def normalise_broker_status(value: object) -> str:
    """Return the internal status token for a broker order status."""

    token = str(value or "").strip().upper()
    return _STATUS_MAP.get(token, token or "UNKNOWN")


def normalise_broker_order_id(
    payload: Mapping[str, Any] | None,
    fallback_order_id: object | None = None,
) -> str:
    """Extract a stable broker order ID from a broker payload."""

    if isinstance(payload, Mapping):
        for field_name in _ORDER_ID_FIELDS:
            value = payload.get(field_name)
            if value not in (None, ""):
                return str(value).strip()
    return str(fallback_order_id or "").strip()


def _first_text(payload: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field_name in fields:
        value = payload.get(field_name)
        if value not in (None, ""):
            token = str(value).strip()
            if token:
                return token
    return ""


def normalise_broker_symbol(payload: Mapping[str, Any] | None) -> str:
    """Return the canonical symbol key used by execution state."""

    if not isinstance(payload, Mapping):
        return ""
    raw_symbol = _first_text(payload, _SYMBOL_FIELDS)
    if not raw_symbol:
        return ""
    if ":" not in raw_symbol:
        exchange = str(payload.get("exchange") or "NFO").strip().upper() or "NFO"
        raw_symbol = f"{exchange}:{raw_symbol}"
    return normalize_symbol(raw_symbol)


def _normalise_side(payload: Mapping[str, Any]) -> str:
    side = _first_text(payload, _SIDE_FIELDS).upper()
    if side in {"BUY", "B"}:
        return "BUY"
    if side in {"SELL", "S"}:
        return "SELL"
    return side


def _extract_quantity(payload: Mapping[str, Any], fields: Iterable[str]) -> int:
    for field_name in fields:
        if field_name not in payload:
            continue
        qty = abs(_safe_int(payload.get(field_name), default=0))
        if qty > 0:
            return qty
    return 0


def _extract_price(payload: Mapping[str, Any]) -> float:
    for field_name in _PRICE_FIELDS:
        if field_name not in payload:
            continue
        price = _safe_float(payload.get(field_name), default=0.0)
        if price > 0:
            return price
    return 0.0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


@dataclass(slots=True)
class BrokerOrderLedgerEntry:
    order_id: str
    symbol: str = ""
    status: str = "UNKNOWN"
    side: str = ""
    quantity: int = 0
    filled_quantity: int = 0
    average_price: float = 0.0
    product: str = ""
    tag: str = ""
    client_order_id: str = ""
    source: str = "broker_reconcile"
    reconciliation_state: str = "seen"
    reason: str = ""
    first_seen_at: str = field(default_factory=_now_iso)
    last_seen_at: str = field(default_factory=_now_iso)
    seen_count: int = 1
    raw_status: str = ""
    intent: str = "UNKNOWN"

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "BrokerOrderLedgerEntry":
        return BrokerOrderLedgerEntry(
            order_id=str(payload.get("order_id") or "").strip(),
            symbol=str(payload.get("symbol") or "").strip().upper(),
            status=str(payload.get("status") or "UNKNOWN").strip().upper(),
            side=str(payload.get("side") or "").strip().upper(),
            quantity=_safe_int(payload.get("quantity"), 0),
            filled_quantity=_safe_int(payload.get("filled_quantity"), 0),
            average_price=_safe_float(payload.get("average_price"), 0.0),
            product=str(payload.get("product") or "").strip().upper(),
            tag=str(payload.get("tag") or "").strip(),
            client_order_id=str(payload.get("client_order_id") or "").strip(),
            source=str(payload.get("source") or "broker_reconcile").strip(),
            reconciliation_state=str(payload.get("reconciliation_state") or "seen").strip(),
            reason=str(payload.get("reason") or "").strip(),
            first_seen_at=str(payload.get("first_seen_at") or _now_iso()),
            last_seen_at=str(payload.get("last_seen_at") or _now_iso()),
            seen_count=max(_safe_int(payload.get("seen_count"), 1), 1),
            raw_status=str(payload.get("raw_status") or "").strip(),
            intent=str(payload.get("intent") or "UNKNOWN").strip().upper(),
        )

    @staticmethod
    def from_broker_payload(
        payload: Mapping[str, Any],
        *,
        fallback_order_id: object | None = None,
        source: str = "broker_reconcile",
    ) -> "BrokerOrderLedgerEntry":
        order_id = normalise_broker_order_id(payload, fallback_order_id)
        raw_status = str(payload.get("status") or "").strip().upper()
        return BrokerOrderLedgerEntry(
            order_id=order_id,
            symbol=normalise_broker_symbol(payload),
            status=normalise_broker_status(raw_status),
            side=_normalise_side(payload),
            quantity=_extract_quantity(payload, _QTY_FIELDS),
            filled_quantity=_extract_quantity(payload, _FILLED_QTY_FIELDS),
            average_price=_extract_price(payload),
            product=str(payload.get("product") or "").strip().upper(),
            tag=str(payload.get("tag") or "").strip(),
            client_order_id=str(payload.get("client_order_id") or payload.get("guid") or "").strip(),
            source=source,
            raw_status=raw_status,
        )

    def merge_broker_payload(self, payload: Mapping[str, Any]) -> "BrokerOrderLedgerEntry":
        incoming = BrokerOrderLedgerEntry.from_broker_payload(
            payload,
            fallback_order_id=self.order_id,
            source=self.source,
        )
        now = _now_iso()
        self.symbol = incoming.symbol or self.symbol
        self.status = incoming.status or self.status
        self.side = incoming.side or self.side
        self.quantity = incoming.quantity or self.quantity
        self.filled_quantity = incoming.filled_quantity or self.filled_quantity
        self.average_price = incoming.average_price or self.average_price
        self.product = incoming.product or self.product
        self.tag = incoming.tag or self.tag
        self.client_order_id = incoming.client_order_id or self.client_order_id
        self.raw_status = incoming.raw_status or self.raw_status
        self.last_seen_at = now
        self.seen_count += 1
        return self

    def mark(self, state: str, reason: str = "") -> "BrokerOrderLedgerEntry":
        self.reconciliation_state = str(state or "seen")
        self.reason = str(reason or "")
        self.last_seen_at = _now_iso()
        return self

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def blocks_new_entries(self) -> bool:
        return self.reconciliation_state in BLOCKING_RECONCILIATION_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "status": self.status,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "product": self.product,
            "tag": self.tag,
            "client_order_id": self.client_order_id,
            "source": self.source,
            "reconciliation_state": self.reconciliation_state,
            "reason": self.reason,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "seen_count": self.seen_count,
            "raw_status": self.raw_status,
            "intent": self.intent,
        }


class BrokerOrderLedger:
    """Durable broker-order index used by reconciliation patches."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._entries: dict[str, BrokerOrderLedgerEntry] = {}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._entries = {}
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._entries = {}
                return
            raw_entries = payload.get("orders") if isinstance(payload, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                self._entries = {}
                return
            restored: dict[str, BrokerOrderLedgerEntry] = {}
            for order_id, raw_entry in raw_entries.items():
                if not isinstance(raw_entry, Mapping):
                    continue
                entry = BrokerOrderLedgerEntry.from_dict(raw_entry)
                key = entry.order_id or str(order_id)
                if key:
                    entry.order_id = key
                    restored[key] = entry
            self._entries = restored

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "updated_at": _now_iso(),
                "orders": {
                    order_id: entry.to_dict()
                    for order_id, entry in sorted(self._entries.items())
                },
            }
        _atomic_write_json(self.path, payload)

    def get(self, order_id: object) -> BrokerOrderLedgerEntry | None:
        key = str(order_id or "").strip()
        with self._lock:
            return self._entries.get(key)

    def upsert_from_broker(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_order_id: object | None = None,
    ) -> BrokerOrderLedgerEntry:
        entry = BrokerOrderLedgerEntry.from_broker_payload(
            payload,
            fallback_order_id=fallback_order_id,
        )
        if not entry.order_id:
            raise ValueError("broker order payload missing order_id")
        with self._lock:
            existing = self._entries.get(entry.order_id)
            if existing is None:
                self._entries[entry.order_id] = entry
                current = entry
            else:
                current = existing.merge_broker_payload(payload)
        self.save()
        return current

    def upsert_local_order(
        self,
        order_id: object,
        *,
        symbol: object,
        side: object,
        quantity: object,
        price: object = 0.0,
        status: object = "PENDING",
        intent: object = "UNKNOWN",
        tag: object = "",
    ) -> BrokerOrderLedgerEntry:
        key = str(order_id or "").strip()
        if not key:
            raise ValueError("local order missing order_id")
        symbol_token = normalize_symbol(str(symbol or ""))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = BrokerOrderLedgerEntry(order_id=key, source="local_order")
                self._entries[key] = entry
            entry.symbol = symbol_token or entry.symbol
            entry.side = str(side or entry.side or "").strip().upper()
            entry.quantity = abs(_safe_int(quantity, entry.quantity))
            entry.average_price = _safe_float(price, entry.average_price)
            entry.status = normalise_broker_status(status)
            entry.intent = str(intent or entry.intent or "UNKNOWN").strip().upper()
            entry.tag = str(tag or entry.tag or "").strip()
            entry.mark("local_pending", "local_order_registered")
        self.save()
        return entry

    def mark(self, order_id: object, state: str, reason: str = "") -> BrokerOrderLedgerEntry | None:
        key = str(order_id or "").strip()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.mark(state, reason)
        self.save()
        return entry

    def entries(self) -> list[BrokerOrderLedgerEntry]:
        with self._lock:
            return list(self._entries.values())

    def blocking_entries(self, symbol: object | None = None) -> list[BrokerOrderLedgerEntry]:
        canonical_symbol = normalize_symbol(str(symbol or "")) if symbol else ""
        with self._lock:
            entries = list(self._entries.values())
        blockers: list[BrokerOrderLedgerEntry] = []
        for entry in entries:
            if not entry.blocks_new_entries:
                continue
            if canonical_symbol and entry.symbol != canonical_symbol:
                continue
            blockers.append(entry)
        return blockers

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path),
                "count": len(self._entries),
                "blocking_count": sum(
                    1 for entry in self._entries.values() if entry.blocks_new_entries
                ),
                "orders": [entry.to_dict() for entry in self._entries.values()],
            }


__all__ = [
    "ACTIVE_STATUSES",
    "BLOCKING_RECONCILIATION_STATES",
    "BrokerOrderLedger",
    "BrokerOrderLedgerEntry",
    "TERMINAL_STATUSES",
    "normalise_broker_order_id",
    "normalise_broker_status",
    "normalise_broker_symbol",
]
