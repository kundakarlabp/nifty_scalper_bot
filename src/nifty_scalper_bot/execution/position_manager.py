"""Position and order state tracking for the scalper bot."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    cast,
)
from zoneinfo import ZoneInfo

from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.reasons import canonical
from nifty_scalper_bot.utils.symbols import is_strategy_instrument

if TYPE_CHECKING:
    from nifty_scalper_bot.data.persistent_state import PersistentStateManager

Side = Literal["LONG", "SHORT"]
OrderSide = Literal["BUY", "SELL"]
OrderIntent = Literal[
    "ENTRY",
    "SCALE_IN",
    "EXIT",
    "REDUCE",
    "REVERSAL",
    "UNKNOWN",
]
OrderStatus = Literal[
    "PENDING",
    "OPEN",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
]

_MIN_RECONCILE_DELAY_S = 0.5


_POSITION_RECONCILE_EVENTS = Counter(
    "position_reconcile_events_total",
    "Position reconciliation outcomes by result",
    ["result"],
)


def _normalize_side(value: str) -> Side:
    normalized = value.upper()
    if normalized not in ("LONG", "SHORT"):
        raise ValueError(f"Unsupported side '{value}'")
    return cast(Side, normalized)


def _normalize_order_side(value: str) -> OrderSide:
    normalized = value.upper()
    if normalized not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported order side '{value}'")
    return cast(OrderSide, normalized)


def _normalize_status(value: str) -> OrderStatus:
    normalized = value.upper()
    if normalized not in (
        "PENDING",
        "OPEN",
        "PARTIALLY_FILLED",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
    ):
        raise ValueError(f"Unsupported status '{value}'")
    return cast(OrderStatus, normalized)


def normalize_broker_order_status(value: object) -> OrderStatus | None:
    """Map broker-specific order statuses to the internal lifecycle states."""

    if value is None:
        return None
    normalized = str(value).strip().upper()
    mapping: dict[str, OrderStatus] = {
        "SUBMITTED": "PENDING",
        "VALIDATION PENDING": "PENDING",
        "PUT ORDER REQ RECEIVED": "PENDING",
        "PUT ORDER REQUEST RECEIVED": "PENDING",
        "OPEN": "OPEN",
        "OPEN PENDING": "OPEN",
        "TRIGGER PENDING": "OPEN",
        "PARTIALLY FILLED": "PARTIALLY_FILLED",
        "PARTIAL": "PARTIALLY_FILLED",
        "COMPLETE": "FILLED",
        "FILLED": "FILLED",
        "CANCELLED": "CANCELLED",
        "CANCELED": "CANCELLED",
        "REJECTED": "REJECTED",
        "EXPIRED": "EXPIRED",
        "PENDING": "PENDING",
    }
    return mapping.get(normalized)


def _normalize_intent(value: object | None) -> OrderIntent:
    normalized = str(value or "UNKNOWN").strip().upper()
    if normalized in {"ENTRY", "SCALE_IN", "EXIT", "REDUCE", "REVERSAL", "UNKNOWN"}:
        return cast(OrderIntent, normalized)
    return "UNKNOWN"

def _to_int(value: object) -> int:
    """Robust integer conversion handling None and strings."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        if not value.strip():
            return 0
        try:
            # Handle "100.0" strings which int() rejects directly
            return int(float(value))
        except (ValueError, TypeError):
            pass
    raise TypeError(f"Unable to convert value {value!r} to int")


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unable to convert value {value!r} to float")


def _to_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return _to_float(value)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist *payload* to *path* atomically with Enum handling (Thread-Safe).

    Args:
        path: Destination filesystem path for the JSON document.
        payload: Mapping to serialise as JSON.

    Returns:
        None.

    Raises:
        None.
        
    ✅ PRODUCTION FIX: Added Enum, datetime, Decimal serialization support.
    """
    import json
    import os
    import uuid
    from contextlib import suppress
    from enum import Enum
    from datetime import datetime, date
    from decimal import Decimal

    # ✅ FIX 1: Custom JSON encoder for Enum, datetime, Decimal
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value if hasattr(obj, 'value') else obj.name
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, '__dict__') and not isinstance(obj, type):
                return obj.__dict__
            return super().default(obj)

    # ✅ FIX 2: Sanitize payload recursively before serialization
    def _sanitize(obj):
        """Recursively convert non-JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_sanitize(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value if hasattr(obj, 'value') else obj.name
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return _sanitize(obj.to_dict())
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return _sanitize(vars(obj))
        return obj

    # Pre-sanitize the payload
    sanitized_payload = _sanitize(dict(payload))

    # Use unique temp identifier per write to prevent Thread Race Conditions
    temp_path = path.with_suffix(f"{path.suffix}.tmp.{uuid.uuid4().hex}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to unique temp file with custom encoder + default=str fallback
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(sanitized_payload, f, indent=2, sort_keys=True, 
                      cls=EnhancedJSONEncoder, default=str)
            f.flush()
            os.fsync(f.fileno())  # Force flush to disk for durability

        # Atomic Move (Overwrite destination)
        os.replace(temp_path, path)

    except Exception as exc:  # noqa: BLE001
        # Cleanup unique temp file on failure to avoid disk clutter
        with suppress(OSError):
            if temp_path.exists():
                os.remove(temp_path)
        
        get_logger(__name__).error("Failure in _atomic_write_json: %s", exc)
        raise



@dataclass(slots=True)
class Position:
    """Represents an open position managed by the bot."""

    symbol: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_distance: float | None = None
    order_id: str | None = None
    realized_pnl: float = 0.0
    state: str | None = None  # intent: track lifecycle overrides like force-closed SL

    @property
    def unrealized_pnl(self) -> float:
        """Return the unrealised profit or loss for the position."""

        direction = 1 if self.side == "LONG" else -1
        return (self.current_price - self.entry_price) * self.quantity * direction

    @property
    def unrealized_pnl_pct(self) -> float:
        """Return the unrealised profit or loss as a percentage of entry notional."""

        # Direction logic is correct
        direction = 1 if self.side == "LONG" else -1
        
        # FIX: Guard against division by zero
        notional = abs(self.entry_price * self.quantity)
        if notional == 0:
            return 0.0
            
        return (self.unrealized_pnl / notional) * 100.0

    @property
    def age_seconds(self) -> float:
        """Return the duration the position has been open in seconds."""

        return max((_now() - self.entry_time).total_seconds(), 0.0)

    def to_dict(self) -> dict[str, object]:
        """Serialize the position for JSON persistence."""

        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_distance": self.trailing_stop_distance,
            "order_id": self.order_id,
            "realized_pnl": self.realized_pnl,
            "state": self.state,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Position":
        """Create a :class:`Position` from serialized state."""

        return Position(
            symbol=str(payload["symbol"]),
            side=_normalize_side(str(payload["side"])),
            quantity=_to_int(payload["quantity"]),
            entry_price=_to_float(payload["entry_price"]),
            entry_time=datetime.fromisoformat(str(payload["entry_time"])),
            current_price=_to_float(payload["current_price"]),
            stop_loss=_to_optional_float(payload.get("stop_loss")),
            take_profit=_to_optional_float(payload.get("take_profit")),
            trailing_stop_distance=_to_optional_float(
                payload.get("trailing_stop_distance")
            ),
            order_id=(
                str(payload["order_id"])
                if payload.get("order_id") is not None
                else None
            ),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            state=str(payload.get("state")) if payload.get("state") is not None else None,
        )


@dataclass(slots=True)
class TerminalOrderMetadata:
    """Durable idempotency record for terminal broker updates."""

    terminal_at: datetime
    normalized_status: OrderStatus
    cumulative_filled_quantity: int
    average_fill_price: float | None
    lifecycle_applied: bool
    accounting_finalized: bool
    terminal_update_seen: bool = True
    fill_recorded: bool = False
    position_applied: bool = False
    bracket_applied: bool = False
    pnl_applied: bool = False
    lifecycle_resolved: bool = False
    symbol: str | None = None
    intent: OrderIntent = "UNKNOWN"
    side: OrderSide | None = None
    trade_lifecycle_id: str | None = None
    linked_entry_order_id: str | None = None
    exit_lifecycle_state: str | None = None
    protected_quantity: int = 0
    protection_confirmed: bool = False
    protection_confirmed_at: datetime | None = None
    protection_failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "terminal_at": self.terminal_at.isoformat(),
            "normalized_status": self.normalized_status,
            "cumulative_filled_quantity": self.cumulative_filled_quantity,
            "average_fill_price": self.average_fill_price,
            "lifecycle_applied": self.lifecycle_applied,
            "accounting_finalized": self.accounting_finalized,
            "terminal_update_seen": self.terminal_update_seen,
            "fill_recorded": self.fill_recorded,
            "position_applied": self.position_applied,
            "bracket_applied": self.bracket_applied,
            "pnl_applied": self.pnl_applied,
            "lifecycle_resolved": self.lifecycle_resolved,
            "symbol": self.symbol,
            "intent": self.intent,
            "side": self.side,
            "trade_lifecycle_id": self.trade_lifecycle_id,
            "linked_entry_order_id": self.linked_entry_order_id,
            "exit_lifecycle_state": self.exit_lifecycle_state,
            "protected_quantity": self.protected_quantity,
            "protection_confirmed": self.protection_confirmed,
            "protection_confirmed_at": (
                self.protection_confirmed_at.isoformat()
                if self.protection_confirmed_at
                else None
            ),
            "protection_failure_reason": self.protection_failure_reason,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TerminalOrderMetadata":
        return TerminalOrderMetadata(
            terminal_at=datetime.fromisoformat(str(payload["terminal_at"])),
            normalized_status=(
                normalize_broker_order_status(payload.get("normalized_status"))
                or _normalize_status(str(payload["normalized_status"]))
            ),
            cumulative_filled_quantity=_to_int(
                payload.get("cumulative_filled_quantity", 0)
            ),
            average_fill_price=_to_optional_float(payload.get("average_fill_price")),
            lifecycle_applied=bool(payload.get("lifecycle_applied", False)),
            accounting_finalized=bool(payload.get("accounting_finalized", False)),
            terminal_update_seen=bool(payload.get("terminal_update_seen", True)),
            fill_recorded=bool(payload.get("fill_recorded", False)),
            position_applied=bool(payload.get("position_applied", False)),
            bracket_applied=bool(payload.get("bracket_applied", False)),
            pnl_applied=bool(payload.get("pnl_applied", False)),
            lifecycle_resolved=bool(payload.get("lifecycle_resolved", False)),
            symbol=(
                str(payload["symbol"]) if payload.get("symbol") is not None else None
            ),
            intent=_normalize_intent(payload.get("intent")),
            side=(
                _normalize_order_side(str(payload["side"]))
                if payload.get("side") is not None
                else None
            ),
            trade_lifecycle_id=(
                str(payload["trade_lifecycle_id"])
                if payload.get("trade_lifecycle_id") is not None
                else None
            ),
            linked_entry_order_id=(
                str(payload["linked_entry_order_id"])
                if payload.get("linked_entry_order_id") is not None
                else None
            ),
            exit_lifecycle_state=(
                str(payload["exit_lifecycle_state"])
                if payload.get("exit_lifecycle_state") is not None
                else None
            ),
            protected_quantity=_to_int(payload.get("protected_quantity", 0)),
            protection_confirmed=bool(payload.get("protection_confirmed", False)),
            protection_confirmed_at=(
                datetime.fromisoformat(str(payload["protection_confirmed_at"]))
                if payload.get("protection_confirmed_at") is not None
                else None
            ),
            protection_failure_reason=(
                str(payload["protection_failure_reason"])
                if payload.get("protection_failure_reason") is not None
                else None
            ),
        )


@dataclass(slots=True)
class FillApplicationResult:
    """Explicit result of applying one broker fill delta."""

    fill_recorded: bool = False
    position_applied: bool = False
    bracket_applied: bool = False
    pnl_applied: bool = False
    accounting_finalized: bool = False
    lifecycle_resolved: bool = False
    quantity_delta: int = 0
    delta_fill_price: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class ExitLifecycleRecord:
    """Durable per-symbol EXIT/REDUCE lifecycle tombstone."""

    symbol: str
    exit_order_id: str
    linked_entry_order_id: str | None
    trade_lifecycle_id: str | None
    bracket_id: str | None
    expected_exit_side: OrderSide
    expected_exit_quantity: int
    state: str = "EXIT_PENDING"
    submitted_at: datetime = field(default_factory=_now)
    broker_flat_at: datetime | None = None
    final_fill_price: float | None = None
    finalized_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "exit_order_id": self.exit_order_id,
            "linked_entry_order_id": self.linked_entry_order_id,
            "trade_lifecycle_id": self.trade_lifecycle_id,
            "bracket_id": self.bracket_id,
            "expected_exit_side": self.expected_exit_side,
            "expected_exit_quantity": self.expected_exit_quantity,
            "state": self.state,
            "submitted_at": self.submitted_at.isoformat(),
            "broker_flat_at": self.broker_flat_at.isoformat() if self.broker_flat_at else None,
            "final_fill_price": self.final_fill_price,
            "finalized_at": self.finalized_at.isoformat() if self.finalized_at else None,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ExitLifecycleRecord":
        return ExitLifecycleRecord(
            symbol=str(payload["symbol"]),
            exit_order_id=str(payload["exit_order_id"]),
            linked_entry_order_id=(
                str(payload["linked_entry_order_id"])
                if payload.get("linked_entry_order_id") is not None
                else None
            ),
            trade_lifecycle_id=(
                str(payload["trade_lifecycle_id"])
                if payload.get("trade_lifecycle_id") is not None
                else None
            ),
            bracket_id=(
                str(payload["bracket_id"])
                if payload.get("bracket_id") is not None
                else None
            ),
            expected_exit_side=_normalize_order_side(str(payload["expected_exit_side"])),
            expected_exit_quantity=_to_int(payload.get("expected_exit_quantity", 0)),
            state=str(payload.get("state") or "EXIT_PENDING"),
            submitted_at=datetime.fromisoformat(str(payload["submitted_at"])),
            broker_flat_at=(
                datetime.fromisoformat(str(payload["broker_flat_at"]))
                if payload.get("broker_flat_at") is not None
                else None
            ),
            final_fill_price=_to_optional_float(payload.get("final_fill_price")),
            finalized_at=(
                datetime.fromisoformat(str(payload["finalized_at"]))
                if payload.get("finalized_at") is not None
                else None
            ),
        )


@dataclass(slots=True)
class Order:
    """Represents a broker order tracked by the manager."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: str
    quantity: int
    price: float
    status: OrderStatus
    timestamp: datetime = field(default_factory=_now)
    filled_quantity: int = 0
    fill_price: float | None = None
    linked_position_symbol: str | None = None
    intent: OrderIntent = "UNKNOWN"
    bracket_id: str | None = None
    signal_id: str | None = None
    signal_fingerprint: str | None = None
    pre_order_position_side: Side | None = None
    pre_order_quantity: int = 0
    terminal_at: datetime | None = None
    applied_filled_quantity: int = 0
    applied_cumulative_notional: float = 0.0
    last_cumulative_average_price: float | None = None
    trade_lifecycle_id: str | None = None
    linked_entry_order_id: str | None = None
    pre_order_entry_price: float | None = None
    protected_quantity: int = 0
    protection_confirmed: bool = False
    protection_confirmed_at: datetime | None = None
    protection_failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the order for JSON persistence."""

        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "filled_quantity": self.filled_quantity,
            "fill_price": self.fill_price,
            "linked_position_symbol": self.linked_position_symbol,
            "intent": self.intent,
            "bracket_id": self.bracket_id,
            "signal_id": self.signal_id,
            "signal_fingerprint": self.signal_fingerprint,
            "pre_order_position_side": self.pre_order_position_side,
            "pre_order_quantity": self.pre_order_quantity,
            "terminal_at": self.terminal_at.isoformat() if self.terminal_at else None,
            "applied_filled_quantity": self.applied_filled_quantity,
            "applied_cumulative_notional": self.applied_cumulative_notional,
            "last_cumulative_average_price": self.last_cumulative_average_price,
            "trade_lifecycle_id": self.trade_lifecycle_id,
            "linked_entry_order_id": self.linked_entry_order_id,
            "pre_order_entry_price": self.pre_order_entry_price,
            "protected_quantity": self.protected_quantity,
            "protection_confirmed": self.protection_confirmed,
            "protection_confirmed_at": (
                self.protection_confirmed_at.isoformat()
                if self.protection_confirmed_at
                else None
            ),
            "protection_failure_reason": self.protection_failure_reason,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Order":
        """Create an :class:`Order` from serialized state."""

        return Order(
            order_id=str(payload["order_id"]),
            symbol=str(payload["symbol"]),
            side=_normalize_order_side(str(payload["side"])),
            order_type=str(payload["order_type"]),
            quantity=_to_int(payload["quantity"]),
            price=_to_float(payload["price"]),
            status=normalize_broker_order_status(payload.get("status"))
            or _normalize_status(str(payload["status"])),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            filled_quantity=_to_int(payload.get("filled_quantity", 0)),
            fill_price=_to_optional_float(payload.get("fill_price")),
            linked_position_symbol=(
                str(payload["linked_position_symbol"])
                if payload.get("linked_position_symbol") is not None
                else None
            ),
            intent=_normalize_intent(payload.get("intent")),
            bracket_id=(
                str(payload["bracket_id"])
                if payload.get("bracket_id") is not None
                else None
            ),
            signal_id=(
                str(payload["signal_id"])
                if payload.get("signal_id") is not None
                else None
            ),
            signal_fingerprint=(
                str(payload["signal_fingerprint"])
                if payload.get("signal_fingerprint") is not None
                else None
            ),
            pre_order_position_side=(
                _normalize_side(str(payload["pre_order_position_side"]))
                if payload.get("pre_order_position_side") is not None
                else None
            ),
            pre_order_quantity=_to_int(payload.get("pre_order_quantity", 0)),
            terminal_at=(
                datetime.fromisoformat(str(payload["terminal_at"]))
                if payload.get("terminal_at") is not None
                else None
            ),
            applied_filled_quantity=_to_int(payload.get("applied_filled_quantity", 0)),
            applied_cumulative_notional=_to_float(
                payload.get("applied_cumulative_notional", 0.0)
            ),
            last_cumulative_average_price=_to_optional_float(
                payload.get("last_cumulative_average_price")
            ),
            trade_lifecycle_id=(
                str(payload["trade_lifecycle_id"])
                if payload.get("trade_lifecycle_id") is not None
                else None
            ),
            linked_entry_order_id=(
                str(payload["linked_entry_order_id"])
                if payload.get("linked_entry_order_id") is not None
                else None
            ),
            pre_order_entry_price=_to_optional_float(payload.get("pre_order_entry_price")),
            protected_quantity=_to_int(payload.get("protected_quantity", 0)),
            protection_confirmed=bool(payload.get("protection_confirmed", False)),
            protection_confirmed_at=(
                datetime.fromisoformat(str(payload["protection_confirmed_at"]))
                if payload.get("protection_confirmed_at") is not None
                else None
            ),
            protection_failure_reason=(
                str(payload["protection_failure_reason"])
                if payload.get("protection_failure_reason") is not None
                else None
            ),
        )


@dataclass(slots=True)
class ActiveContract:
    """Persist underlying-to-contract association for reuse.

    Args:
        underlying: Canonical underlying symbol.
        symbol: Option contract identifier.
        option_type: Option type, typically ``CE`` or ``PE``.
        strike: Strike price for the contract.
        expiry: Contract expiry timestamp.

    Returns:
        None.

    Raises:
        None.
    """

    underlying: str
    symbol: str
    option_type: str
    strike: float
    expiry: datetime

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract mapping for persistence.

        Args:
            None.

        Returns:
            Dictionary payload for JSON storage.

        Raises:
            None.
        """

        return {
            "underlying": self.underlying,
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ActiveContract":
        """Hydrate :class:`ActiveContract` from saved state.

        Args:
            payload: Serialized contract mapping.

        Returns:
            Active contract populated from payload.

        Raises:
            ValueError: If expiry is missing or invalid.
        """

        expiry_raw = payload.get("expiry")
        if isinstance(expiry_raw, str):
            expiry_dt = datetime.fromisoformat(expiry_raw)
        elif isinstance(expiry_raw, datetime):
            expiry_dt = expiry_raw
        else:
            raise ValueError("expiry missing for active contract")
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        return ActiveContract(
            underlying=str(payload["underlying"]).strip().upper(),
            symbol=str(payload["symbol"]).strip().upper(),
            option_type=str(payload["option_type"]).strip().upper(),
            strike=float(payload["strike"]),
            expiry=expiry_dt,
        )

    @staticmethod
    def from_selection(
        underlying: str, contract: "SelectedContract | ActiveContract"
    ) -> "ActiveContract":
        """Coerce selector result into an active contract mapping.

        Args:
            underlying: Canonical underlying symbol.
            contract: Selection returned by the strike selector.

        Returns:
            Active contract representation suitable for caching.

        Raises:
            None.
        """

        if isinstance(contract, ActiveContract):
            return contract
        expiry_dt = contract.expiry
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        return ActiveContract(
            underlying=underlying.strip().upper(),
            symbol=str(contract.symbol).strip().upper(),
            option_type=str(contract.option_type).strip().upper(),
            strike=float(contract.strike),
            expiry=expiry_dt,
        )


class PositionManager:
    """Track open positions and pending orders with persistence support."""

    FINAL_STATUSES: tuple[OrderStatus, ...] = (
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
    )

    def __init__(self, state_file: str = "positions.json") -> None:
        """Initialize the manager, optionally loading from ``state_file``."""

        self._logger = get_logger(__name__)
        self._state_path = Path(state_file)
        legacy_candidate = self._state_path.parent / "positions_state.json"
        if self._state_path.name == "positions_state.json":
            self._legacy_state_path: Path | None = None
        else:
            self._legacy_state_path = (
                legacy_candidate if legacy_candidate.exists() else None
            )
        self._positions: Dict[str, Position] = {}
        self._lock = threading.RLock()
        self._order_locks: dict[str, threading.RLock] = {}
        self._symbol_lifecycle_locks: dict[str, threading.RLock] = {}
        self._orders: Dict[str, Order] = {}
        self._terminal_orders: dict[str, TerminalOrderMetadata] = {}
        self._unresolved_terminal_orders: dict[str, TerminalOrderMetadata] = {}
        self._exit_lifecycles: dict[str, ExitLifecycleRecord] = {}
        self._max_terminal_orders = 5000  # Limit persisted idempotency history.
        self._daily_realized_pnl: float = 0.0
        self._local_realized_pnl: float = 0.0
        self._broker_realized_pnl: float | None = None
        self._local_provisional_realized_pnl: float = 0.0
        self._authoritative_realized_pnl: float = 0.0
        self._pnl_authority: str = "unresolved"
        self._pnl_reconciliation_status: str = "unresolved"
        self._pnl_snapshot_at: datetime | None = None
        self._session_opening_realized_baseline: float | None = None
        self._pnl_trading_date: str | None = None
        self._pnl_account_fingerprint: str | None = None
        self._pnl_product_scope: str = "MIS"
        self._baseline_established_at: datetime | None = None
        self._baseline_source: str | None = None
        self._require_pnl_baseline_for_entries: bool = False
        self._active_contracts: Dict[str, ActiveContract] = {}
        self._contract_index: Dict[str, str] = {}
        self._persistent_state: PersistentStateManager | None = None
        self._broker_client: Any | None = None
        self._reconcile_timer: threading.Timer | None = None
        self._reconcile_interval_s: float = 60.0
        self._reconcile_retry_interval_s: float = 10.0
        self._reconcile_listeners: list[Callable[[str, Mapping[str, object]], None]] = (
            []
        )
        self._last_reconciled_state: Dict[str, Position] = {}
        self._last_reconcile_attempt: datetime | None = None
        self._last_reconcile_success_at: datetime | None = None
        self._last_reconcile_error: str | None = None
        self._consecutive_reconcile_failures: int = 0
        self._persistence_flush_interval_s = 5.0
        self._persistence_max_age_s = 30.0
        self._persistence_pending_threshold = 10
        self._last_persistence_check = 0.0
        self.load_state()
        self._last_reconciled_state = copy.deepcopy(self._positions)

    def set_on_symbols_flat(self, hook: Any | None) -> None:
        """Attach a callback invoked with symbols pruned during broker sync.

        Lets the runner forward externally-closed symbols (manual square-off /
        auto-square-off) to the bracket manager so lingering brackets are dropped
        instead of being re-adopted forever.
        """
        self._on_symbols_flat_hook = hook

    @property
    def _processed_order_ids(self) -> set[str]:
        """Backward-compatible read view for older tests and diagnostics."""

        return {
            order_id
            for order_id, metadata in self._terminal_orders.items()
            if metadata.lifecycle_applied
        }

    @_processed_order_ids.setter
    def _processed_order_ids(self, value: Iterable[str]) -> None:
        now = _now()
        self._terminal_orders = {
            str(order_id): TerminalOrderMetadata(
                terminal_at=now,
                normalized_status="FILLED",
                cumulative_filled_quantity=0,
                average_fill_price=None,
                lifecycle_applied=True,
                accounting_finalized=True,
            )
            for order_id in value
        }

    def _order_lock_for(self, order_id: str) -> threading.RLock:
        with self._lock:
            return self._order_locks.setdefault(str(order_id), threading.RLock())

    def _symbol_lifecycle_lock_for(self, symbol: str) -> threading.RLock:
        with self._lock:
            return self._symbol_lifecycle_locks.setdefault(
                symbol.upper(), threading.RLock()
            )

    def set_broker_client(self, broker_client: Any | None) -> None:
        """Attach the broker client used for reconciliation.

        Args:
            broker_client: Broker client implementation or ``None`` to detach.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered set_broker_client",
            extra={"event": "position_manager_set_broker"},
        )
        self._broker_client = broker_client

    def _resolve_broker_position_fetcher(self) -> Callable[[], Any] | None:
        """Return the broker callable used to fetch positions.

        Args:
            None.

        Returns:
            Callable returning broker positions or ``None`` when unavailable.

        Raises:
            None.
        """

        broker = self._broker_client
        if broker is None:
            return None
        candidates = (
            "get_positions",
            "list_positions",
            "positions",
            "fetch_positions",
        )
        for name in candidates:
            fetcher = getattr(broker, name, None)
            if callable(fetcher):
                return cast(Callable[[], Any], fetcher)
        return None

    def _cancel_reconcile_timer(self) -> None:
        """Cancel any outstanding reconciliation timer.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        timer = self._reconcile_timer
        if timer is not None:
            timer.cancel()
            self._reconcile_timer = None

    def _schedule_reconcile(self, delay_s: float) -> None:
        """Schedule the next reconciliation attempt.

        Args:
            delay_s: Delay in seconds before the next execution.

        Returns:
            None.

        Raises:
            None.
        """

        delay = max(float(delay_s), 0.0)
        if delay <= 0.0:
            delay = _MIN_RECONCILE_DELAY_S
        self._cancel_reconcile_timer()
        timer = threading.Timer(delay, self.reconcile_periodic)
        timer.daemon = True
        self._reconcile_timer = timer
        timer.start()
        self._logger.debug(
            "Scheduled position reconciliation",
            extra={
                "event": "position_reconcile_schedule",
                "delay_sec": round(delay, 3),
            },
        )

    def _compute_retry_delay(self) -> float:
        """Return delay in seconds before scheduling the next reconcile retry.

        Args:
            None.

        Returns:
            Delay in seconds before the subsequent reconcile attempt.

        Raises:
            None.
        """

        base_delay = max(self._reconcile_retry_interval_s, _MIN_RECONCILE_DELAY_S)
        failures = max(self._consecutive_reconcile_failures, 1)
        multiplier = min(2 ** (failures - 1), 16.0)
        max_delay = max(self._reconcile_interval_s, base_delay) * 4.0
        return float(min(base_delay * multiplier, max_delay))

    def _schedule_retry_after_failure(self, delay_s: float | None = None) -> None:
        """Schedule a soft retry using exponential backoff delays.

        Args:
            delay_s: Optional pre-computed retry delay override.

        Returns:
            None.

        Raises:
            None.
        """

        retry_delay = (
            float(delay_s) if delay_s is not None else self._compute_retry_delay()
        )
        self._schedule_reconcile(retry_delay)

    def add_reconcile_listener(
        self, callback: Callable[[str, Mapping[str, object]], None]
    ) -> None:
        """Register *callback* to receive reconciliation lifecycle events.

        Args:
            callback: Callable invoked with the event name and payload.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered add_reconcile_listener",
            extra={"event": "position_reconcile_listener_add"},
        )
        if not callable(callback):
            self._logger.warning(
                "Ignoring non-callable reconcile listener",
                extra={"event": "position_reconcile_listener_invalid"},
            )
            return
        self._reconcile_listeners.append(callback)

    def _notify_reconcile_event(
        self, event: str, payload: Mapping[str, object]
    ) -> None:
        """Dispatch reconcile *event* with *payload* to registered listeners.

        Args:
            event: Event name emitted by the reconciliation pipeline.
            payload: Supplementary event payload cloned per listener.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _notify_reconcile_event",
            extra={"event": "position_reconcile_notify", "event_name": event},
        )
        listeners = list(self._reconcile_listeners)
        for listener in listeners:
            try:
                listener(event, dict(payload))
            except Exception as exc:  # noqa: BLE001 - defensive listener isolation
                self._logger.error(
                    "Failure in _notify_reconcile_event: %s",
                    exc,
                    extra={
                        "event": "position_reconcile_listener_error",
                        "listener": getattr(listener, "__name__", repr(listener)),
                    },
                )

    def _handle_reconcile_failure(
        self,
        *,
        reason: str,
        error: Exception | None,
        payload_count: int,
        previous_positions: Mapping[str, Position] | None,
    ) -> None:
        """Record failure without replacing newer local fill/position state."""
        self._last_reconcile_attempt = _now()
        self._consecutive_reconcile_failures += 1
        self._last_reconcile_error = str(error) if error is not None else reason
        reason_token = canonical(reason)
        event_key = f"failure:{self._last_reconcile_attempt.isoformat()}:{reason_token}"
        try:
            METRICS.record_broker_sync(
                success=False,
                reason=reason_token,
                latency_seconds=None,
                event_id=event_key,
            )
            METRICS.increment_retry_event(
                label="position_reconcile",
                stage="apply",
                outcome=reason_token,
            )
        except Exception as metrics_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in reconcile failure metrics: %s",
                metrics_exc,
                extra={"event": "position_reconcile_metric_failure"},
            )
        with self._lock:
            preserved_count = len(self._positions)
        retry_delay = self._compute_retry_delay()
        payload: dict[str, object] = {
            "reason": reason_token,
            "failures": self._consecutive_reconcile_failures,
            "retry_sec": retry_delay,
            "count": payload_count,
            "timestamp": self._last_reconcile_attempt.isoformat(),
            "restored": False,
            "source": "current_state_preserved",
            "preserved_count": preserved_count,
        }
        if error is not None:
            payload["error"] = str(error)
        try:
            _POSITION_RECONCILE_EVENTS.labels("failed").inc()
        except Exception:  # noqa: BLE001
            pass
        self._notify_reconcile_event("position_reconcile_failed", payload)
        self._schedule_retry_after_failure(retry_delay)

    def _handle_reconcile_success(self, payload_count: int) -> None:
        """Persist reconciliation success metadata and notify observers.

        Args:
            payload_count: Number of broker payloads applied.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _handle_reconcile_success",
            extra={"event": "position_reconcile_success"},
        )
        self._last_reconcile_attempt = _now()
        self._last_reconcile_success_at = self._last_reconcile_attempt
        previous_failures = self._consecutive_reconcile_failures
        self._consecutive_reconcile_failures = 0
        self._last_reconcile_error = None
        event_key = f"success:{self._last_reconcile_success_at.isoformat()}"
        try:
            METRICS.record_broker_sync(
                success=True,
                reason="ok",
                latency_seconds=None,
                event_id=event_key,
            )
            if previous_failures:
                METRICS.increment_retry_event(
                    label="position_reconcile",
                    stage="apply",
                    outcome="success",
                )
        except Exception as metrics_exc:  # noqa: BLE001 - defensive metric guard
            self._logger.error(
                "Failure in reconcile success metrics: %s",
                metrics_exc,
                extra={"event": "position_reconcile_metric_failure"},
            )
        try:
            self._last_reconciled_state = copy.deepcopy(self._positions)
        except Exception as exc:  # noqa: BLE001 - defensive snapshot guard
            self._logger.error(
                "Failure in _handle_reconcile_success snapshot: %s",
                exc,
                extra={"event": "position_reconcile_snapshot_failed"},
            )
        event_payload: dict[str, object] = {
            "count": payload_count,
            "timestamp": self._last_reconcile_success_at.isoformat(),
        }
        if previous_failures:
            event_payload["previous_failures"] = previous_failures
        try:
            _POSITION_RECONCILE_EVENTS.labels("ok").inc()
        except Exception:  # noqa: BLE001 - optional metrics backend
            pass
        self._notify_reconcile_event("position_reconcile_ok", event_payload)

    def reconcile_now(self) -> bool:
        """Fetch and atomically apply one authoritative broker snapshot."""
        payload_count = 0
        fetcher = self._resolve_broker_position_fetcher()
        if fetcher is None:
            self._handle_reconcile_failure(
                reason=canonical("fetcher_missing"),
                error=None,
                payload_count=0,
                previous_positions=None,
            )
            return False
        try:
            response = fetcher()
            snapshot = decode_position_snapshot(response)
        except Exception as exc:  # noqa: BLE001
            reason = canonical(
                "payload_invalid" if isinstance(exc, PositionSnapshotError) else "fetch_error"
            )
            self._logger.warning(
                "Position reconciliation snapshot failed: %s",
                exc,
                extra={"event": "position_reconcile_failed", "reason": reason},
                exc_info=exc,
            )
            self._handle_reconcile_failure(
                reason=reason,
                error=exc,
                payload_count=0,
                previous_positions=None,
            )
            return False

        payloads = snapshot.raw_rows()
        payload_count = len(payloads)
        try:
            self.synchronize_with_broker(payloads)
        except Exception as exc:  # noqa: BLE001
            reason = canonical("apply_error")
            self._logger.warning(
                "Position reconciliation apply failed: %s",
                exc,
                extra={"event": "position_reconcile_failed", "reason": reason},
                exc_info=exc,
            )
            self._handle_reconcile_failure(
                reason=reason,
                error=exc,
                payload_count=payload_count,
                previous_positions=None,
            )
            return False

        self._logger.info(
            "POSITION_RECONCILE_OK count=%s source=%s",
            payload_count,
            snapshot.source,
            extra={
                "event": "position_reconcile_ok",
                "count": payload_count,
                "source": snapshot.source,
            },
        )
        self._handle_reconcile_success(payload_count)
        return True

    def reconcile_periodic(
        self,
        *,
        interval_sec: float | None = None,
        retry_sec: float | None = None,
    ) -> None:
        """Reconcile now and schedule the next attempt.

        Args:
            interval_sec: Interval between successful reconciliations.
            retry_sec: Delay applied after a failed reconciliation.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered reconcile_periodic",
            extra={"event": "position_reconcile_periodic"},
        )
        try:
            if interval_sec is not None:
                self._reconcile_interval_s = max(
                    float(interval_sec), _MIN_RECONCILE_DELAY_S
                )
            if retry_sec is not None:
                self._reconcile_retry_interval_s = max(
                    float(retry_sec), _MIN_RECONCILE_DELAY_S
                )
            success = self.reconcile_now()
        except Exception as exc:  # noqa: BLE001 - defensive periodic guard
            reason_token = canonical("periodic_error")
            self._logger.warning(
                "Unexpected error during periodic reconciliation: %s",
                exc,
                extra={
                    "event": "position_reconcile_failed",
                    "reason": reason_token,
                },
                exc_info=exc,
            )
            self._handle_reconcile_failure(
                reason=reason_token,
                error=exc,
                payload_count=0,
                previous_positions=None,
            )
            return
        self._maybe_flush_persistent_state()
        if success:
            self._schedule_reconcile(self._reconcile_interval_s)

    def get_active_contract(self, underlying: str) -> ActiveContract | None:
        """Return cached contract for an *underlying*.

        Args:
            underlying: Underlying instrument identifier.

        Returns:
            Cached contract when available, else ``None``.

        Raises:
            None.
        """

        key = underlying.strip().upper()
        self._logger.debug(
            "Entered get_active_contract",
            extra={"event": "get_active_contract", "underlying": key},
        )
        if not key:
            return None
        try:
            return self._active_contracts.get(key)
        except Exception as exc:  # noqa: BLE001 - defensive guard
            self._logger.error("Failure in get_active_contract: %s", exc)
            return None

    def set_active_contract(
        self, underlying: str, contract: SelectedContract | ActiveContract | None
    ) -> None:
        """Persist latest selected contract for an *underlying*.

        Args:
            underlying: Underlying instrument identifier.
            contract: Selection metadata or ``None`` to clear.

        Returns:
            None.

        Raises:
            None.
        """

        normalized = underlying.strip().upper()
        self._logger.debug(
            "Entered set_active_contract",
            extra={"event": "set_active_contract", "underlying": normalized},
        )
        if not normalized:
            return
        try:
            if contract is None:
                removed = self._active_contracts.pop(normalized, None)
                if removed is not None:
                    self._contract_index.pop(removed.symbol, None)
                    self._logger.info(
                        "Condition met: active_contract_cleared",
                        extra={
                            "event": "active_contract_cleared",
                            "underlying": normalized,
                            "symbol": removed.symbol,
                        },
                    )
                self.save_state()
                return
            active = ActiveContract.from_selection(normalized, contract)
            self._active_contracts[normalized] = active
            self._contract_index[active.symbol] = normalized
            self._logger.info(
                "Condition met: active_contract_registered",
                extra={
                    "event": "active_contract_registered",
                    "underlying": normalized,
                    "symbol": active.symbol,
                },
            )
        except Exception as exc:  # noqa: BLE001 - defensive guard
            self._logger.error("Failure in set_active_contract: %s", exc)
            return
        self.save_state()

    def clear_active_contract(self, underlying: str) -> None:
        """Remove cached contract for an *underlying*.

        Args:
            underlying: Underlying instrument identifier.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered clear_active_contract",
            extra={"event": "clear_active_contract", "underlying": underlying},
        )
        self.set_active_contract(underlying, None)

    def clear_active_contract_by_symbol(self, symbol: str) -> None:
        """Remove cached contract resolved by *symbol*.

        Args:
            symbol: Option contract identifier.

        Returns:
            None.

        Raises:
            None.
        """

        normalized = symbol.strip().upper()
        self._logger.debug(
            "Entered clear_active_contract_by_symbol",
            extra={"event": "clear_active_contract_by_symbol", "symbol": normalized},
        )
        if not normalized:
            return
        try:
            underlying = self._contract_index.pop(normalized, None)
            if underlying:
                if self._active_contracts.pop(underlying, None) is not None:
                    self._logger.info(
                        "Condition met: active_contract_symbol_cleared",
                        extra={
                            "event": "active_contract_symbol_cleared",
                            "underlying": underlying,
                            "symbol": normalized,
                        },
                    )
                    self.save_state()
        except Exception as exc:  # noqa: BLE001 - defensive guard
            self._logger.error(
                "Failure in clear_active_contract_by_symbol: %s",
                exc,
            )

    def is_flat(self, symbol: str) -> bool:
        """Return ``True`` when *symbol* has no open position.

        Args:
            symbol: Option contract identifier.

        Returns:
            ``True`` if quantity is zero or position missing.

        Raises:
            None.
        """

        lookup = symbol.strip().upper()
        self._logger.debug(
            "Entered is_flat", extra={"event": "is_flat", "symbol": lookup}
        )
        try:
            position = self._positions.get(lookup)
        except Exception as exc:  # noqa: BLE001 - defensive guard
            self._logger.error("Failure in is_flat: %s", exc)
            return True
        return position is None or position.quantity <= 0

    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        entry_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_distance: float | None = None,
        order_id: str | None = None,
    ) -> Position:
        """Open a position under the same lock used by broker reconciliation."""
        symbol_key = symbol.upper()
        position = Position(
            symbol=symbol_key,
            side=_normalize_side(str(side)),
            quantity=int(quantity),
            entry_price=float(entry_price),
            entry_time=_now(),
            current_price=float(entry_price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_distance=trailing_stop_distance,
            order_id=order_id,
        )
        with self._lock:
            if symbol_key in self._positions:
                raise ValueError(f"Position already exists for {symbol_key}")
            self._positions[symbol_key] = position
        self._logger.info("Opened %s position for %s", position.side, symbol_key)
        self.save_state()
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        close_time: datetime | None = None,
    ) -> Position:
        """Close a position atomically and retain conservative realised P&L."""
        symbol_key = symbol.upper()
        with self._lock:
            position = self._positions.get(symbol_key)
            if position is None:
                raise ValueError(f"No open position for {symbol_key}")
            qty = position.quantity
            realized = self._calculate_realized_pnl(
                position.side, position.entry_price, float(exit_price), qty
            )
            position.realized_pnl += realized
            self._local_realized_pnl += realized
            self._refresh_realized_pnl_locked()
            position.current_price = float(exit_price)
            position.quantity = 0
            del self._positions[symbol_key]
        closed_at = close_time or _now()
        self._logger.info(
            "Closed %s position for %s at %.2f (%s) due to %s [PnL=%.2f]",
            position.side,
            symbol_key,
            exit_price,
            closed_at.isoformat(),
            reason,
            realized,
        )
        self.clear_active_contract_by_symbol(symbol_key)
        self.save_state()
        return position

    def update_from_order(self, order: Order) -> None:
        """Apply a confirmed local :class:`Order` through the normal fill lifecycle."""
        if not isinstance(order, Order):
            raise TypeError("update_from_order requires position_manager.Order")
        if order.status != "FILLED":
            return
        if order.filled_quantity <= 0:
            raise ValueError("filled order has no filled quantity")
        fill_price = order.fill_price
        if fill_price is None or float(fill_price) <= 0:
            raise ValueError("filled order has no valid fill_price")
        with self._lock:
            self._orders.setdefault(order.order_id, order)
        self.update_order_status(order.order_id, "FILLED", fill_price=float(fill_price))

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update the mark price of an open position under the state lock."""
        symbol_key = symbol.upper()
        with self._lock:
            position = self._positions.get(symbol_key)
            if position is None:
                return
            position.current_price = float(current_price)
        self.save_state()

    def get_position(self, symbol: str) -> Position | None:
        """Return the :class:`Position` for ``symbol`` if it exists."""

        return self._positions.get(symbol.upper())

    def get_all_positions(self) -> list[Position]:
        """Return all currently open positions."""

        return list(self._positions.values())

    def get_open_positions(self) -> list[Position]:
        """Alias for :meth:`get_all_positions` for compatibility with protocols."""

        return list(self._positions.values())

    def has_position(self, symbol: str) -> bool:
        """Return ``True`` if a position exists for ``symbol``."""

        return symbol.upper() in self._positions

    def has_open_position(self, symbol: str) -> bool:
        """Return whether an open position exists. Args: symbol. Returns: bool. Raises: None."""
        return self.has_position(symbol)

    def get_total_exposure(self) -> float:
        """Return the total notional exposure across open positions."""

        return float(
            sum(
                abs(position.quantity * position.current_price)
                for position in self._positions.values()
            )
        )

    def get_net_pnl(self) -> float:
        """Return total realized plus unrealized profit and loss."""

        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_unrealized_pnl(self) -> float:
        """Return the aggregate unrealized profit and loss."""

        return float(
            sum(position.unrealized_pnl for position in self._positions.values())
        )

    def _refresh_realized_pnl_locked(self) -> None:
        """Refresh authoritative confirmed P&L without silently taking min()."""

        local_confirmed = float(self._local_realized_pnl)
        broker_confirmed = None
        if (
            self._broker_realized_pnl is not None
            and self._session_opening_realized_baseline is not None
        ):
            broker_confirmed = float(self._broker_realized_pnl) - float(
                self._session_opening_realized_baseline
            )
        if local_confirmed != 0.0:
            authoritative = local_confirmed
            authority = "local_confirmed_ledger"
            if (
                broker_confirmed is not None
                and abs(local_confirmed - broker_confirmed) > 1.0
            ):
                status = "mismatch"
            else:
                status = "matched" if broker_confirmed is not None else "local_only"
        elif broker_confirmed is not None:
            authoritative = broker_confirmed
            authority = "validated_broker_positions"
            status = "broker_only"
        else:
            authoritative = 0.0
            authority = "unresolved"
            status = "unresolved"
        self._authoritative_realized_pnl = authoritative
        self._daily_realized_pnl = authoritative
        self._pnl_authority = authority
        self._pnl_reconciliation_status = status
        self._pnl_snapshot_at = _now()

    @staticmethod
    def _trading_date_ist(now: datetime | None = None) -> str:
        """Return the exchange trading date in IST for P&L baselines."""

        current = now or _now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()

    def establish_pnl_session_baseline(
        self,
        broker_realized: float,
        *,
        account_fingerprint: str | None = None,
        product_scope: str = "MIS",
        snapshot_at: datetime | None = None,
        source: str = "validated_broker_positions",
        trading_date: str | None = None,
    ) -> bool:
        """Persist the opening broker-realized baseline for today's bot P&L.

        Same-day restarts retain the existing baseline so an old cumulative
        broker realized value is not misread as today's bot loss.
        """

        value = float(broker_realized)
        if not math.isfinite(value):
            raise ValueError("broker_realized must be finite")
        as_of = snapshot_at or _now()
        session_date = trading_date or self._trading_date_ist(as_of)
        with self._lock:
            if (
                self._session_opening_realized_baseline is not None
                and self._pnl_trading_date == session_date
            ):
                self._broker_realized_pnl = value
                self._refresh_realized_pnl_locked()
                return False
            self._session_opening_realized_baseline = value
            self._pnl_trading_date = session_date
            self._pnl_account_fingerprint = account_fingerprint
            self._pnl_product_scope = product_scope
            self._baseline_established_at = as_of
            self._baseline_source = source
            self._broker_realized_pnl = value
            self._refresh_realized_pnl_locked()
        self.save_state()
        return True

    def broker_session_realized_pnl(self) -> float | None:
        """Return broker cumulative realized minus the persisted session baseline."""

        with self._lock:
            if (
                self._broker_realized_pnl is None
                or self._session_opening_realized_baseline is None
            ):
                return None
            return float(self._broker_realized_pnl) - float(
                self._session_opening_realized_baseline
            )
    def get_realized_pnl(self) -> float:
        """Return conservative realised P&L used by capital-protection gates."""
        with self._lock:
            return float(self._daily_realized_pnl)

    def pnl_reconciliation_snapshot(self) -> dict[str, object]:
        """Return current confirmed P&L authority and mismatch details."""

        with self._lock:
            return {
                "local_confirmed_realized": float(self._local_realized_pnl),
                "local_provisional_realized": float(
                    self._local_provisional_realized_pnl
                ),
                "broker_realized_snapshot": self._broker_realized_pnl,
                "broker_session_realized": (
                    None
                    if self._broker_realized_pnl is None
                    or self._session_opening_realized_baseline is None
                    else float(self._broker_realized_pnl)
                    - float(self._session_opening_realized_baseline)
                ),
                "authoritative_realized": float(self._authoritative_realized_pnl),
                "pnl_authority": self._pnl_authority,
                "pnl_reconciliation_status": self._pnl_reconciliation_status,
                "session_opening_realized_baseline": (
                    self._session_opening_realized_baseline
                ),
                "pnl_trading_date": self._pnl_trading_date,
                "pnl_account_fingerprint": self._pnl_account_fingerprint,
                "pnl_product_scope": self._pnl_product_scope,
                "baseline_established_at": (
                    self._baseline_established_at.isoformat()
                    if self._baseline_established_at
                    else None
                ),
                "baseline_source": self._baseline_source,
                "pnl_snapshot_at": (
                    self._pnl_snapshot_at.isoformat() if self._pnl_snapshot_at else None
                ),
            }

    def current_pnl_reconciliation_blocker(self) -> str | None:
        """Block new entries when confirmed local and broker P&L disagree."""

        with self._lock:
            if (
                self._require_pnl_baseline_for_entries
                and self._session_opening_realized_baseline is None
            ):
                return "pnl_baseline_uninitialized"
            if self._pnl_reconciliation_status == "mismatch":
                return "pnl_reconciliation_mismatch"
            return None

    def require_pnl_session_baseline(self, required: bool = True) -> None:
        """Require a validated session baseline before new entries are accepted."""

        with self._lock:
            self._require_pnl_baseline_for_entries = bool(required)

    def current_entry_protection_blocker(self, symbol: str | None = None) -> str | None:
        """Return current entry blocker when a filled entry lacks SL protection."""

        symbol_key = symbol.upper() if symbol else None
        with self._lock:
            for order in self._orders.values():
                if symbol_key is not None and order.symbol != symbol_key:
                    continue
                if order.intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                    continue
                if order.applied_filled_quantity <= 0:
                    continue
                if (
                    not order.protection_confirmed
                    or order.protected_quantity < order.applied_filled_quantity
                ):
                    return "entry_protection_incomplete"
            for metadata in self._unresolved_terminal_orders.values():
                if symbol_key is not None and metadata.symbol != symbol_key:
                    continue
                if metadata.intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                    continue
                if (
                    not metadata.protection_confirmed
                    or metadata.protected_quantity
                    < metadata.cumulative_filled_quantity
                ):
                    return "entry_protection_incomplete"
        return None

    def add_pending_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_type: str,
        intent: OrderIntent | str | None = None,
        bracket_id: str | None = None,
        signal_id: str | None = None,
        signal_fingerprint: str | None = None,
    ) -> None:
        """Track a newly submitted order.
        
        ✅ PRODUCTION FIX: Skip orders that have already been fully processed.
        This prevents the infinite loop where historical filled orders are
        re-added and re-processed every reconciliation cycle.
        """
        order_id = str(order_id).strip()
        
        # ✅ FIX 1: Don't re-add orders that were already processed
        if (
            hasattr(self, "_terminal_orders")
            and order_id in self._terminal_orders
            and self._terminal_orders[order_id].lifecycle_applied
        ):
            self._logger.debug(
                f"Skipping add_pending_order for already-processed: {order_id}",
                extra={"event": "order_add_skip_processed", "order_id": order_id}
            )
            return
        
        # ✅ FIX 2: Don't re-add orders that are already being tracked
        if order_id in self._orders:
            self._logger.debug(
                f"Order already tracked, skipping: {order_id}",
                extra={"event": "order_add_skip_existing", "order_id": order_id}
            )
            return

        symbol_key = symbol.upper()
        existing_position = self._positions.get(symbol_key)
        normalized_side = _normalize_order_side(side)
        normalized_intent = _normalize_intent(intent)
        if normalized_intent == "UNKNOWN":
            if existing_position is not None:
                exit_side = "SELL" if existing_position.side == "LONG" else "BUY"
                normalized_intent = (
                    "EXIT" if normalized_side == exit_side else "SCALE_IN"
                )
        order = Order(
            order_id=order_id,
            symbol=symbol_key,
            side=normalized_side,
            order_type=order_type,
            quantity=int(qty),
            price=float(price),
            status="PENDING",
            linked_position_symbol=(
                symbol_key if existing_position is not None else None
            ),
            intent=normalized_intent,
            bracket_id=bracket_id,
            signal_id=signal_id,
            signal_fingerprint=signal_fingerprint,
            pre_order_position_side=(
                existing_position.side if existing_position else None
            ),
            pre_order_quantity=existing_position.quantity if existing_position else 0,
            trade_lifecycle_id=bracket_id or signal_id or order_id,
            linked_entry_order_id=(
                existing_position.order_id
                if existing_position is not None and normalized_intent in ("EXIT", "REDUCE")
                else None
            ),
            pre_order_entry_price=(
                existing_position.entry_price if existing_position is not None else None
            ),
        )
        self._orders[order.order_id] = order
        if normalized_intent in ("EXIT", "REDUCE"):
            self._exit_lifecycles[order.order_id] = ExitLifecycleRecord(
                symbol=symbol_key,
                exit_order_id=order.order_id,
                linked_entry_order_id=order.linked_entry_order_id,
                trade_lifecycle_id=order.trade_lifecycle_id,
                bracket_id=order.bracket_id,
                expected_exit_side=order.side,
                expected_exit_quantity=order.quantity,
            )
        self._persist_order_state(order)
        self.save_state()

    def update_order_status(
        self,
        order_id: str,
        status: str,
        fill_price: float | None = None,
    ) -> None:
        """Update the status of a tracked order and react to fills.
        
        ✅ PRODUCTION FIX: Added guard against re-processing completed orders.
        This prevents the position thrashing loop where the same filled orders
        are processed over and over again, causing:
        - "Opened LONG position" 
        - "Position fully closed via order"
        to repeat infinitely.
        """
        order_id = str(order_id).strip()
        
        # ✅ FIX: Initialize _processed_order_ids if not exists (backward compat)
        if not hasattr(self, "_terminal_orders"):
            self._terminal_orders = {}
            self._max_terminal_orders = 5000
        
        # ✅ FIX: Skip if this order was already fully processed
        terminal_record = self._terminal_orders.get(order_id)
        if terminal_record is not None and terminal_record.lifecycle_resolved:
            self._logger.debug(
                f"Skipping already-processed order: {order_id}",
                extra={"event": "order_already_processed", "order_id": order_id}
            )
            return
        incoming_status = normalize_broker_order_status(status)
        if (
            terminal_record is not None
            and terminal_record.normalized_status in self.FINAL_STATUSES
            and incoming_status not in self.FINAL_STATUSES
        ):
            self._logger.warning(
                "Ignoring terminal order status regression for %s: %s -> %s",
                order_id,
                terminal_record.normalized_status,
                incoming_status,
                extra={
                    "event": "order_status_regression_ignored",
                    "order_id": order_id,
                    "from_status": terminal_record.normalized_status,
                    "to_status": incoming_status,
                },
            )
            return
        
        order = self._orders.get(order_id)
        if order is None:
            # ✅ FIX: Also skip unknown orders that might be historical
            self._logger.debug(
                f"Attempted to update unknown order {order_id} - may be historical",
                extra={"event": "order_update_skip_unknown", "order_id": order_id}
            )
            return

        try:
            order.status = incoming_status or _normalize_status(str(status))
        except ValueError:
            self._logger.warning(
                "Ignoring unsupported status '%s' for order %s", status, order_id
            )
            return

        if fill_price is not None:
            order.fill_price = float(fill_price)

        fill_result = FillApplicationResult()
        if order.status in ("PARTIALLY_FILLED", "FILLED") and order.fill_price is not None:
            if order.filled_quantity <= 0:
                order.filled_quantity = order.quantity
            fill_result = self._handle_filled_order(order)
        if order.status == "FILLED" and order.fill_price is not None:
            existing_terminal = self._terminal_orders.get(order_id)
            if existing_terminal is not None and not fill_result.fill_recorded:
                return
            order.terminal_at = _now()
            self._terminal_orders[order_id] = TerminalOrderMetadata(
                terminal_at=order.terminal_at,
                normalized_status=order.status,
                cumulative_filled_quantity=order.filled_quantity,
                average_fill_price=order.fill_price,
                lifecycle_applied=fill_result.fill_recorded,
                accounting_finalized=fill_result.accounting_finalized,
                terminal_update_seen=True,
                fill_recorded=fill_result.fill_recorded,
                position_applied=fill_result.position_applied,
                bracket_applied=fill_result.bracket_applied,
                pnl_applied=fill_result.pnl_applied,
                lifecycle_resolved=fill_result.lifecycle_resolved,
                symbol=order.symbol,
                intent=order.intent,
                side=order.side,
                trade_lifecycle_id=order.trade_lifecycle_id,
                linked_entry_order_id=order.linked_entry_order_id,
                exit_lifecycle_state=(
                    self._exit_lifecycles[order_id].state
                    if order_id in self._exit_lifecycles
                    else None
                ),
                protected_quantity=order.protected_quantity,
                protection_confirmed=order.protection_confirmed,
                protection_confirmed_at=order.protection_confirmed_at,
                protection_failure_reason=order.protection_failure_reason,
            )
            if not self._terminal_orders[order_id].lifecycle_resolved:
                self._unresolved_terminal_orders[order_id] = self._terminal_orders[order_id]
            else:
                self._unresolved_terminal_orders.pop(order_id, None)
            self._logger.debug(
                f"Marked terminal order: {order_id}",
                extra={
                    "event": "order_terminal_recorded",
                    "order_id": order_id,
                    "lifecycle_applied": fill_result.position_applied or fill_result.pnl_applied,
                }
            )
            self._evict_old_terminal_orders()

        self._persist_order_state(order)

        if (
            order.status in self.FINAL_STATUSES
            and self._terminal_orders.get(order.order_id) is not None
            and self._terminal_orders[order.order_id].lifecycle_resolved
        ):
            del self._orders[order.order_id]

        self.save_state()

    def apply_broker_order_update(
        self, order_id: str, broker_payload: Mapping[str, Any]
    ) -> None:
        """Canonical position-side broker update ingress."""

        order_key = str(order_id)
        order_lock = self._order_lock_for(order_key)
        with order_lock:
            order = self._orders.get(order_key)
            symbol_lock = (
                self._symbol_lifecycle_lock_for(order.symbol)
                if order is not None
                else self._lock
            )
            with symbol_lock:
                status = broker_payload.get("status")
                fill_price_raw = (
                    broker_payload.get("average_price")
                    or broker_payload.get("fill_price")
                    or broker_payload.get("price")
                )
                filled_qty = broker_payload.get("filled_quantity") or broker_payload.get(
                    "filled"
                )
                if order is not None and filled_qty is not None:
                    with suppress(Exception):
                        order.filled_quantity = int(float(filled_qty))
                fill_price: float | None = None
                if fill_price_raw is not None:
                    with suppress(Exception):
                        fill_price = float(fill_price_raw)
                self.update_order_status(order_key, str(status or ""), fill_price)

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
        """Return tracked orders, optionally filtered by ``symbol``."""

        symbol_key = symbol.upper() if symbol else None
        orders: Iterable[Order] = self._orders.values()
        if symbol_key is not None:
            orders = (order for order in orders if order.symbol == symbol_key)
        return [order for order in orders if order.status not in self.FINAL_STATUSES]

    def unresolved_terminal_summary(self) -> dict[str, object]:
        """Return count and oldest age for terminal fills awaiting reconciliation."""

        now = _now()
        unresolved = list(self._unresolved_terminal_orders.values())
        oldest_age_s = None
        if unresolved:
            oldest = min(item.terminal_at for item in unresolved)
            oldest_age_s = max((now - oldest).total_seconds(), 0.0)
        return {
            "count": len(unresolved),
            "oldest_age_s": oldest_age_s,
        }

    def confirm_entry_protection(
        self,
        order_id: str,
        bracket_id: str,
        protected_quantity: int,
    ) -> None:
        """Acknowledge verified SL/TP protection for a filled entry order.

        Bracket metadata alone is not proof of protection. The canonical
        bracket/order runtime must call this only after it has verified the
        active bracket, configured stop loss, symbol and lifecycle linkage.
        """

        order_key = str(order_id).strip()
        bracket_key = str(bracket_id).strip()
        protected_qty = int(protected_quantity)
        if protected_qty <= 0:
            raise ValueError("protected_quantity must be positive")
        with self._lock:
            order = self._orders.get(order_key)
            if order is None:
                raise KeyError(f"Unknown entry order '{order_key}'")
            if order.intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                raise ValueError("Only entry-intent orders can confirm protection")
            if order.bracket_id and order.bracket_id != bracket_key:
                raise ValueError("Bracket ID does not match entry order")
            if order.applied_filled_quantity <= 0:
                raise ValueError("Entry fill must be applied before protection")
            if protected_qty < order.applied_filled_quantity:
                order.protected_quantity = protected_qty
                order.protection_confirmed = False
                order.protection_failure_reason = "entry_protection_incomplete"
                self._persist_order_state(order)
                self.save_state()
                raise ValueError("protected quantity is below filled quantity")

            now = _now()
            order.bracket_id = bracket_key
            order.protected_quantity = protected_qty
            order.protection_confirmed = True
            order.protection_confirmed_at = now
            order.protection_failure_reason = None

            metadata = self._terminal_orders.get(order_key)
            if metadata is not None:
                metadata.bracket_applied = True
                metadata.protected_quantity = protected_qty
                metadata.protection_confirmed = True
                metadata.protection_confirmed_at = now
                metadata.protection_failure_reason = None
                if metadata.fill_recorded and metadata.position_applied:
                    metadata.lifecycle_resolved = True
                    metadata.accounting_finalized = False
                    self._unresolved_terminal_orders.pop(order_key, None)
            self._persist_order_state(order)
        self.save_state()

    def save_state(self) -> None:
        """Persist one coherent positions/orders snapshot to disk."""
        with self._lock:
            state = {
                "positions": [position.to_dict() for position in self._positions.values()],
                "orders": [order.to_dict() for order in self._orders.values()],
                "terminal_orders": {
                    order_id: metadata.to_dict()
                    for order_id, metadata in self._terminal_orders.items()
                },
                "unresolved_terminal_orders": {
                    order_id: metadata.to_dict()
                    for order_id, metadata in self._unresolved_terminal_orders.items()
                },
                "exit_lifecycles": {
                    order_id: lifecycle.to_dict()
                    for order_id, lifecycle in self._exit_lifecycles.items()
                },
                "daily_realized_pnl": self._daily_realized_pnl,
                "local_realized_pnl": self._local_realized_pnl,
                "broker_realized_pnl": self._broker_realized_pnl,
                "local_provisional_realized_pnl": self._local_provisional_realized_pnl,
                "authoritative_realized_pnl": self._authoritative_realized_pnl,
                "pnl_authority": self._pnl_authority,
                "pnl_reconciliation_status": self._pnl_reconciliation_status,
                "pnl_snapshot_at": (
                    self._pnl_snapshot_at.isoformat() if self._pnl_snapshot_at else None
                ),
                "session_opening_realized_baseline": (
                    self._session_opening_realized_baseline
                ),
                "pnl_trading_date": self._pnl_trading_date,
                "pnl_account_fingerprint": self._pnl_account_fingerprint,
                "pnl_product_scope": self._pnl_product_scope,
                "baseline_established_at": (
                    self._baseline_established_at.isoformat()
                    if self._baseline_established_at
                    else None
                ),
                "baseline_source": self._baseline_source,
                "require_pnl_baseline_for_entries": (
                    self._require_pnl_baseline_for_entries
                ),
                "active_contracts": [
                    contract.to_dict() for contract in self._active_contracts.values()
                ],
            }
            reconciled_snapshot = copy.deepcopy(self._positions)
        try:
            _atomic_write_json(self._state_path, state)
        except Exception as exc:  # noqa: BLE001 - handled by callers/diagnostics
            self._logger.error("Failed to save position state: %s", exc)
            return
        self._persist_positions_snapshot()
        with self._lock:
            self._last_reconciled_state = reconciled_snapshot
        self._maybe_flush_persistent_state()

    def load_state(self) -> None:
        """Load persisted state from disk if available."""

        path_to_read = self._state_path
        manager = self._persistent_state
        if not path_to_read.exists() and self._legacy_state_path is not None:
            path_to_read = self._legacy_state_path
            if not path_to_read.exists():
                if manager is not None:
                    self._restore_from_persistent_manager(manager)
                return
        elif not path_to_read.exists():
            if manager is not None:
                self._restore_from_persistent_manager(manager)
            return
        try:
            raw = path_to_read.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, ValueError) as exc:
            self._logger.error("Failed to load position state: %s", exc)
            if manager is not None:
                self._restore_from_persistent_manager(manager)
            return

        if not isinstance(payload, dict):
            self._logger.error("Invalid state payload (expected object)")
            if manager is not None:
                self._restore_from_persistent_manager(manager)
            return

        positions: Dict[str, Position] = {}
        for item in payload.get("positions", []):
            try:
                position = Position.from_dict(cast(Mapping[str, Any], item))
            except (ValueError, TypeError) as exc:
                self._logger.error("Skipping invalid position state: %s", exc)
                continue
            positions[position.symbol.upper()] = position

        orders: Dict[str, Order] = {}
        for item in payload.get("orders", []):
            try:
                order = Order.from_dict(cast(Mapping[str, Any], item))
            except (ValueError, TypeError) as exc:
                self._logger.error("Skipping invalid order state: %s", exc)
                continue
            orders[order.order_id] = order

        self._positions = positions
        self._orders = orders
        terminal_raw = payload.get("terminal_orders", {})
        restored_terminal: dict[str, TerminalOrderMetadata] = {}
        if isinstance(terminal_raw, Mapping):
            for order_id, metadata in terminal_raw.items():
                if not isinstance(metadata, Mapping):
                    continue
                try:
                    restored_terminal[str(order_id)] = TerminalOrderMetadata.from_dict(
                        cast(Mapping[str, Any], metadata)
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._logger.error("Skipping invalid terminal order: %s", exc)
        else:
            # Backward compatibility with the prior reviewed patch.
            processed_raw = payload.get("processed_order_ids", [])
            if isinstance(processed_raw, list):
                now = _now()
                restored_terminal = {
                    str(item).strip(): TerminalOrderMetadata(
                        terminal_at=now,
                        normalized_status="FILLED",
                        cumulative_filled_quantity=0,
                        average_fill_price=None,
                        lifecycle_applied=True,
                        accounting_finalized=True,
                    )
                    for item in processed_raw
                    if str(item).strip()
                }
        self._terminal_orders = restored_terminal
        unresolved_raw = payload.get("unresolved_terminal_orders", {})
        restored_unresolved: dict[str, TerminalOrderMetadata] = {}
        if isinstance(unresolved_raw, Mapping):
            for order_id, metadata in unresolved_raw.items():
                if not isinstance(metadata, Mapping):
                    continue
                try:
                    restored_unresolved[str(order_id)] = (
                        TerminalOrderMetadata.from_dict(
                            cast(Mapping[str, Any], metadata)
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._logger.error(
                        "Skipping invalid unresolved terminal order: %s", exc
                    )
        else:
            restored_unresolved = {
                order_id: metadata
                for order_id, metadata in restored_terminal.items()
                if not metadata.lifecycle_resolved
            }
        self._unresolved_terminal_orders = restored_unresolved
        exit_lifecycles_raw = payload.get("exit_lifecycles", {})
        restored_exit_lifecycles: dict[str, ExitLifecycleRecord] = {}
        if isinstance(exit_lifecycles_raw, Mapping):
            for order_id, lifecycle in exit_lifecycles_raw.items():
                if not isinstance(lifecycle, Mapping):
                    continue
                try:
                    restored_exit_lifecycles[str(order_id)] = (
                        ExitLifecycleRecord.from_dict(
                            cast(Mapping[str, Any], lifecycle)
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self._logger.error("Skipping invalid exit lifecycle: %s", exc)
        self._exit_lifecycles = restored_exit_lifecycles
        contracts: Dict[str, ActiveContract] = {}
        index: Dict[str, str] = {}
        for item in payload.get("active_contracts", []):
            try:
                contract = ActiveContract.from_dict(cast(Mapping[str, Any], item))
            except (ValueError, TypeError) as exc:
                self._logger.error("Skipping invalid contract state: %s", exc)
                continue
            contracts[contract.underlying] = contract
            index[contract.symbol] = contract.underlying

        self._active_contracts = contracts
        self._contract_index = index
        legacy_daily = float(payload.get("daily_realized_pnl", 0.0))
        self._local_realized_pnl = float(
            payload.get("local_realized_pnl", legacy_daily)
        )
        broker_realized = payload.get("broker_realized_pnl")
        self._broker_realized_pnl = (
            None if broker_realized is None else float(broker_realized)
        )
        self._local_provisional_realized_pnl = float(
            payload.get("local_provisional_realized_pnl", 0.0)
        )
        self._authoritative_realized_pnl = float(
            payload.get("authoritative_realized_pnl", self._local_realized_pnl)
        )
        self._pnl_authority = str(payload.get("pnl_authority", "unresolved"))
        self._pnl_reconciliation_status = str(
            payload.get("pnl_reconciliation_status", "unresolved")
        )
        pnl_snapshot_at = payload.get("pnl_snapshot_at")
        if isinstance(pnl_snapshot_at, str) and pnl_snapshot_at:
            with suppress(ValueError):
                self._pnl_snapshot_at = datetime.fromisoformat(pnl_snapshot_at)
        baseline = payload.get("session_opening_realized_baseline")
        self._session_opening_realized_baseline = (
            None if baseline is None else float(baseline)
        )
        self._pnl_trading_date = (
            str(payload["pnl_trading_date"])
            if payload.get("pnl_trading_date") is not None
            else None
        )
        self._pnl_account_fingerprint = (
            str(payload["pnl_account_fingerprint"])
            if payload.get("pnl_account_fingerprint") is not None
            else None
        )
        self._pnl_product_scope = str(payload.get("pnl_product_scope", "MIS"))
        baseline_established_at = payload.get("baseline_established_at")
        if isinstance(baseline_established_at, str) and baseline_established_at:
            with suppress(ValueError):
                self._baseline_established_at = datetime.fromisoformat(
                    baseline_established_at
                )
        self._baseline_source = (
            str(payload["baseline_source"])
            if payload.get("baseline_source") is not None
            else None
        )
        self._require_pnl_baseline_for_entries = bool(
            payload.get("require_pnl_baseline_for_entries", False)
        )
        with self._lock:
            self._refresh_realized_pnl_locked()
        try:
            self._last_reconciled_state = copy.deepcopy(self._positions)
        except Exception as exc:  # noqa: BLE001 - defensive snapshot guard
            self._logger.error(
                "Failure in load_state snapshot: %s",
                exc,
                extra={"event": "position_reconcile_snapshot_failed"},
            )

    def _restore_from_persistent_manager(
        self, manager: "PersistentStateManager"
    ) -> None:
        """Recover positions and orders from attached persistent manager.

        Args:
            manager: Persistent state manager providing durable snapshots.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _restore_from_persistent_manager",
            extra={"event": "position_manager_restore_persistent"},
        )
        try:
            payloads = manager.load_positions()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _restore_from_persistent_manager positions: %s",
                exc,
            )
            return
        self.restore_positions(payloads)
        try:
            orders_payloads = manager.load_open_orders()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _restore_from_persistent_manager orders: %s",
                exc,
            )
            orders_payloads = []
        rebuilt_orders: Dict[str, Order] = {}
        for item in orders_payloads:
            if not isinstance(item, Mapping):
                continue
            try:
                order = Order.from_dict(cast(Mapping[str, Any], item))
            except (KeyError, TypeError, ValueError) as exc:
                self._logger.error(
                    "Failure in _restore_from_persistent_manager decode: %s",
                    exc,
                )
                continue
            rebuilt_orders[order.order_id] = order
        self._orders = rebuilt_orders
        if rebuilt_orders:
            self._logger.info(
                "Condition met: restore_from_persistent_orders",
                extra={
                    "event": "position_manager_restored_orders",
                    "count": len(rebuilt_orders),
                },
            )

    # ------------------------------------------------------------------
    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        """Attach a persistent state manager for durable snapshots.

        Args:
            manager: Persistent state manager coordinating disk writes.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered attach_persistent_state",
            extra={"event": "position_manager_attach_persistent"},
        )
        self._persistent_state = manager

    def restore_positions(self, payloads: Iterable[Mapping[str, Any]]) -> None:
        """Restore a validated persisted snapshot, including an explicit empty state."""
        self._logger.debug(
            "Entered restore_positions",
            extra={"event": "position_manager_restore"},
        )
        try:
            items = list(payloads)
        except TypeError as exc:
            raise ValueError("persisted position snapshot is not iterable") from exc

        rebuilt: Dict[str, Position] = {}
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"persisted position row {index} is not a mapping")
            try:
                position = Position.from_dict(cast(Mapping[str, Any], item))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid persisted position row {index}") from exc
            rebuilt[position.symbol.upper()] = position

        with self._lock:
            self._positions = rebuilt
        self._logger.info(
            "Condition met: restore_positions_applied",
            extra={"event": "position_manager_restore_applied", "count": len(rebuilt)},
        )
        self.save_state()

    @staticmethod
    def _safe_get_net_qty(record: Mapping[str, object]) -> int:
        """Return broker net quantity without converting missing/invalid data to flat."""
        quantity_keys = ("net_qty", "net_quantity", "netQuantity", "net", "quantity")
        found = False
        for key in quantity_keys:
            if key not in record:
                continue
            found = True
            value = record.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid broker quantity field {key}={value!r}") from exc
        if not found:
            raise ValueError("broker position quantity field missing")
        raise ValueError("broker position quantity is null or invalid")
    

    def synchronize_with_broker(
        self, broker_positions: Sequence[Mapping[str, object]]
    ) -> None:
        """Validate and atomically replace managed positions from broker truth."""
        snapshot = decode_position_snapshot(broker_positions)

        def get_float(
            record: Mapping[str, object],
            keys: Sequence[str],
            *,
            default: float = 0.0,
        ) -> float:
            for key in keys:
                if key not in record or record.get(key) is None:
                    continue
                try:
                    value = float(cast(Any, record.get(key)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid broker numeric field {key}={record.get(key)!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"invalid broker numeric field {key}={record.get(key)!r}"
                    )
                return value
            return float(default)

        with self._lock:
            existing_positions = copy.deepcopy(self._positions)
            reconciled: Dict[str, Position] = {}
            snapshot_realized_pnl = 0.0
            snapshot_realized_seen = False
            for row in snapshot.rows:
                record = row.raw
                symbol = row.symbol
                if not is_strategy_instrument(symbol):
                    continue
                product = str(record.get("product") or "").strip().upper()
                if product != "MIS":
                    if symbol in existing_positions:
                        raise ValueError(
                            f"managed broker position {symbol} has unexpected product "
                            f"{product or 'missing'}"
                        )
                    continue
                quantity = row.quantity
                realized_pnl = get_float(
                    record, ("realised", "realized"), default=0.0
                )
                if "realised" in record or "realized" in record:
                    snapshot_realized_seen = True
                    snapshot_realized_pnl += realized_pnl
                if quantity == 0:
                    continue
                side: Side = "LONG" if quantity > 0 else "SHORT"
                abs_quantity = abs(quantity)
                entry_price = get_float(
                    record,
                    ("average_price", "avg_price", "price", "buy_price"),
                )
                current_price = get_float(
                    record,
                    ("last_price", "ltp", "close", "sell_price"),
                    default=entry_price,
                )
                if entry_price <= 0.0 and current_price > 0.0:
                    entry_price = current_price
                if current_price <= 0.0 and entry_price > 0.0:
                    current_price = entry_price
                if entry_price <= 0.0 or current_price <= 0.0:
                    raise ValueError(f"broker position {symbol} has no valid price")
                existing = existing_positions.get(symbol)
                if existing is None:
                    position = self._create_position(
                        symbol=symbol,
                        quantity=abs_quantity,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        realized_pnl=realized_pnl,
                        source="broker_sync",
                    )
                else:
                    position = self._update_position(
                        position=existing,
                        quantity=abs_quantity,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        realized_pnl=realized_pnl,
                        source="broker_sync",
                    )
                reconciled[symbol] = position

            old_keys = set(self._positions)
            new_keys = set(reconciled)
            removed_symbols = sorted(old_keys - new_keys)
            added_symbols = sorted(new_keys - old_keys)
            now = _now()
            for order in self._orders.values():
                if (
                    order.symbol in removed_symbols
                    and order.intent in ("EXIT", "REDUCE")
                    and order.status not in self.FINAL_STATUSES
                ):
                    lifecycle = self._exit_lifecycles.get(order.order_id)
                    if lifecycle is None:
                        lifecycle = ExitLifecycleRecord(
                            symbol=order.symbol,
                            exit_order_id=order.order_id,
                            linked_entry_order_id=order.linked_entry_order_id,
                            trade_lifecycle_id=order.trade_lifecycle_id,
                            bracket_id=order.bracket_id,
                            expected_exit_side=order.side,
                            expected_exit_quantity=order.quantity,
                        )
                        self._exit_lifecycles[order.order_id] = lifecycle
                    lifecycle.state = "BROKER_FLAT_AWAITING_FILL"
                    lifecycle.broker_flat_at = now
            self._positions = reconciled
            if snapshot_realized_seen:
                self._broker_realized_pnl = float(snapshot_realized_pnl)
                self._refresh_realized_pnl_locked()

        if removed_symbols:
            hook = getattr(self, "_on_symbols_flat_hook", None)
            if hook is not None:
                try:
                    hook(list(removed_symbols))
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in on_symbols_flat hook: %s",
                        exc,
                        extra={"event": "position_manager_flat_hook_error"},
                    )
        if not old_keys and not new_keys and not snapshot_realized_seen:
            return
        self.save_state()
        self._logger.info(
            "POSITION_SYNC_COMMITTED total=%s added=%s removed=%s "
            "realized_authoritative=%s",
            len(reconciled),
            len(added_symbols),
            len(removed_symbols),
            snapshot_realized_seen,
            extra={
                "event": "POSITION_SYNC_COMMITTED",
                "total_managed": len(reconciled),
                "added": len(added_symbols),
                "removed": len(removed_symbols),
                "realized_pnl_authoritative": snapshot_realized_seen,
            },
        )

    def _create_position(
        self,
        *,
        symbol: str,
        quantity: int,
        side: Side,
        entry_price: float,
        current_price: float,
        realized_pnl: float,
        source: str,
    ) -> Position:
        """Create a :class:`Position` from broker sync metadata.

        Args:
            symbol: Trading symbol to map.
            quantity: Absolute quantity to record.
            side: Side of the position, long or short.
            entry_price: Average broker entry price.
            current_price: Latest mark price for the symbol.
            realized_pnl: Broker-reported realised profit/loss.
            source: Human readable provenance label for logging.

        Returns:
            Constructed :class:`Position` instance.

        Raises:
            Exception: Propagates unexpected errors after logging.
        """

        self._logger.debug(
            "Entered _create_position",
            extra={
                "event": "position_manager_sync_create",
                "symbol": symbol,
                "source": source,
            },
        )
        try:
            normalized_side = _normalize_side(str(side))
            normalized_quantity = int(max(quantity, 0))
            entry_value = (
                float(entry_price) if entry_price > 0.0 else float(current_price)
            )
            if entry_value < 0.0:
                entry_value = 0.0
            mark_value = float(current_price) if current_price > 0.0 else entry_value
            position = Position(
                symbol=str(symbol).strip().upper(),
                side=normalized_side,
                quantity=normalized_quantity,
                entry_price=entry_value,
                entry_time=_now(),
                current_price=mark_value,
                realized_pnl=float(realized_pnl),
            )
        except Exception as exc:  # noqa: BLE001 - defensive construct
            self._logger.error(
                "Failure in _create_position: %s",
                exc,
                extra={
                    "event": "position_manager_sync_create_error",
                    "symbol": symbol,
                    "source": source,
                },
                exc_info=exc,
            )
            raise
        return position

    def _update_position(
        self,
        *,
        position: Position,
        quantity: int,
        side: Side,
        entry_price: float,
        current_price: float,
        realized_pnl: float,
        source: str,
    ) -> Position:
        """Update *position* with broker-provided sync metadata.

        Args:
            position: Existing local position to mutate.
            quantity: Absolute quantity to record.
            side: Side of the position, long or short.
            entry_price: Broker average entry price.
            current_price: Broker latest mark price.
            realized_pnl: Broker realised PnL snapshot.
            source: Human readable provenance label for logging.

        Returns:
            Updated :class:`Position` instance.

        Raises:
            Exception: Propagates unexpected errors after logging.
        """

        self._logger.debug(
            "Entered _update_position",
            extra={
                "event": "position_manager_sync_update_apply",
                "symbol": position.symbol,
                "source": source,
            },
        )
        try:
            position.side = _normalize_side(str(side))
            position.quantity = int(max(quantity, 0))
            if entry_price > 0.0:
                position.entry_price = float(entry_price)
            if current_price > 0.0:
                position.current_price = float(current_price)
            position.realized_pnl = float(realized_pnl)
            if position.entry_time.tzinfo is None:
                position.entry_time = position.entry_time.replace(tzinfo=timezone.utc)
        except Exception as exc:  # noqa: BLE001 - defensive update
            self._logger.error(
                "Failure in _update_position: %s",
                exc,
                extra={
                    "event": "position_manager_sync_update_error",
                    "symbol": position.symbol,
                    "source": source,
                },
                exc_info=exc,
            )
            raise
        return position

    def _persist_positions_snapshot(self) -> None:
        """Persist position and order snapshots captured from one locked state instant."""
        manager = self._persistent_state
        if manager is None:
            return
        with self._lock:
            position_snapshot = [
                position.to_dict() for position in self._positions.values()
            ]
            order_snapshot = [order.to_dict() for order in self._orders.values()]

        current_symbols = {
            str(entry.get("symbol", "")).strip().upper()
            for entry in position_snapshot
            if str(entry.get("symbol", "")).strip()
        }
        try:
            stored = manager.load_positions()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_positions_snapshot: %s", exc)
            stored = []
        stored_symbols = {
            str(item.get("symbol", "")).strip().upper()
            for item in stored
            if isinstance(item, Mapping)
        }
        for entry in position_snapshot:
            try:
                manager.save_position(entry)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _persist_positions_snapshot save: %s", exc
                )
        for symbol in stored_symbols - current_symbols:
            try:
                manager.save_position({"symbol": symbol, "quantity": 0})
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _persist_positions_snapshot remove: %s", exc
                )
        for payload in order_snapshot:
            try:
                manager.save_order(payload)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _persist_positions_snapshot order save: %s", exc
                )
        try:
            manager.flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_positions_snapshot flush: %s", exc)

    def _maybe_flush_persistent_state(self) -> None:
        """Flush persistence backend when queue depth or age exceeds bounds.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _maybe_flush_persistent_state",
            extra={"event": "position_manager_persistence_check"},
        )
        manager = self._persistent_state
        if manager is None:
            return
        now = time.monotonic()
        if (now - self._last_persistence_check) < self._persistence_flush_interval_s:
            return
        self._last_persistence_check = now
        try:
            telemetry = manager.telemetry()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _maybe_flush_persistent_state telemetry: %s",
                exc,
            )
            return
        pending_source = telemetry.get("pending_events")
        if not isinstance(pending_source, (int, float)):
            pending_source = telemetry.get("pending_queue_depth")
        pending_events = (
            int(pending_source) if isinstance(pending_source, (int, float)) else 0
        )
        last_flush_epoch = telemetry.get("last_flush_epoch")
        should_flush = False
        reason = "queue_depth"
        if pending_events >= self._persistence_pending_threshold:
            should_flush = True
            reason = "queue_depth"
        elif pending_events > 0:
            if isinstance(last_flush_epoch, (int, float)):
                age_seconds = time.time() - float(last_flush_epoch)
                if age_seconds >= self._persistence_max_age_s:
                    should_flush = True
                    reason = "stale_age"
            else:
                should_flush = True
                reason = "unknown_age"
        if not should_flush:
            return
        self._logger.info(
            "Condition met: persistence_flush_required",
            extra={
                "event": "position_manager_persistence_flush",
                "reason": reason,
                "pending": pending_events,
            },
        )
        try:
            manager.flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _maybe_flush_persistent_state: %s",
                exc,
            )

    def _evict_old_terminal_orders(self) -> None:
        """Keep durable terminal idempotency records bounded deterministically."""

        overflow = len(self._terminal_orders) - self._max_terminal_orders
        if overflow <= 0:
            return
        ordered = sorted(
            (
                (order_id, metadata)
                for order_id, metadata in self._terminal_orders.items()
                if metadata.lifecycle_resolved
                and order_id not in self._unresolved_terminal_orders
            ),
            key=lambda item: (item[1].terminal_at, item[0]),
        )
        for order_id, _metadata in ordered[:overflow]:
            self._terminal_orders.pop(order_id, None)

    def _persist_fill(
        self,
        order: Order,
        quantity: int,
        fill_price: float,
        *,
        lifecycle_applied: bool,
        accounting_finalized: bool,
        pnl_applied: bool = False,
        position_applied: bool = False,
        lifecycle_resolved: bool = False,
    ) -> None:
        """Persist executed fill metadata to durable storage.

        Args:
            order: Order instance representing the executed trade.
            quantity: Absolute filled quantity for the execution.
            fill_price: Executed price for the fill.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _persist_fill",
            extra={
                "event": "position_manager_persist_fill",
                "order_id": order.order_id,
            },
        )
        manager = self._persistent_state
        if manager is None:
            return
        timestamp = order.timestamp
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            timestamp_iso = timestamp.astimezone(timezone.utc).isoformat()
        else:
            timestamp_iso = datetime.now(timezone.utc).isoformat()
        payload: dict[str, object] = {
            "fill_id": (
                f"{order.order_id}:{order.applied_filled_quantity + int(quantity)}"
            ),
            "order_id": order.order_id,
            "intent": order.intent,
            "bracket_id": order.bracket_id,
            "signal_id": order.signal_id,
            "signal_fingerprint": order.signal_fingerprint,
            "symbol": order.symbol,
            "side": order.side,
            "quantity_delta": int(quantity),
            "cumulative_filled_quantity": int(
                order.applied_filled_quantity + int(quantity)
            ),
            "fill_price": float(fill_price),
            "status": order.status,
            "timestamp": timestamp_iso,
            "broker_order_timestamp": timestamp_iso,
            "exchange_timestamp": timestamp_iso,
            "exchange_update_timestamp": timestamp_iso,
            "applied_cumulative_notional": float(
                order.applied_cumulative_notional + (float(fill_price) * int(quantity))
            ),
            "last_cumulative_average_price": order.fill_price,
            "lifecycle_applied": bool(lifecycle_applied),
            "position_applied": bool(position_applied),
            "pnl_applied": bool(pnl_applied),
            "accounting_finalized": bool(accounting_finalized),
            "lifecycle_resolved": bool(lifecycle_resolved),
        }
        linked_symbol = getattr(order, "linked_position_symbol", None)
        if linked_symbol:
            payload["linked_position_symbol"] = linked_symbol
        try:
            manager.save_fill(payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_fill: %s", exc)
        else:
            try:
                manager.flush()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _persist_fill flush: %s",
                    exc,
                )

    def _persist_order_state(self, order: Order) -> None:
        """Persist *order* snapshot to the persistent state manager.

        Args:
            order: Order instance requiring persistence.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _persist_order_state",
            extra={
                "event": "position_manager_persist_order",
                "order_id": order.order_id,
            },
        )
        manager = self._persistent_state
        if manager is None:
            return
        try:
            manager.save_order(order.to_dict())
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_order_state: %s", exc)

    def _persist_order_snapshots(self, manager: "PersistentStateManager") -> None:
        """Persist all tracked orders using *manager*.

        Args:
            manager: Persistent state manager coordinating disk writes.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _persist_order_snapshots",
            extra={"event": "position_manager_persist_orders"},
        )
        orders_snapshot = [order.to_dict() for order in self._orders.values()]
        for payload in orders_snapshot:
            try:
                manager.save_order(payload)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _persist_order_snapshots: %s",
                    exc,
                )

    def reset_daily_pnl(self) -> None:
        """Reset realized profit and loss at the start of a new session."""

        self._daily_realized_pnl = 0.0
        for position in self._positions.values():
            position.realized_pnl = 0.0
        self.save_state()

    # Internal helpers -------------------------------------------------

    def _handle_filled_order(self, order: Order) -> FillApplicationResult:
        symbol_key = order.symbol
        cumulative_qty = (
            order.quantity if order.filled_quantity == 0 else order.filled_quantity
        )
        cumulative_qty = int(cumulative_qty)
        previous_qty = int(order.applied_filled_quantity or 0)
        if cumulative_qty > int(order.quantity):
            self._logger.warning(
                "Ignoring overfilled broker update for order %s: %s > %s",
                order.order_id,
                cumulative_qty,
                order.quantity,
            )
            return FillApplicationResult(reason="cumulative_quantity_exceeds_order")
        qty = cumulative_qty - previous_qty
        if qty <= 0:
            # Correct idempotency drop — but the same terminal fill can be
            # replayed by every reconcile cycle; warn once per (order,
            # cumulative snapshot) instead of flooding logs and Telegram.
            _dedupe_key = (str(order.order_id), int(cumulative_qty))
            _seen = getattr(self, "_non_incremental_fill_warned", None)
            if _seen is None:
                _seen = set()
                self._non_incremental_fill_warned = _seen
            if _dedupe_key not in _seen:
                _seen.add(_dedupe_key)
                self._logger.warning(
                    "Ignoring non-incremental fill for order %s (cumulative=%s; further replays suppressed)",
                    order.order_id,
                    cumulative_qty,
                )
            return FillApplicationResult(reason="non_incremental_cumulative_quantity")

        cumulative_avg = order.fill_price
        if cumulative_avg is None or not math.isfinite(float(cumulative_avg)) or float(cumulative_avg) <= 0:
            self._logger.warning("Missing/invalid cumulative fill price for order %s", order.order_id)
            return FillApplicationResult(reason="invalid_cumulative_average_price")

        new_cumulative_notional = cumulative_qty * float(cumulative_avg)
        delta_notional = new_cumulative_notional - float(order.applied_cumulative_notional or 0.0)
        if delta_notional <= 0 or not math.isfinite(delta_notional):
            self._logger.warning(
                "Ignoring invalid cumulative notional for order %s", order.order_id
            )
            return FillApplicationResult(reason="invalid_cumulative_notional")
        fill_price = delta_notional / qty
        if not math.isfinite(fill_price) or fill_price <= 0:
            self._logger.warning("Invalid delta fill price for order %s", order.order_id)
            return FillApplicationResult(reason="invalid_delta_fill_price")

        side = order.side
        intent = _normalize_intent(order.intent)
        is_terminal = order.status == "FILLED"

        def mark_applied() -> None:
            order.applied_filled_quantity += qty
            order.applied_cumulative_notional = new_cumulative_notional
            order.last_cumulative_average_price = float(cumulative_avg)

        if not self.has_position(symbol_key):
            if intent in ("EXIT", "REDUCE"):
                entry_price = order.pre_order_entry_price
                if entry_price is None or entry_price <= 0:
                    self._logger.warning(
                        "Exit fill while flat is retained until linked entry price is available",
                        extra={
                            "event": "exit_fill_without_entry_price",
                            "order_id": order.order_id,
                            "symbol": symbol_key,
                            "intent": intent,
                        },
                    )
                    return FillApplicationResult(
                        quantity_delta=qty,
                        delta_fill_price=fill_price,
                        reason="linked_entry_price_missing",
                    )
                self._persist_fill(
                    order,
                    qty,
                    fill_price,
                    lifecycle_applied=True,
                    position_applied=False,
                    pnl_applied=True,
                    accounting_finalized=is_terminal,
                    lifecycle_resolved=is_terminal,
                )
                position_side = order.pre_order_position_side or (
                    "LONG" if side == "SELL" else "SHORT"
                )
                realized = self._calculate_realized_pnl(
                    position_side,
                    float(entry_price),
                    fill_price,
                    min(qty, order.pre_order_quantity or qty),
                )
                self._local_realized_pnl += realized
                with self._lock:
                    self._refresh_realized_pnl_locked()
                lifecycle = self._exit_lifecycles.get(order.order_id)
                if lifecycle is not None:
                    lifecycle.final_fill_price = float(cumulative_avg)
                    lifecycle.state = "EXIT_FINALIZED" if is_terminal else "EXIT_PARTIALLY_FILLED"
                    if is_terminal:
                        lifecycle.finalized_at = _now()
                mark_applied()
                return FillApplicationResult(
                    fill_recorded=True,
                    position_applied=False,
                    bracket_applied=False,
                    pnl_applied=True,
                    accounting_finalized=is_terminal,
                    lifecycle_resolved=is_terminal,
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="exit_fill_finalized_after_broker_flat" if is_terminal else "exit_partial_after_broker_flat",
                )

            if intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                self._logger.warning(
                    "Ignoring %s %s fill while flat; explicit entry intent required",
                    intent,
                    side,
                    extra={
                        "event": "ambiguous_fill_quarantined",
                        "order_id": order.order_id,
                        "symbol": symbol_key,
                        "side": side,
                        "intent": intent,
                    },
                )
                return FillApplicationResult(
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="ambiguous_fill_quarantined",
                )
            paired_exit = next(
                (
                    metadata
                    for metadata in sorted(
                        self._terminal_orders.values(),
                        key=lambda item: item.terminal_at,
                        reverse=True,
                    )
                    if metadata.symbol == symbol_key
                    and metadata.intent in ("EXIT", "REDUCE")
                    and metadata.side == "SELL"
                    and metadata.average_fill_price is not None
                    and (
                        metadata.linked_entry_order_id == order.order_id
                        or (
                            metadata.trade_lifecycle_id is not None
                            and metadata.trade_lifecycle_id == order.trade_lifecycle_id
                        )
                    )
                ),
                None,
            )
            if intent in ("ENTRY", "SCALE_IN") and side == "BUY" and paired_exit:
                if paired_exit.pnl_applied and paired_exit.accounting_finalized:
                    self._persist_fill(
                        order,
                        qty,
                        fill_price,
                        lifecycle_applied=True,
                        position_applied=False,
                        pnl_applied=False,
                        accounting_finalized=True,
                        lifecycle_resolved=True,
                    )
                    mark_applied()
                    return FillApplicationResult(
                        fill_recorded=True,
                        position_applied=False,
                        pnl_applied=False,
                        accounting_finalized=True,
                        lifecycle_resolved=True,
                        quantity_delta=qty,
                        delta_fill_price=fill_price,
                        reason="historical_entry_fill_recorded_after_finalized_exit",
                    )
                self._persist_fill(
                    order,
                    qty,
                    fill_price,
                    lifecycle_applied=True,
                    position_applied=False,
                    pnl_applied=True,
                    accounting_finalized=True,
                    lifecycle_resolved=True,
                )
                paired_qty = min(qty, abs(int(paired_exit.cumulative_filled_quantity)))
                realized = (float(paired_exit.average_fill_price) - fill_price) * paired_qty
                self._local_realized_pnl += realized
                with self._lock:
                    self._refresh_realized_pnl_locked()
                mark_applied()
                self._logger.warning(
                    "historical_entry_fill_reconciled_after_exit",
                    extra={
                        "event": "historical_entry_fill_reconciled_after_exit",
                        "order_id": order.order_id,
                        "symbol": symbol_key,
                        "quantity": paired_qty,
                        "realized_pnl": realized,
                    },
                )
                return FillApplicationResult(
                    fill_recorded=True,
                    position_applied=False,
                    pnl_applied=True,
                    accounting_finalized=True,
                    lifecycle_resolved=True,
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="historical_entry_fill_reconciled_after_exit",
                )
            self._persist_fill(
                order,
                qty,
                fill_price,
                lifecycle_applied=True,
                position_applied=True,
                accounting_finalized=False,
                lifecycle_resolved=False,
            )
            position_side: Side = "LONG" if side == "BUY" else "SHORT"
            self.open_position(
                symbol=symbol_key,
                side=position_side,
                quantity=qty,
                entry_price=fill_price,
                order_id=order.order_id,
            )
            order.protection_confirmed = False
            order.protection_failure_reason = "entry_filled_unprotected"
            mark_applied()
            return FillApplicationResult(
                fill_recorded=True,
                position_applied=True,
                bracket_applied=False,
                accounting_finalized=False,
                lifecycle_resolved=False,
                quantity_delta=qty,
                delta_fill_price=fill_price,
                reason="entry_filled_unprotected",
            )

        position = self._positions[symbol_key]
        entry_side_matches = (
            (position.side == "LONG" and side == "BUY")
            or (position.side == "SHORT" and side == "SELL")
        )
        if intent in ("ENTRY", "SCALE_IN", "REVERSAL") and entry_side_matches:
            expected_post_fill_qty = int(order.pre_order_quantity or 0) + cumulative_qty
            if position.quantity >= expected_post_fill_qty:
                # Broker position snapshots are absolute state. If the authoritative
                # quantity already includes this cumulative fill, record the fill
                # lifecycle but do not apply the quantity delta a second time.
                self._persist_fill(
                    order,
                    qty,
                    fill_price,
                    lifecycle_applied=True,
                    position_applied=True,
                    accounting_finalized=False,
                    lifecycle_resolved=False,
                )
                if order.pre_order_quantity == 0 and position.order_id is None:
                    position.order_id = order.order_id
                order.protection_confirmed = False
                order.protection_failure_reason = "entry_protection_incomplete"
                mark_applied()
                self._logger.info(
                    "ENTRY_FILL_ALREADY_REFLECTED_BY_BROKER_SYNC "
                    "order_id=%s symbol=%s broker_qty=%s expected_qty=%s "
                    "cumulative_fill_qty=%s",
                    order.order_id,
                    symbol_key,
                    position.quantity,
                    expected_post_fill_qty,
                    cumulative_qty,
                    extra={
                        "event": "ENTRY_FILL_ALREADY_REFLECTED_BY_BROKER_SYNC",
                        "order_id": order.order_id,
                        "symbol": symbol_key,
                        "broker_quantity": position.quantity,
                        "expected_post_fill_quantity": expected_post_fill_qty,
                        "cumulative_filled_quantity": cumulative_qty,
                    },
                )
                return FillApplicationResult(
                    fill_recorded=True,
                    position_applied=True,
                    bracket_applied=False,
                    accounting_finalized=False,
                    lifecycle_resolved=False,
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="entry_fill_already_reflected_by_broker_sync",
                )

        if (position.side == "LONG" and side == "SELL") or (
            position.side == "SHORT" and side == "BUY"
        ):
            self._persist_fill(
                order,
                qty,
                fill_price,
                lifecycle_applied=True,
                position_applied=True,
                pnl_applied=True,
                accounting_finalized=is_terminal,
                lifecycle_resolved=is_terminal,
            )
            self._reduce_or_close_position(position, qty, fill_price)
            lifecycle = self._exit_lifecycles.get(order.order_id)
            if lifecycle is not None:
                lifecycle.final_fill_price = float(cumulative_avg)
                lifecycle.state = "EXIT_FINALIZED" if is_terminal else "EXIT_PARTIALLY_FILLED"
                if is_terminal:
                    lifecycle.finalized_at = _now()
            mark_applied()
            return FillApplicationResult(
                fill_recorded=True,
                position_applied=True,
                pnl_applied=True,
                accounting_finalized=is_terminal,
                lifecycle_resolved=is_terminal,
                quantity_delta=qty,
                delta_fill_price=fill_price,
                reason="exit_fill_applied",
            )
        else:
            if intent in ("EXIT", "REDUCE"):
                self._logger.warning(
                    "Ignoring exit fill that does not match open position side",
                    extra={
                        "event": "exit_fill_side_mismatch",
                        "order_id": order.order_id,
                        "symbol": symbol_key,
                        "position_side": position.side,
                        "order_side": side,
                    },
                )
                return FillApplicationResult(
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="exit_fill_side_mismatch",
                )
            self._persist_fill(
                order,
                qty,
                fill_price,
                lifecycle_applied=True,
                position_applied=True,
                accounting_finalized=False,
                lifecycle_resolved=False,
            )
            self._scale_position(position, qty, fill_price)
            order.protection_confirmed = False
            order.protection_failure_reason = "entry_protection_incomplete"
            mark_applied()
            return FillApplicationResult(
                fill_recorded=True,
                position_applied=True,
                bracket_applied=False,
                accounting_finalized=False,
                lifecycle_resolved=False,
                quantity_delta=qty,
                delta_fill_price=fill_price,
                reason="scale_fill_unprotected",
            )

    def _scale_position(self, position: Position, qty: int, fill_price: float) -> None:
        new_qty = position.quantity + qty
        if new_qty <= 0:
            self._logger.warning(
                "Scaling produced non-positive quantity for %s", position.symbol
            )
            return
        position.entry_price = (
            (position.entry_price * position.quantity) + (fill_price * qty)
        ) / new_qty
        position.quantity = new_qty
        position.current_price = fill_price
        self._logger.info(
            "Scaled position %s to quantity %s", position.symbol, position.quantity
        )

    def _reduce_or_close_position(
        self, position: Position, qty: int, fill_price: float
    ) -> None:
        reduce_qty = min(qty, position.quantity)
        realized = self._calculate_realized_pnl(
            position.side, position.entry_price, fill_price, reduce_qty
        )
        position.quantity -= reduce_qty
        position.realized_pnl += realized
        self._local_realized_pnl += realized
        with self._lock:
            self._refresh_realized_pnl_locked()
        position.current_price = fill_price
        if position.quantity == 0:
            self._logger.info("Position %s fully closed via order", position.symbol)
            del self._positions[position.symbol]
            self.clear_active_contract_by_symbol(position.symbol)
        else:
            self._logger.info(
                "Reduced position %s by %s (remaining=%s)",
                position.symbol,
                reduce_qty,
                position.quantity,
            )

    @staticmethod
    def _calculate_realized_pnl(
        side: Side, entry_price: float, exit_price: float, qty: int
    ) -> float:
        if side == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty


__all__ = [
    "Order",
    "OrderIntent",
    "Position",
    "PositionManager",
    "ActiveContract",
    "TerminalOrderMetadata",
    "normalize_broker_order_status",
]
