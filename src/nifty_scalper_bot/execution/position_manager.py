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
from nifty_scalper_bot.execution.position_reconciliation_identity import (
    _canonicalize_position_store,
    _merge_cost_basis_quarantine,
    _prepare_broker_positions,
    _prepared_row_symbol,
    _restore_owned_position_lifecycle,
    _snapshot_owned_position_lifecycle,
)
from nifty_scalper_bot.execution.position_snapshot import (
    BrokerExposureState,
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.reasons import canonical
from nifty_scalper_bot.utils.symbols import is_strategy_instrument, normalize_symbol

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
_EXIT_RECONCILIATION_GRACE_DEFAULT_S = 2.0
_EXIT_RECONCILIATION_GRACE_MIN_S = 0.25
_EXIT_RECONCILIATION_GRACE_MAX_S = 5.0
_BROKER_POSITION_SNAPSHOT_MAX_AGE_DEFAULT_S = 20.0
_BROKER_POSITION_SNAPSHOT_MAX_AGE_MIN_S = 1.0
_BROKER_POSITION_SNAPSHOT_MAX_AGE_MAX_S = 300.0


def _resolve_broker_position_snapshot_max_age_seconds() -> float:
    raw = os.getenv("BROKER_POSITION_SNAPSHOT_MAX_AGE_SECONDS")
    if raw is None or str(raw).strip() == "":
        return _BROKER_POSITION_SNAPSHOT_MAX_AGE_DEFAULT_S
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return _BROKER_POSITION_SNAPSHOT_MAX_AGE_DEFAULT_S
    if not math.isfinite(value):
        return _BROKER_POSITION_SNAPSHOT_MAX_AGE_DEFAULT_S
    return min(
        _BROKER_POSITION_SNAPSHOT_MAX_AGE_MAX_S,
        max(_BROKER_POSITION_SNAPSHOT_MAX_AGE_MIN_S, value),
    )


def _resolve_exit_reconciliation_grace_seconds() -> float:
    raw = os.getenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS")
    if raw is None or str(raw).strip() == "":
        return _EXIT_RECONCILIATION_GRACE_DEFAULT_S
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return _EXIT_RECONCILIATION_GRACE_DEFAULT_S
    if not math.isfinite(value):
        return _EXIT_RECONCILIATION_GRACE_DEFAULT_S
    return min(
        _EXIT_RECONCILIATION_GRACE_MAX_S,
        max(_EXIT_RECONCILIATION_GRACE_MIN_S, value),
    )


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
    """Persist *payload* to *path* atomically with Enum handling (Thread-Safe)."""
    import json
    import os
    import uuid
    from contextlib import suppress
    from enum import Enum
    from datetime import datetime, date
    from decimal import Decimal

    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value if hasattr(obj, "value") else obj.name
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                return obj.__dict__
            return super().default(obj)

    def _sanitize(obj):
        """Recursively convert non-JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_sanitize(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value if hasattr(obj, "value") else obj.name
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, "to_dict"):
            return _sanitize(obj.to_dict())
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            return _sanitize(vars(obj))
        return obj

    sanitized_payload = _sanitize(dict(payload))
    temp_path = path.with_suffix(f"{path.suffix}.tmp.{uuid.uuid4().hex}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                sanitized_payload,
                f,
                indent=2,
                sort_keys=True,
                cls=EnhancedJSONEncoder,
                default=str,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception as exc:  # noqa: BLE001
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
    state: str | None = None

    @property
    def unrealized_pnl(self) -> float:
        direction = 1 if self.side == "LONG" else -1
        return (self.current_price - self.entry_price) * self.quantity * direction

    @property
    def unrealized_pnl_pct(self) -> float:
        direction = 1 if self.side == "LONG" else -1
        notional = abs(self.entry_price * self.quantity)
        if notional == 0:
            return 0.0
        return (self.unrealized_pnl / notional) * 100.0

    @property
    def age_seconds(self) -> float:
        return max((_now() - self.entry_time).total_seconds(), 0.0)

    def to_dict(self) -> dict[str, object]:
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
            state=(
                str(payload.get("state")) if payload.get("state") is not None else None
            ),
        )


@dataclass(slots=True)
class ExitSettlementGuard:
    completed_exit_at_monotonic: float
    grace_until_monotonic: float
    stale_snapshot_count: int = 0
    last_stale_quantity: int = 0
    last_log_monotonic: float = 0.0


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
            "broker_flat_at": (
                self.broker_flat_at.isoformat() if self.broker_flat_at else None
            ),
            "final_fill_price": self.final_fill_price,
            "finalized_at": (
                self.finalized_at.isoformat() if self.finalized_at else None
            ),
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
            expected_exit_side=_normalize_order_side(
                str(payload["expected_exit_side"])
            ),
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
            pre_order_entry_price=_to_optional_float(
                payload.get("pre_order_entry_price")
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
class ActiveContract:
    underlying: str
    symbol: str
    option_type: str
    strike: float
    expiry: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "underlying": self.underlying,
            "symbol": self.symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ActiveContract":
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
        self._trades_today_date: str | None = None
        self._trades_today_count: int = 0
        self._order_locks: dict[str, threading.RLock] = {}
        self._symbol_lifecycle_locks: dict[str, threading.RLock] = {}
        self._orders: Dict[str, Order] = {}
        self._terminal_orders: dict[str, TerminalOrderMetadata] = {}
        self._unresolved_terminal_orders: dict[str, TerminalOrderMetadata] = {}
        self._exit_lifecycles: dict[str, ExitLifecycleRecord] = {}
        self._broker_order_ledger: dict[str, dict[str, Any]] = {}
        self._quarantined_broker_exposures: dict[str, dict[str, Any]] = {}
        self._max_terminal_orders = 5000
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
        self._single_reconcile_lock = threading.Lock()
        self._single_reconcile_generation = 0
        self._single_reconcile_coalesced = 0
        self._cost_basis_unresolved_symbols: set[str] = set()
        self._reconcile_interval_s: float = 60.0
        self._reconcile_retry_interval_s: float = 10.0
        self._reconcile_listeners: list[Callable[[str, Mapping[str, object]], None]] = []
        self._last_reconciled_state: Dict[str, Position] = {}
        self._last_broker_position_snapshot_at: float | None = None
        self._last_broker_position_snapshot_mono: float | None = None
        self._last_broker_position_snapshot_valid: bool = False
        self._last_broker_quantities_by_symbol: dict[str, int] = {}
        self._last_broker_position_snapshot_source: str | None = None
        self._last_broker_position_snapshot_failure_at: float | None = None
        self._last_broker_position_snapshot_failure_reason: str | None = None
        self._broker_position_snapshot_max_age_seconds = (
            _resolve_broker_position_snapshot_max_age_seconds()
        )
        self._local_position_generation: int = 0
        self._broker_snapshot_local_generation: int = -1
        self._recently_flat_exit_until_monotonic: dict[str, float] = {}
        self._recently_flat_exit_metadata: dict[str, ExitSettlementGuard] = {}
        self._recently_flat_exit_grace_seconds: float = (
            _resolve_exit_reconciliation_grace_seconds()
        )
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
        self._on_symbols_flat_hook = hook

    @property
    def _processed_order_ids(self) -> set[str]:
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
        self._logger.debug(
            "Entered set_broker_client",
            extra={"event": "position_manager_set_broker"},
        )
        self._broker_client = broker_client

    def _resolve_broker_position_fetcher(self) -> Callable[[], Any] | None:
        broker = self._broker_client
        if broker is None:
            return None
        for name in ("get_positions", "list_positions", "positions", "fetch_positions"):
            fetcher = getattr(broker, name, None)
            if callable(fetcher):
                return cast(Callable[[], Any], fetcher)
        return None

    def _cancel_reconcile_timer(self) -> None:
        timer = self._reconcile_timer
        if timer is not None:
            timer.cancel()
            self._reconcile_timer = None

    def _schedule_reconcile(self, delay_s: float) -> None:
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
            extra={"event": "position_reconcile_schedule", "delay_sec": round(delay, 3)},
        )

    def _compute_retry_delay(self) -> float:
        base_delay = max(self._reconcile_retry_interval_s, _MIN_RECONCILE_DELAY_S)
        failures = max(self._consecutive_reconcile_failures, 1)
        multiplier = min(2 ** (failures - 1), 16.0)
        max_delay = max(self._reconcile_interval_s, base_delay) * 4.0
        return float(min(base_delay * multiplier, max_delay))

    def _schedule_retry_after_failure(self, delay_s: float | None = None) -> None:
        retry_delay = float(delay_s) if delay_s is not None else self._compute_retry_delay()
        self._schedule_reconcile(retry_delay)

    def add_reconcile_listener(
        self, callback: Callable[[str, Mapping[str, object]], None]
    ) -> None:
        if callable(callback):
            self._reconcile_listeners.append(callback)

    def _notify_reconcile_event(
        self, event: str, payload: Mapping[str, object]
    ) -> None:
        for listener in list(self._reconcile_listeners):
            try:
                listener(event, dict(payload))
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _notify_reconcile_event: %s",
                    exc,
                    extra={"event": "position_reconcile_listener_error"},
                )

    def _handle_reconcile_failure(
        self,
        *,
        reason: str,
        error: Exception | None,
        payload_count: int,
        previous_positions: Mapping[str, Position] | None,
    ) -> None:
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
                label="position_reconcile", stage="apply", outcome=reason_token
            )
        except Exception as metrics_exc:  # noqa: BLE001
            self._logger.error("Failure in reconcile failure metrics: %s", metrics_exc)
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
        with suppress(Exception):
            _POSITION_RECONCILE_EVENTS.labels("failed").inc()
        self._notify_reconcile_event("position_reconcile_failed", payload)
        self._schedule_retry_after_failure(retry_delay)

    def _handle_reconcile_success(self, payload_count: int) -> None:
        self._last_reconcile_attempt = _now()
        self._last_reconcile_success_at = self._last_reconcile_attempt
        previous_failures = self._consecutive_reconcile_failures
        self._consecutive_reconcile_failures = 0
        self._last_reconcile_error = None
        event_key = f"success:{self._last_reconcile_success_at.isoformat()}"
        try:
            METRICS.record_broker_sync(
                success=True, reason="ok", latency_seconds=None, event_id=event_key
            )
        except Exception as metrics_exc:  # noqa: BLE001
            self._logger.error("Failure in reconcile success metrics: %s", metrics_exc)
        self._last_reconciled_state = copy.deepcopy(self._positions)
        event_payload: dict[str, object] = {
            "count": payload_count,
            "timestamp": self._last_reconcile_success_at.isoformat(),
        }
        if previous_failures:
            event_payload["previous_failures"] = previous_failures
        with suppress(Exception):
            _POSITION_RECONCILE_EVENTS.labels("ok").inc()
        self._notify_reconcile_event("position_reconcile_ok", event_payload)

    def reconcile_now(self) -> bool:
        lock = self._single_reconcile_lock
        if not lock.acquire(False):
            self._single_reconcile_coalesced += 1
            return bool(self._last_reconcile_success_at)
        try:
            self._single_reconcile_generation += 1
            return bool(self._reconcile_positions_from_broker())
        finally:
            lock.release()

    def _reconcile_positions_from_broker(self) -> bool:
        fetcher = self._resolve_broker_position_fetcher()
        if fetcher is None:
            self._handle_reconcile_failure(
                reason=canonical("fetcher_missing"), error=None, payload_count=0, previous_positions=None
            )
            return False
        try:
            response = fetcher()
            snapshot = decode_position_snapshot(response)
        except Exception as exc:  # noqa: BLE001
            reason = canonical(
                "payload_invalid" if isinstance(exc, PositionSnapshotError) else "fetch_error"
            )
            self._handle_reconcile_failure(
                reason=reason, error=exc, payload_count=0, previous_positions=None
            )
            return False
        payloads = snapshot.raw_rows()
        try:
            self.synchronize_with_broker(payloads)
        except Exception as exc:  # noqa: BLE001
            self._handle_reconcile_failure(
                reason=canonical("apply_error"),
                error=exc,
                payload_count=len(payloads),
                previous_positions=None,
            )
            return False
        self._handle_reconcile_success(len(payloads))
        return True

    def reconcile_periodic(
        self,
        *,
        interval_sec: float | None = None,
        retry_sec: float | None = None,
    ) -> None:
        try:
            if interval_sec is not None:
                self._reconcile_interval_s = max(float(interval_sec), _MIN_RECONCILE_DELAY_S)
            if retry_sec is not None:
                self._reconcile_retry_interval_s = max(float(retry_sec), _MIN_RECONCILE_DELAY_S)
            success = self.reconcile_now()
        except Exception as exc:  # noqa: BLE001
            self._handle_reconcile_failure(
                reason=canonical("periodic_error"), error=exc, payload_count=0, previous_positions=None
            )
            return
        self._maybe_flush_persistent_state()
        if success:
            self._schedule_reconcile(self._reconcile_interval_s)

    def get_active_contract(self, underlying: str) -> ActiveContract | None:
        return self._active_contracts.get(underlying.strip().upper())

    def set_active_contract(
        self, underlying: str, contract: SelectedContract | ActiveContract | None
    ) -> None:
        normalized = underlying.strip().upper()
        if not normalized:
            return
        if contract is None:
            removed = self._active_contracts.pop(normalized, None)
            if removed is not None:
                self._contract_index.pop(removed.symbol, None)
            self.save_state()
            return
        active = ActiveContract.from_selection(normalized, contract)
        self._active_contracts[normalized] = active
        self._contract_index[active.symbol] = normalized
        self.save_state()

    def clear_active_contract(self, underlying: str) -> None:
        self.set_active_contract(underlying, None)

    def clear_active_contract_by_symbol(self, symbol: str) -> None:
        normalized = symbol.strip().upper()
        if not normalized:
            return
        underlying = self._contract_index.pop(normalized, None)
        if underlying and self._active_contracts.pop(underlying, None) is not None:
            self.save_state()

    def _mark_local_position_mutation_locked(self) -> None:
        self._local_position_generation += 1

    def _broker_snapshot_age_locked(self) -> float | None:
        if self._last_broker_position_snapshot_mono is None:
            return None
        return time.monotonic() - self._last_broker_position_snapshot_mono

    def _broker_snapshot_fresh_locked(self) -> tuple[bool, float | None]:
        age = self._broker_snapshot_age_locked()
        if age is None or age < 0 or age > self._broker_position_snapshot_max_age_seconds:
            return False, age
        if self._broker_snapshot_local_generation != self._local_position_generation:
            return False, age
        return True, age

    def is_flat(self, symbol: str) -> bool:
        with self._lock:
            position = self._positions.get(symbol.strip().upper())
        return position is None or position.quantity <= 0

    def broker_exposure_snapshot(self) -> dict[str, object]:
        with self._lock:
            fresh, age = self._broker_snapshot_fresh_locked()
            return {
                "valid": self._last_broker_position_snapshot_valid,
                "fresh": bool(self._last_broker_position_snapshot_valid and fresh),
                "age_seconds": age,
                "max_age_seconds": self._broker_position_snapshot_max_age_seconds,
                "fetched_at": self._last_broker_position_snapshot_at,
                "source": self._last_broker_position_snapshot_source,
                "quantities_by_symbol": dict(self._last_broker_quantities_by_symbol),
                "failure_at": self._last_broker_position_snapshot_failure_at,
                "failure_reason": self._last_broker_position_snapshot_failure_reason,
                "local_position_generation": self._local_position_generation,
                "snapshot_local_generation": self._broker_snapshot_local_generation,
            }

    def broker_exposure_state(self, symbol: str) -> BrokerExposureState:
        lookup = normalize_symbol(symbol) or symbol.strip().upper()
        with self._lock:
            if not self._last_broker_position_snapshot_valid:
                return BrokerExposureState.UNKNOWN
            fresh, _ = self._broker_snapshot_fresh_locked()
            if not fresh:
                return BrokerExposureState.UNKNOWN
            if lookup not in self._last_broker_quantities_by_symbol:
                return BrokerExposureState.ABSENT
            qty = self._last_broker_quantities_by_symbol[lookup]
        return BrokerExposureState.FLAT if qty == 0 else BrokerExposureState.NONZERO

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
            self._clear_recent_exit_guard_locked(symbol_key)
            self._positions[symbol_key] = position
            self._increment_trades_today_locked()
            self._mark_local_position_mutation_locked()
        self.save_state()
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        close_time: datetime | None = None,
    ) -> Position:
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
            self._mark_local_position_mutation_locked()
            self._mark_recent_exit_flat_locked(symbol_key)
        self.clear_active_contract_by_symbol(symbol_key)
        self.save_state()
        return position

    def update_from_order(self, order: Order) -> None:
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
        with self._lock:
            position = self._positions.get(symbol.upper())
            if position is None:
                return
            position.current_price = float(current_price)
        self.save_state()

    def get_position(self, symbol: str) -> Position | None:
        with self._lock:
            return self._positions.get(symbol.upper())

    def get_all_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def get_open_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def has_position(self, symbol: str) -> bool:
        with self._lock:
            return symbol.upper() in self._positions

    def has_open_position(self, symbol: str) -> bool:
        return self.has_position(symbol)

    def get_total_exposure(self) -> float:
        return float(
            sum(abs(position.quantity * position.current_price) for position in self._positions.values())
        )

    def get_net_pnl(self) -> float:
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_unrealized_pnl(self) -> float:
        return float(sum(position.unrealized_pnl for position in self._positions.values()))

    def _refresh_realized_pnl_locked(self) -> None:
        local_confirmed = float(self._local_realized_pnl)
        broker_confirmed = None
        if self._broker_realized_pnl is not None and self._session_opening_realized_baseline is not None:
            broker_confirmed = float(self._broker_realized_pnl) - float(self._session_opening_realized_baseline)
        if local_confirmed != 0.0:
            authoritative = local_confirmed
            authority = "local_confirmed_ledger"
            status = (
                "mismatch"
                if broker_confirmed is not None and abs(local_confirmed - broker_confirmed) > 1.0
                else "matched" if broker_confirmed is not None else "local_only"
            )
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

    def _increment_trades_today_locked(self) -> None:
        today = self._trading_date_ist()
        if self._trades_today_date != today:
            self._trades_today_date = today
            self._trades_today_count = 0
        self._trades_today_count += 1

    def trades_today(self) -> int:
        with self._lock:
            if self._trades_today_date != self._trading_date_ist():
                return 0
            return int(self._trades_today_count)

    @staticmethod
    def _trading_date_ist(now: datetime | None = None) -> str:
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
        value = float(broker_realized)
        if not math.isfinite(value):
            raise ValueError("broker_realized must be finite")
        as_of = snapshot_at or _now()
        session_date = trading_date or self._trading_date_ist(as_of)
        with self._lock:
            if self._session_opening_realized_baseline is not None and self._pnl_trading_date == session_date:
                self._broker_realized_pnl = value
                self._refresh_realized_pnl_locked()
                return False
            if self._pnl_trading_date != session_date:
                self._local_realized_pnl = 0.0
                self._local_provisional_realized_pnl = 0.0
                for position in self._positions.values():
                    position.realized_pnl = 0.0
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
        with self._lock:
            if self._broker_realized_pnl is None or self._session_opening_realized_baseline is None:
                return None
            return float(self._broker_realized_pnl) - float(self._session_opening_realized_baseline)

    def get_realized_pnl(self) -> float:
        with self._lock:
            return float(self._daily_realized_pnl)

    def pnl_reconciliation_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "local_confirmed_realized": float(self._local_realized_pnl),
                "local_provisional_realized": float(self._local_provisional_realized_pnl),
                "broker_realized_snapshot": self._broker_realized_pnl,
                "broker_session_realized": self.broker_session_realized_pnl(),
                "authoritative_realized": float(self._authoritative_realized_pnl),
                "pnl_authority": self._pnl_authority,
                "pnl_reconciliation_status": self._pnl_reconciliation_status,
                "session_opening_realized_baseline": self._session_opening_realized_baseline,
                "pnl_trading_date": self._pnl_trading_date,
                "pnl_account_fingerprint": self._pnl_account_fingerprint,
                "pnl_product_scope": self._pnl_product_scope,
                "baseline_established_at": self._baseline_established_at.isoformat() if self._baseline_established_at else None,
                "baseline_source": self._baseline_source,
                "pnl_snapshot_at": self._pnl_snapshot_at.isoformat() if self._pnl_snapshot_at else None,
            }

    def current_pnl_reconciliation_blocker(self) -> str | None:
        with self._lock:
            if self._require_pnl_baseline_for_entries and self._session_opening_realized_baseline is None:
                return "pnl_baseline_uninitialized"
            if self._require_pnl_baseline_for_entries and self._pnl_trading_date != self._trading_date_ist():
                return "pnl_session_date_unverified"
            if self._pnl_reconciliation_status == "mismatch":
                return "pnl_reconciliation_mismatch"
            return None

    def require_pnl_session_baseline(self, required: bool = True) -> None:
        with self._lock:
            self._require_pnl_baseline_for_entries = bool(required)

    def current_entry_protection_blocker(self, symbol: str | None = None) -> str | None:
        symbol_key = normalize_symbol(symbol) if symbol else None
        with self._lock:
            exposures = self._quarantined_broker_exposures
            if symbol_key is not None:
                exposure = exposures.get(symbol_key)
                if exposure is not None:
                    if str(exposure.get("reason") or "") == "broker_state_unverified":
                        return "broker_state_unverified"
                    return "broker_exposure_quarantined"
            elif exposures:
                if any(
                    str(exposure.get("reason") or "") == "broker_state_unverified"
                    for exposure in exposures.values()
                ):
                    return "broker_state_unverified"
                return "broker_exposure_quarantined"
            for order in self._orders.values():
                if symbol_key is not None and order.symbol != symbol_key:
                    continue
                if order.intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                    continue
                if order.applied_filled_quantity <= 0:
                    continue
                if not order.protection_confirmed or order.protected_quantity < order.applied_filled_quantity:
                    return "entry_protection_incomplete"
            for metadata in self._unresolved_terminal_orders.values():
                if symbol_key is not None and metadata.symbol != symbol_key:
                    continue
                if metadata.intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                    continue
                if not metadata.protection_confirmed or metadata.protected_quantity < metadata.cumulative_filled_quantity:
                    return "entry_protection_incomplete"
        return None

    def get_quarantined_broker_exposures(
        self, symbol: str | None = None
    ) -> dict[str, dict[str, Any]] | list[dict[str, Any]]:
        wanted = normalize_symbol(symbol) if symbol else None
        with self._lock:
            if wanted is None:
                return {
                    key: dict(value)
                    for key, value in self._quarantined_broker_exposures.items()
                }
            exposure = self._quarantined_broker_exposures.get(wanted)
            return [dict(exposure)] if exposure is not None else []

    def clear_quarantined_broker_exposure(self, symbol: str) -> bool:
        wanted = normalize_symbol(symbol)
        with self._lock:
            removed = self._quarantined_broker_exposures.pop(wanted, None) is not None
            if removed:
                self._cost_basis_unresolved_symbols.discard(wanted)
        if removed:
            self.save_state()
        return removed

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
        order_id = str(order_id).strip()
        if order_id in self._terminal_orders and self._terminal_orders[order_id].lifecycle_applied:
            return
        if order_id in self._orders:
            return
        symbol_key = symbol.upper()
        existing_position = self._positions.get(symbol_key)
        normalized_side = _normalize_order_side(side)
        normalized_intent = _normalize_intent(intent)
        if normalized_intent == "UNKNOWN" and existing_position is not None:
            exit_side = "SELL" if existing_position.side == "LONG" else "BUY"
            normalized_intent = "EXIT" if normalized_side == exit_side else "SCALE_IN"
        order = Order(
            order_id=order_id,
            symbol=symbol_key,
            side=normalized_side,
            order_type=order_type,
            quantity=int(qty),
            price=float(price),
            status="PENDING",
            linked_position_symbol=symbol_key if existing_position is not None else None,
            intent=normalized_intent,
            bracket_id=bracket_id,
            signal_id=signal_id,
            signal_fingerprint=signal_fingerprint,
            pre_order_position_side=existing_position.side if existing_position else None,
            pre_order_quantity=existing_position.quantity if existing_position else 0,
            trade_lifecycle_id=bracket_id or signal_id or order_id,
            linked_entry_order_id=(
                existing_position.order_id
                if existing_position is not None and normalized_intent in ("EXIT", "REDUCE")
                else None
            ),
            pre_order_entry_price=existing_position.entry_price if existing_position is not None else None,
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

    def remove_pending_order(self, order_id: str) -> None:
        order_key = str(order_id or "").strip()
        if not order_key:
            return
        with self._lock:
            order = self._orders.get(order_key)
            if order is not None and order.status not in self.FINAL_STATUSES:
                self._orders.pop(order_key, None)
            self._exit_lifecycles.pop(order_key, None)
        self.save_state()

    def is_exit_converging(self, symbol: str) -> bool:
        symbol_key = symbol.upper()
        with self._lock:
            for order in self._orders.values():
                if order.symbol == symbol_key and order.intent in ("EXIT", "REDUCE") and order.status not in self.FINAL_STATUSES:
                    return True
            for metadata in self._unresolved_terminal_orders.values():
                if metadata.symbol == symbol_key and metadata.intent in ("EXIT", "REDUCE"):
                    return True
            return symbol_key in self._recently_flat_exit_until_monotonic

    def bind_pending_order_id(self, provisional_order_id: str, final_order_id: str) -> None:
        provisional_key = str(provisional_order_id or "").strip()
        final_key = str(final_order_id or "").strip()
        if not provisional_key or not final_key or provisional_key == final_key:
            return
        with self._lock:
            order = self._orders.pop(provisional_key, None)
            if order is not None:
                existing = self._orders.get(final_key)
                if existing is None:
                    order.order_id = final_key
                    self._orders[final_key] = order
                else:
                    order = existing
                self._persist_order_state(order)
            metadata = self._terminal_orders.pop(provisional_key, None)
            if metadata is not None:
                self._terminal_orders[final_key] = metadata
                unresolved = self._unresolved_terminal_orders.pop(provisional_key, None)
                if unresolved is not None:
                    self._unresolved_terminal_orders[final_key] = unresolved
            lifecycle = self._exit_lifecycles.pop(provisional_key, None)
            if lifecycle is not None:
                lifecycle.exit_order_id = final_key
                self._exit_lifecycles[final_key] = lifecycle
        self.save_state()

    def update_order_status(
        self, order_id: str, status: str, fill_price: float | None = None
    ) -> None:
        order_id = str(order_id).strip()
        if not hasattr(self, "_terminal_orders"):
            self._terminal_orders = {}
            self._max_terminal_orders = 5000
        terminal_record = self._terminal_orders.get(order_id)
        if terminal_record is not None and terminal_record.lifecycle_resolved:
            return
        incoming_status = normalize_broker_order_status(status)
        if terminal_record is not None and terminal_record.normalized_status in self.FINAL_STATUSES and incoming_status not in self.FINAL_STATUSES:
            return
        order = self._orders.get(order_id)
        if order is None:
            return
        try:
            order.status = incoming_status or _normalize_status(str(status))
        except ValueError:
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
            metadata = TerminalOrderMetadata(
                terminal_at=order.terminal_at,
                normalized_status=order.status,
                cumulative_filled_quantity=order.filled_quantity,
                average_fill_price=order.fill_price,
                lifecycle_applied=fill_result.fill_recorded,
                accounting_finalized=fill_result.accounting_finalized,
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
                protected_quantity=order.protected_quantity,
                protection_confirmed=order.protection_confirmed,
                protection_confirmed_at=order.protection_confirmed_at,
                protection_failure_reason=order.protection_failure_reason,
            )
            self._terminal_orders[order_id] = metadata
            if not metadata.lifecycle_resolved:
                self._unresolved_terminal_orders[order_id] = metadata
            else:
                self._unresolved_terminal_orders.pop(order_id, None)
            self._evict_old_terminal_orders()
        self._persist_order_state(order)
        if order.status in self.FINAL_STATUSES and self._terminal_orders.get(order.order_id) is not None and self._terminal_orders[order.order_id].lifecycle_resolved:
            del self._orders[order.order_id]
        self.save_state()

    def apply_broker_order_update(
        self, order_id: str, broker_payload: Mapping[str, Any]
    ) -> None:
        order_key = str(order_id)
        with self._order_lock_for(order_key):
            order = self._orders.get(order_key)
            symbol_lock = self._symbol_lifecycle_lock_for(order.symbol) if order is not None else self._lock
            with symbol_lock:
                status = broker_payload.get("status")
                fill_price_raw = broker_payload.get("average_price") or broker_payload.get("fill_price") or broker_payload.get("price")
                filled_qty = broker_payload.get("filled_quantity") or broker_payload.get("filled")
                if order is not None and filled_qty is not None:
                    with suppress(Exception):
                        order.filled_quantity = int(float(filled_qty))
                fill_price = None
                if fill_price_raw is not None:
                    with suppress(Exception):
                        fill_price = float(fill_price_raw)
                self.update_order_status(order_key, str(status or ""), fill_price)

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
        symbol_key = symbol.upper() if symbol else None
        orders: Iterable[Order] = self._orders.values()
        if symbol_key is not None:
            orders = (order for order in orders if order.symbol == symbol_key)
        return [order for order in orders if order.status not in self.FINAL_STATUSES]

    def unresolved_terminal_summary(self) -> dict[str, object]:
        now = _now()
        unresolved = list(self._unresolved_terminal_orders.values())
        oldest_age_s = None
        if unresolved:
            oldest_age_s = max((now - min(item.terminal_at for item in unresolved)).total_seconds(), 0.0)
        return {"count": len(unresolved), "oldest_age_s": oldest_age_s}

    def confirm_entry_protection(
        self, order_id: str, bracket_id: str, protected_quantity: int
    ) -> None:
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
            if order.applied_filled_quantity <= 0:
                raise ValueError("Entry fill must be applied before protection")
            if protected_qty < order.applied_filled_quantity:
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
                    self._unresolved_terminal_orders.pop(order_key, None)
            self._persist_order_state(order)
        self.save_state()

    def save_state(self) -> None:
        """Persist one coherent positions/orders/ledger/quarantine snapshot to disk."""
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
                "broker_order_ledger": {
                    str(order_id): dict(row)
                    for order_id, row in self._broker_order_ledger.items()
                    if isinstance(row, Mapping)
                },
                "quarantined_broker_exposures": {
                    normalize_symbol(symbol): dict(exposure)
                    for symbol, exposure in self._quarantined_broker_exposures.items()
                    if isinstance(exposure, Mapping) and normalize_symbol(symbol)
                },
                "daily_realized_pnl": self._daily_realized_pnl,
                "local_realized_pnl": self._local_realized_pnl,
                "broker_realized_pnl": self._broker_realized_pnl,
                "local_provisional_realized_pnl": self._local_provisional_realized_pnl,
                "authoritative_realized_pnl": self._authoritative_realized_pnl,
                "pnl_authority": self._pnl_authority,
                "pnl_reconciliation_status": self._pnl_reconciliation_status,
                "pnl_snapshot_at": self._pnl_snapshot_at.isoformat() if self._pnl_snapshot_at else None,
                "session_opening_realized_baseline": self._session_opening_realized_baseline,
                "pnl_trading_date": self._pnl_trading_date,
                "pnl_account_fingerprint": self._pnl_account_fingerprint,
                "pnl_product_scope": self._pnl_product_scope,
                "baseline_established_at": self._baseline_established_at.isoformat() if self._baseline_established_at else None,
                "baseline_source": self._baseline_source,
                "require_pnl_baseline_for_entries": self._require_pnl_baseline_for_entries,
                "active_contracts": [contract.to_dict() for contract in self._active_contracts.values()],
            }
            reconciled_snapshot = copy.deepcopy(self._positions)
        try:
            _atomic_write_json(self._state_path, state)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to save position state: %s", exc)
            return
        self._persist_positions_snapshot()
        with self._lock:
            self._last_reconciled_state = reconciled_snapshot
        self._maybe_flush_persistent_state()

    def load_state(self) -> None:
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
            payload = json.loads(path_to_read.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._logger.error("Failed to load position state: %s", exc)
            return
        if not isinstance(payload, dict):
            self._logger.error("Invalid state payload (expected object)")
            return

        ledger_raw = payload.get("broker_order_ledger", {})
        restored_ledger: dict[str, dict[str, Any]] = {}
        if isinstance(ledger_raw, Mapping):
            for order_id, row in ledger_raw.items():
                if not isinstance(row, Mapping):
                    continue
                cloned = dict(row)
                symbol = normalize_symbol(
                    str(cloned.get("symbol") or cloned.get("tradingsymbol") or "")
                )
                if symbol:
                    cloned["symbol"] = symbol
                    cloned["tradingsymbol"] = symbol
                restored_ledger[str(order_id)] = cloned
        exposures_raw = payload.get("quarantined_broker_exposures", {})
        restored_exposures: dict[str, dict[str, Any]] = {}
        if isinstance(exposures_raw, Mapping):
            for raw_symbol, exposure in exposures_raw.items():
                if not isinstance(exposure, Mapping):
                    continue
                cloned = dict(exposure)
                symbol = normalize_symbol(
                    str(cloned.get("symbol") or cloned.get("tradingsymbol") or raw_symbol)
                )
                if not symbol:
                    continue
                cloned["symbol"] = symbol
                cloned["tradingsymbol"] = symbol
                restored_exposures[symbol] = cloned
        self._broker_order_ledger = restored_ledger
        self._quarantined_broker_exposures = restored_exposures
        self._cost_basis_unresolved_symbols = {
            symbol
            for symbol, exposure in restored_exposures.items()
            if str(exposure.get("reason") or "") == "cost_basis_unresolved"
        }

        positions: Dict[str, Position] = {}
        for item in payload.get("positions", []):
            try:
                position = Position.from_dict(cast(Mapping[str, Any], item))
            except (ValueError, TypeError):
                continue
            positions[position.symbol.upper()] = position
        orders: Dict[str, Order] = {}
        for item in payload.get("orders", []):
            try:
                order = Order.from_dict(cast(Mapping[str, Any], item))
            except (ValueError, TypeError):
                continue
            orders[order.order_id] = order
        self._positions = positions
        self._orders = orders

        terminal_raw = payload.get("terminal_orders", {})
        restored_terminal: dict[str, TerminalOrderMetadata] = {}
        if isinstance(terminal_raw, Mapping):
            for order_id, metadata in terminal_raw.items():
                if isinstance(metadata, Mapping):
                    with suppress(KeyError, TypeError, ValueError):
                        restored_terminal[str(order_id)] = TerminalOrderMetadata.from_dict(metadata)
        self._terminal_orders = restored_terminal
        unresolved_raw = payload.get("unresolved_terminal_orders", {})
        restored_unresolved: dict[str, TerminalOrderMetadata] = {}
        if isinstance(unresolved_raw, Mapping):
            for order_id, metadata in unresolved_raw.items():
                if isinstance(metadata, Mapping):
                    with suppress(KeyError, TypeError, ValueError):
                        restored_unresolved[str(order_id)] = TerminalOrderMetadata.from_dict(metadata)
        self._unresolved_terminal_orders = restored_unresolved
        exit_raw = payload.get("exit_lifecycles", {})
        restored_exit: dict[str, ExitLifecycleRecord] = {}
        if isinstance(exit_raw, Mapping):
            for order_id, lifecycle in exit_raw.items():
                if isinstance(lifecycle, Mapping):
                    with suppress(KeyError, TypeError, ValueError):
                        restored_exit[str(order_id)] = ExitLifecycleRecord.from_dict(lifecycle)
        self._exit_lifecycles = restored_exit

        contracts: Dict[str, ActiveContract] = {}
        index: Dict[str, str] = {}
        for item in payload.get("active_contracts", []):
            with suppress(ValueError, TypeError):
                contract = ActiveContract.from_dict(cast(Mapping[str, Any], item))
                contracts[contract.underlying] = contract
                index[contract.symbol] = contract.underlying
        self._active_contracts = contracts
        self._contract_index = index
        legacy_daily = float(payload.get("daily_realized_pnl", 0.0))
        self._local_realized_pnl = float(payload.get("local_realized_pnl", legacy_daily))
        broker_realized = payload.get("broker_realized_pnl")
        self._broker_realized_pnl = None if broker_realized is None else float(broker_realized)
        self._local_provisional_realized_pnl = float(payload.get("local_provisional_realized_pnl", 0.0))
        self._authoritative_realized_pnl = float(payload.get("authoritative_realized_pnl", self._local_realized_pnl))
        self._pnl_authority = str(payload.get("pnl_authority", "unresolved"))
        self._pnl_reconciliation_status = str(payload.get("pnl_reconciliation_status", "unresolved"))
        pnl_snapshot_at = payload.get("pnl_snapshot_at")
        if isinstance(pnl_snapshot_at, str) and pnl_snapshot_at:
            with suppress(ValueError):
                self._pnl_snapshot_at = datetime.fromisoformat(pnl_snapshot_at)
        baseline = payload.get("session_opening_realized_baseline")
        self._session_opening_realized_baseline = None if baseline is None else float(baseline)
        self._pnl_trading_date = str(payload["pnl_trading_date"]) if payload.get("pnl_trading_date") is not None else None
        self._pnl_account_fingerprint = str(payload["pnl_account_fingerprint"]) if payload.get("pnl_account_fingerprint") is not None else None
        self._pnl_product_scope = str(payload.get("pnl_product_scope", "MIS"))
        baseline_established_at = payload.get("baseline_established_at")
        if isinstance(baseline_established_at, str) and baseline_established_at:
            with suppress(ValueError):
                self._baseline_established_at = datetime.fromisoformat(baseline_established_at)
        self._baseline_source = str(payload["baseline_source"]) if payload.get("baseline_source") is not None else None
        self._require_pnl_baseline_for_entries = bool(payload.get("require_pnl_baseline_for_entries", False))
        with self._lock:
            self._refresh_realized_pnl_locked()
        self._last_reconciled_state = copy.deepcopy(self._positions)

    def _restore_from_persistent_manager(self, manager: "PersistentStateManager") -> None:
        try:
            payloads = manager.load_positions()
        except Exception:
            return
        self.restore_positions(payloads)
        try:
            orders_payloads = manager.load_open_orders()
        except Exception:
            orders_payloads = []
        rebuilt_orders: Dict[str, Order] = {}
        for item in orders_payloads:
            if isinstance(item, Mapping):
                with suppress(KeyError, TypeError, ValueError):
                    order = Order.from_dict(item)
                    rebuilt_orders[order.order_id] = order
        self._orders = rebuilt_orders

    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        self._persistent_state = manager

    def restore_positions(self, payloads: Iterable[Mapping[str, Any]]) -> None:
        items = list(payloads)
        rebuilt: Dict[str, Position] = {}
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"persisted position row {index} is not a mapping")
            try:
                position = Position.from_dict(item)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid persisted position row {index}") from exc
            rebuilt[position.symbol.upper()] = position
        with self._lock:
            self._positions = rebuilt
        self.save_state()

    @staticmethod
    def _safe_get_net_qty(record: Mapping[str, object]) -> int:
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

    def synchronize_with_broker(self, broker_positions: Any) -> None:
        """Canonicalize broker truth and refresh only cost-basis quarantine rows."""
        lifecycle_snapshot = _snapshot_owned_position_lifecycle(self)
        prepared, unresolved = _prepare_broker_positions(self, broker_positions)
        with self._lock:
            had_managed_positions = bool(self._positions)
            previous_quarantine = {
                key: dict(value)
                for key, value in self._quarantined_broker_exposures.items()
            }
            merged_quarantine = _merge_cost_basis_quarantine(
                previous_quarantine, prepared, set(unresolved)
            )
            quarantine_changed = merged_quarantine != previous_quarantine
            self._quarantined_broker_exposures = merged_quarantine
            self._cost_basis_unresolved_symbols = set(unresolved)
        if unresolved and isinstance(prepared, list):
            prepared = [row for row in prepared if _prepared_row_symbol(row) not in unresolved]
        self._synchronize_managed_positions_from_broker(prepared)
        _canonicalize_position_store(self)
        restored = _restore_owned_position_lifecycle(self, lifecycle_snapshot)
        with self._lock:
            no_managed_positions = not self._positions
        if restored or (quarantine_changed and not had_managed_positions and no_managed_positions):
            self.save_state()

    def _synchronize_managed_positions_from_broker(self, broker_positions: Any) -> None:
        try:
            snapshot = decode_position_snapshot(broker_positions)
        except Exception as exc:
            with self._lock:
                self._last_broker_position_snapshot_failure_at = time.time()
                self._last_broker_position_snapshot_failure_reason = str(exc)
            raise

        def get_float(record: Mapping[str, object], keys: Sequence[str], *, default: float = 0.0) -> float:
            for key in keys:
                if key not in record or record.get(key) is None:
                    continue
                value = float(cast(Any, record.get(key)))
                if not math.isfinite(value):
                    raise ValueError(f"invalid broker numeric field {key}={record.get(key)!r}")
                return value
            return float(default)

        baseline_initialized = False
        with self._lock:
            existing_positions = copy.deepcopy(self._positions)
            reconciled: Dict[str, Position] = {}
            snapshot_realized_pnl = 0.0
            snapshot_realized_seen = False
            snapshot_symbols = {row.symbol for row in snapshot.rows}
            for recent_symbol in list(self._recently_flat_exit_until_monotonic):
                if recent_symbol not in snapshot_symbols:
                    self._clear_recent_exit_guard_locked(recent_symbol)
            for row in snapshot.rows:
                record = row.raw
                symbol = row.symbol
                if not is_strategy_instrument(symbol):
                    continue
                product = str(record.get("product") or "").strip().upper()
                if product != "MIS":
                    if symbol in existing_positions:
                        raise ValueError(f"managed broker position {symbol} has unexpected product {product or 'missing'}")
                    continue
                quantity = row.quantity
                realized_pnl = get_float(record, ("realised", "realized"), default=0.0)
                if "realised" in record or "realized" in record:
                    snapshot_realized_seen = True
                    snapshot_realized_pnl += realized_pnl
                if quantity == 0:
                    self._clear_recent_exit_guard_locked(symbol)
                    continue
                if self._should_ignore_recent_exit_stale_snapshot_locked(symbol, quantity):
                    continue
                side: Side = "LONG" if quantity > 0 else "SHORT"
                abs_quantity = abs(quantity)
                entry_price = get_float(record, ("average_price", "avg_price", "price", "buy_price"))
                current_price = get_float(record, ("last_price", "ltp", "close", "sell_price"), default=entry_price)
                if entry_price <= 0.0 and current_price > 0.0:
                    entry_price = current_price
                if current_price <= 0.0 and entry_price > 0.0:
                    current_price = entry_price
                if entry_price <= 0.0 or current_price <= 0.0:
                    raise ValueError(f"broker position {symbol} has no valid price")
                existing = existing_positions.get(symbol)
                position = (
                    self._create_position(
                        symbol=symbol,
                        quantity=abs_quantity,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        realized_pnl=realized_pnl,
                        source="broker_sync",
                    )
                    if existing is None
                    else self._update_position(
                        position=existing,
                        quantity=abs_quantity,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        realized_pnl=realized_pnl,
                        source="broker_sync",
                    )
                )
                reconciled[symbol] = position
            old_keys = set(self._positions)
            new_keys = set(reconciled)
            removed_symbols = sorted(old_keys - new_keys)
            added_symbols = sorted(new_keys - old_keys)
            if set(self._positions) != set(reconciled) or any(
                int(getattr(self._positions.get(symbol), "quantity", 0) or 0)
                != int(getattr(position, "quantity", 0) or 0)
                for symbol, position in reconciled.items()
            ):
                self._mark_local_position_mutation_locked()
            self._positions = reconciled
            self._last_broker_quantities_by_symbol = {row.symbol: int(row.quantity) for row in snapshot.rows}
            self._last_broker_position_snapshot_at = snapshot.fetched_at
            self._last_broker_position_snapshot_mono = time.monotonic()
            self._last_broker_position_snapshot_valid = True
            self._broker_snapshot_local_generation = self._local_position_generation
            self._last_broker_position_snapshot_source = snapshot.source
            self._last_broker_position_snapshot_failure_reason = None
            session_date = self._trading_date_ist()
            baseline_missing_or_stale = self._session_opening_realized_baseline is None or self._pnl_trading_date != session_date
            empty_snapshot_can_seed_zero = not snapshot.rows and self._local_realized_pnl == 0.0
            if baseline_missing_or_stale and (snapshot_realized_seen or empty_snapshot_can_seed_zero):
                self._session_opening_realized_baseline = float(
                    snapshot_realized_pnl - self._local_realized_pnl if snapshot_realized_seen else 0.0
                )
                self._pnl_trading_date = session_date
                self._pnl_product_scope = "MIS"
                self._baseline_established_at = _now()
                self._baseline_source = "validated_broker_positions" if snapshot_realized_seen else "validated_broker_empty_snapshot"
                baseline_initialized = True
            if snapshot_realized_seen:
                self._broker_realized_pnl = float(snapshot_realized_pnl)
            elif baseline_initialized:
                self._broker_realized_pnl = 0.0
            if snapshot_realized_seen or baseline_initialized:
                self._refresh_realized_pnl_locked()
        if removed_symbols:
            hook = getattr(self, "_on_symbols_flat_hook", None)
            if hook is not None:
                with suppress(Exception):
                    hook(list(removed_symbols))
        if not old_keys and not new_keys and not snapshot_realized_seen and not baseline_initialized:
            return
        self.save_state()
        self._logger.info(
            "POSITION_SYNC_COMMITTED total=%s added=%s removed=%s realized_authoritative=%s",
            len(reconciled),
            len(added_symbols),
            len(removed_symbols),
            snapshot_realized_seen,
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
        return Position(
            symbol=str(symbol).strip().upper(),
            side=_normalize_side(str(side)),
            quantity=int(max(quantity, 0)),
            entry_price=float(entry_price if entry_price > 0.0 else current_price),
            entry_time=_now(),
            current_price=float(current_price if current_price > 0.0 else entry_price),
            realized_pnl=float(realized_pnl),
        )

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
        position.side = _normalize_side(str(side))
        position.quantity = int(max(quantity, 0))
        if entry_price > 0.0:
            position.entry_price = float(entry_price)
        if current_price > 0.0:
            position.current_price = float(current_price)
        position.realized_pnl = float(realized_pnl)
        return position

    def _persist_positions_snapshot(self) -> None:
        manager = self._persistent_state
        if manager is None:
            return
        with self._lock:
            position_snapshot = [position.to_dict() for position in self._positions.values()]
            order_snapshot = [order.to_dict() for order in self._orders.values()]
        try:
            stored = manager.load_positions()
        except Exception:
            stored = []
        current_symbols = {str(entry.get("symbol", "")).strip().upper() for entry in position_snapshot}
        stored_symbols = {str(item.get("symbol", "")).strip().upper() for item in stored if isinstance(item, Mapping)}
        for entry in position_snapshot:
            with suppress(Exception):
                manager.save_position(entry)
        for symbol in stored_symbols - current_symbols:
            with suppress(Exception):
                manager.save_position({"symbol": symbol, "quantity": 0})
        for payload in order_snapshot:
            with suppress(Exception):
                manager.save_order(payload)
        with suppress(Exception):
            manager.flush()

    def _maybe_flush_persistent_state(self) -> None:
        manager = self._persistent_state
        if manager is None:
            return
        now = time.monotonic()
        if now - self._last_persistence_check < self._persistence_flush_interval_s:
            return
        self._last_persistence_check = now
        try:
            telemetry = manager.telemetry()
        except Exception:
            return
        pending_source = telemetry.get("pending_events")
        if not isinstance(pending_source, (int, float)):
            pending_source = telemetry.get("pending_queue_depth")
        pending_events = int(pending_source) if isinstance(pending_source, (int, float)) else 0
        if pending_events >= self._persistence_pending_threshold:
            with suppress(Exception):
                manager.flush()

    def _evict_old_terminal_orders(self) -> None:
        overflow = len(self._terminal_orders) - self._max_terminal_orders
        if overflow <= 0:
            return
        ordered = sorted(
            (
                (order_id, metadata)
                for order_id, metadata in self._terminal_orders.items()
                if metadata.lifecycle_resolved and order_id not in self._unresolved_terminal_orders
            ),
            key=lambda item: (item[1].terminal_at, item[0]),
        )
        for order_id, _ in ordered[:overflow]:
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
        manager = self._persistent_state
        if manager is None:
            return
        timestamp = order.timestamp if isinstance(order.timestamp, datetime) else _now()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        payload = {
            "fill_id": f"{order.order_id}:{order.applied_filled_quantity + int(quantity)}",
            "order_id": order.order_id,
            "intent": order.intent,
            "bracket_id": order.bracket_id,
            "signal_id": order.signal_id,
            "signal_fingerprint": order.signal_fingerprint,
            "symbol": order.symbol,
            "side": order.side,
            "quantity_delta": int(quantity),
            "cumulative_filled_quantity": int(order.applied_filled_quantity + int(quantity)),
            "fill_price": float(fill_price),
            "status": order.status,
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "lifecycle_applied": bool(lifecycle_applied),
            "position_applied": bool(position_applied),
            "pnl_applied": bool(pnl_applied),
            "accounting_finalized": bool(accounting_finalized),
            "lifecycle_resolved": bool(lifecycle_resolved),
        }
        with suppress(Exception):
            manager.save_fill(payload)
            manager.flush()

    def _persist_order_state(self, order: Order) -> None:
        manager = self._persistent_state
        if manager is not None:
            with suppress(Exception):
                manager.save_order(order.to_dict())

    def _persist_order_snapshots(self, manager: "PersistentStateManager") -> None:
        for payload in [order.to_dict() for order in self._orders.values()]:
            with suppress(Exception):
                manager.save_order(payload)

    def reset_daily_pnl(self) -> None:
        with self._lock:
            self._local_realized_pnl = 0.0
            self._local_provisional_realized_pnl = 0.0
            self._pnl_trading_date = self._trading_date_ist()
            self._session_opening_realized_baseline = self._broker_realized_pnl
            for position in self._positions.values():
                position.realized_pnl = 0.0
            self._refresh_realized_pnl_locked()
        self.save_state()

    def _handle_filled_order(self, order: Order) -> FillApplicationResult:
        symbol_key = order.symbol
        cumulative_qty = order.quantity if order.filled_quantity == 0 else order.filled_quantity
        previous_qty = int(order.applied_filled_quantity or 0)
        qty = int(cumulative_qty) - previous_qty
        if qty <= 0:
            return FillApplicationResult(reason="non_incremental_cumulative_quantity")
        cumulative_avg = order.fill_price
        if cumulative_avg is None or float(cumulative_avg) <= 0:
            return FillApplicationResult(reason="invalid_cumulative_average_price")
        fill_price = float(cumulative_avg)
        intent = _normalize_intent(order.intent)
        is_terminal = order.status == "FILLED"

        def mark_applied() -> None:
            order.applied_filled_quantity += qty
            order.applied_cumulative_notional += fill_price * qty
            order.last_cumulative_average_price = float(cumulative_avg)

        if not self.has_position(symbol_key):
            if intent not in ("ENTRY", "SCALE_IN", "REVERSAL"):
                return FillApplicationResult(reason="ambiguous_fill_quarantined")
            self._persist_fill(
                order,
                qty,
                fill_price,
                lifecycle_applied=True,
                position_applied=True,
                accounting_finalized=False,
            )
            self.open_position(
                symbol=symbol_key,
                side="LONG" if order.side == "BUY" else "SHORT",
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
                quantity_delta=qty,
                delta_fill_price=fill_price,
                reason="entry_filled_unprotected",
            )

        position = self._positions[symbol_key]
        entry_side_matches = (position.side == "LONG" and order.side == "BUY") or (
            position.side == "SHORT" and order.side == "SELL"
        )
        if intent in ("ENTRY", "SCALE_IN", "REVERSAL") and entry_side_matches:
            expected_post_fill_qty = int(order.pre_order_quantity or 0) + int(cumulative_qty)
            if position.quantity >= expected_post_fill_qty:
                if order.pre_order_quantity == 0 and position.order_id is None:
                    position.order_id = order.order_id
                mark_applied()
                return FillApplicationResult(
                    fill_recorded=True,
                    position_applied=True,
                    quantity_delta=qty,
                    delta_fill_price=fill_price,
                    reason="entry_fill_already_reflected_by_broker_sync",
                )
        if (position.side == "LONG" and order.side == "SELL") or (
            position.side == "SHORT" and order.side == "BUY"
        ):
            self._reduce_or_close_position(position, qty, fill_price)
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
        self._scale_position(position, qty, fill_price)
        mark_applied()
        return FillApplicationResult(
            fill_recorded=True,
            position_applied=True,
            quantity_delta=qty,
            delta_fill_price=fill_price,
            reason="scale_fill_unprotected",
        )

    def _scale_position(self, position: Position, qty: int, fill_price: float) -> None:
        new_qty = position.quantity + qty
        if new_qty <= 0:
            return
        position.entry_price = (
            position.entry_price * position.quantity + fill_price * qty
        ) / new_qty
        position.quantity = new_qty
        position.current_price = fill_price
        with self._lock:
            self._mark_local_position_mutation_locked()

    def _reduce_or_close_position(self, position: Position, qty: int, fill_price: float) -> None:
        reduce_qty = min(qty, position.quantity)
        realized = self._calculate_realized_pnl(
            position.side, position.entry_price, fill_price, reduce_qty
        )
        position.quantity -= reduce_qty
        position.realized_pnl += realized
        self._local_realized_pnl += realized
        with self._lock:
            self._refresh_realized_pnl_locked()
            self._mark_local_position_mutation_locked()
        position.current_price = fill_price
        if position.quantity == 0:
            del self._positions[position.symbol]
            self._mark_recent_exit_flat_locked(position.symbol)
            self.clear_active_contract_by_symbol(position.symbol)

    def _mark_recent_exit_flat_locked(self, symbol: str) -> None:
        key = symbol.upper()
        now = time.monotonic()
        until = now + self._recently_flat_exit_grace_seconds
        self._recently_flat_exit_until_monotonic[key] = until
        self._recently_flat_exit_metadata[key] = ExitSettlementGuard(
            completed_exit_at_monotonic=now, grace_until_monotonic=until
        )

    def _clear_recent_exit_guard_locked(self, symbol: str) -> None:
        key = symbol.upper()
        self._recently_flat_exit_until_monotonic.pop(key, None)
        self._recently_flat_exit_metadata.pop(key, None)

    def _should_ignore_recent_exit_stale_snapshot_locked(
        self, symbol: str, quantity: int
    ) -> bool:
        if symbol.upper() in self._positions:
            return False
        key = symbol.upper()
        until = self._recently_flat_exit_until_monotonic.get(key)
        if until is None:
            return False
        if time.monotonic() > float(until):
            self._clear_recent_exit_guard_locked(key)
            return False
        return True

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
