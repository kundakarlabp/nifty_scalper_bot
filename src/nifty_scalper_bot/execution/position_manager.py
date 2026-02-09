"""Position and order state tracking for the scalper bot."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
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

from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.reasons import canonical

if TYPE_CHECKING:
    from nifty_scalper_bot.data.persistent_state import PersistentStateManager

Side = Literal["LONG", "SHORT"]
OrderSide = Literal["BUY", "SELL"]
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
            status=_normalize_status(str(payload["status"])),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            filled_quantity=_to_int(payload.get("filled_quantity", 0)),
            fill_price=_to_optional_float(payload.get("fill_price")),
            linked_position_symbol=(
                str(payload["linked_position_symbol"])
                if payload.get("linked_position_symbol") is not None
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
        self._orders: Dict[str, Order] = {}
        self._processed_order_ids: set[str] = set()
        self._max_processed_ids = 1000  # Limit memory usage
        self._daily_realized_pnl: float = 0.0
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
        """Record reconciliation failure, revert state, and trigger notifications.

        Args:
            reason: Categorical reason for the failure.
            error: Optional exception raised during reconciliation.
            payload_count: Number of broker payloads processed.
            previous_positions: Snapshot used to restore known-good state.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _handle_reconcile_failure",
            extra={"event": "position_reconcile_failure", "reason": reason},
        )
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
        except Exception as metrics_exc:  # noqa: BLE001 - defensive metric guard
            self._logger.error(
                "Failure in reconcile failure metrics: %s",
                metrics_exc,
                extra={"event": "position_reconcile_metric_failure"},
            )
        fallback = (
            previous_positions
            if previous_positions is not None
            else self._last_reconciled_state
        )
        fallback_source = (
            "previous_attempt"
            if previous_positions is not None
            else ("last_reconciled" if self._last_reconciled_state else "unavailable")
        )
        restore_applied = False
        if fallback is not None:
            try:
                self._positions = copy.deepcopy(dict(fallback))
                restore_applied = True
            except Exception as exc:  # noqa: BLE001 - defensive revert guard
                self._logger.error(
                    "Failure in _handle_reconcile_failure revert: %s",
                    exc,
                    extra={"event": "position_reconcile_revert_failed"},
                )
        if restore_applied:
            self._logger.info(
                "Condition met: position_reconcile_state_restored",
                extra={
                    "event": "position_reconcile_state_restored",
                    "source": fallback_source,
                    "count": len(self._positions),
                },
            )
            try:
                self.save_state()
            except Exception as exc:  # noqa: BLE001 - defensive persistence guard
                self._logger.error(
                    "Failure in _handle_reconcile_failure persist: %s",
                    exc,
                    extra={"event": "position_reconcile_persist_failed"},
                )
        retry_delay = self._compute_retry_delay()
        event_payload: dict[str, object] = {
            "reason": reason_token,
            "failures": self._consecutive_reconcile_failures,
            "retry_sec": retry_delay,
            "count": payload_count,
            "timestamp": self._last_reconcile_attempt.isoformat(),
        }
        event_payload["restored"] = restore_applied
        event_payload["source"] = fallback_source
        if error is not None:
            event_payload["error"] = str(error)
        try:
            _POSITION_RECONCILE_EVENTS.labels("failed").inc()
        except Exception:  # noqa: BLE001 - optional metrics backend
            pass
        self._notify_reconcile_event("position_reconcile_failed", event_payload)
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
        """Synchronize local positions with the broker immediately.

        Args:
            None.

        Returns:
            ``True`` when reconciliation succeeded, otherwise ``False``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered reconcile_now",
            extra={"event": "position_reconcile_enter"},
        )
        previous_positions: Dict[str, Position] | None = None
        payloads: list[Mapping[str, object]] = []
        try:
            try:
                previous_positions = copy.deepcopy(self._positions)
            except Exception as exc:  # noqa: BLE001 - defensive snapshot guard
                self._logger.error(
                    "Failure in reconcile_now snapshot: %s",
                    exc,
                    extra={"event": "position_reconcile_snapshot_failed"},
                )
            fetcher = self._resolve_broker_position_fetcher()
            if fetcher is None:
                reason_token = canonical("fetcher_missing")
                self._logger.warning(
                    "Position reconciliation skipped (fetcher unavailable)",
                    extra={
                        "event": "position_reconcile_failed",
                        "reason": reason_token,
                    },
                )
                self._handle_reconcile_failure(
                    reason=reason_token,
                    error=None,
                    payload_count=0,
                    previous_positions=previous_positions,
                )
                return False
            try:
                response = fetcher()
            except Exception as exc:  # noqa: BLE001 - defensive broker call
                reason_token = canonical("fetch_error")
                self._logger.warning(
                    "Position reconciliation fetch failed: %s",
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
                    previous_positions=previous_positions,
                )
                return False
            if response is None:
                payloads = []
            elif isinstance(response, Mapping):
                payloads.append(cast(Mapping[str, object], response))
            elif isinstance(response, Iterable) and not isinstance(
                response, (str, bytes)
            ):
                for item in response:
                    if isinstance(item, Mapping):
                        payloads.append(cast(Mapping[str, object], item))
            else:
                reason_token = canonical("payload_type")
                self._logger.warning(
                    "Position reconciliation returned unsupported payload %s",
                    type(response),
                    extra={
                        "event": "position_reconcile_failed",
                        "reason": reason_token,
                    },
                )
                self._handle_reconcile_failure(
                    reason=reason_token,
                    error=None,
                    payload_count=0,
                    previous_positions=previous_positions,
                )
                return False
            try:
                self.synchronize_with_broker(payloads)
            except Exception as exc:  # noqa: BLE001 - defensive sync guard
                reason_token = canonical("apply_error")
                self._logger.warning(
                    "Position reconciliation apply failed: %s",
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
                    payload_count=len(payloads),
                    previous_positions=previous_positions,
                )
                return False
            reason_token = canonical("ok")
            self._logger.info(
                "Condition met: position_reconcile_ok",
                extra={
                    "event": "position_reconcile_ok",
                    "count": len(payloads),
                    "reason": reason_token,
                },
            )
            self._handle_reconcile_success(len(payloads))
            return True
        except Exception as exc:  # noqa: BLE001 - defensive outer guard
            reason_token = canonical("unexpected_error")
            self._logger.warning(
                "Unexpected error during reconcile_now: %s",
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
                payload_count=len(payloads),
                previous_positions=previous_positions,
            )
            return False

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
        """Open a new position, raising if one already exists for ``symbol``."""

        symbol_key = symbol.upper()
        if self.has_position(symbol_key):
            raise ValueError(f"Position already exists for {symbol_key}")

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
        """Close an existing position and compute realized PnL."""

        symbol_key = symbol.upper()
        position = self._positions.get(symbol_key)
        if position is None:
            raise ValueError(f"No open position for {symbol_key}")

        qty = position.quantity
        realized = self._calculate_realized_pnl(
            position.side, position.entry_price, float(exit_price), qty
        )
        position.realized_pnl += realized
        self._daily_realized_pnl += realized
        position.current_price = float(exit_price)
        position.quantity = 0
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
        del self._positions[symbol_key]
        self.clear_active_contract_by_symbol(symbol_key)
        self.save_state()
        return position

    def update_from_order(self, order) -> None:
        """
        CRITICAL FIX: Allows OrderManager to sync state without crashing.
        Used by _adopt_orphan_position AND _generate_adjustment_order.
        """
        if order.status == "FILLED":
            self.update_position(
                symbol=order.symbol,
                qty=order.filled_quantity,
                price=order.average_price,
                side=order.side,
                product="MIS"  # Defaulting to MIS for safety
            )

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update the mark price of an open position."""

        position = self._positions.get(symbol.upper())
        if position is None:
            raise ValueError(f"No open position for {symbol}")
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

    def get_realized_pnl(self) -> float:
        """Return the accumulated realized profit and loss for the day."""

        return self._daily_realized_pnl

    def add_pending_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_type: str,
    ) -> None:
        """Track a newly submitted order.
        
        ✅ PRODUCTION FIX: Skip orders that have already been fully processed.
        This prevents the infinite loop where historical filled orders are
        re-added and re-processed every reconciliation cycle.
        """
        order_id = str(order_id).strip()
        
        # ✅ FIX 1: Don't re-add orders that were already processed
        if hasattr(self, '_processed_order_ids') and order_id in self._processed_order_ids:
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
        order = Order(
            order_id=order_id,
            symbol=symbol_key,
            side=_normalize_order_side(side),
            order_type=order_type,
            quantity=int(qty),
            price=float(price),
            status="PENDING",
            linked_position_symbol=(
                symbol_key if self.has_position(symbol_key) else None
            ),
        )
        self._orders[order.order_id] = order
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
        if not hasattr(self, '_processed_order_ids'):
            self._processed_order_ids = set()
            self._max_processed_ids = 1000
        
        # ✅ FIX: Skip if this order was already fully processed
        if order_id in self._processed_order_ids:
            self._logger.debug(
                f"Skipping already-processed order: {order_id}",
                extra={"event": "order_already_processed", "order_id": order_id}
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
            order.status = _normalize_status(str(status))
        except ValueError:
            self._logger.warning(
                "Ignoring unsupported status '%s' for order %s", status, order_id
            )
            return

        if fill_price is not None:
            order.fill_price = float(fill_price)

        if order.status == "FILLED" and order.fill_price is not None:
            order.filled_quantity = order.quantity
            self._handle_filled_order(order)
            
            # ✅ FIX: Mark this order as processed AFTER handling the fill
            # This ensures we never process the same fill twice
            self._processed_order_ids.add(order_id)
            self._logger.debug(
                f"Marked order as processed: {order_id}",
                extra={"event": "order_marked_processed", "order_id": order_id}
            )
            
            # ✅ FIX: Prevent memory leak - trim old IDs if too many
            if len(self._processed_order_ids) > self._max_processed_ids:
                to_remove = list(self._processed_order_ids)[:self._max_processed_ids // 2]
                for old_id in to_remove:
                    self._processed_order_ids.discard(old_id)
                self._logger.debug(
                    f"Trimmed processed_order_ids: removed {len(to_remove)} old entries"
                )

        self._persist_order_state(order)

        if order.status in self.FINAL_STATUSES:
            del self._orders[order.order_id]

        self.save_state()

    def get_pending_orders(self, symbol: str | None = None) -> list[Order]:
        """Return tracked orders, optionally filtered by ``symbol``."""

        symbol_key = symbol.upper() if symbol else None
        orders: Iterable[Order] = self._orders.values()
        if symbol_key is not None:
            orders = (order for order in orders if order.symbol == symbol_key)
        return [order for order in orders if order.status not in self.FINAL_STATUSES]

    def save_state(self) -> None:
        """Persist positions and orders to disk."""

        state = {
            "positions": [position.to_dict() for position in self._positions.values()],
            "orders": [order.to_dict() for order in self._orders.values()],
            "daily_realized_pnl": self._daily_realized_pnl,
            "active_contracts": [
                contract.to_dict() for contract in self._active_contracts.values()
            ],
        }
        try:
            _atomic_write_json(self._state_path, state)
        except Exception as exc:  # noqa: BLE001 - handled upstream
            self._logger.error("Failed to save position state: %s", exc)
            return
        self._persist_positions_snapshot()
        try:
            self._last_reconciled_state = copy.deepcopy(self._positions)
        except Exception as exc:  # noqa: BLE001 - defensive snapshot guard
            self._logger.error(
                "Failure in save_state snapshot: %s",
                exc,
                extra={"event": "position_reconcile_snapshot_failed"},
            )
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
        self._daily_realized_pnl = float(payload.get("daily_realized_pnl", 0.0))
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
            payloads = []
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
        """Restore open positions from persisted *payloads*.

        Args:
            payloads: Iterable of serialised position dictionaries.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered restore_positions",
            extra={"event": "position_manager_restore"},
        )
        rebuilt: Dict[str, Position] = {}
        for item in payloads:
            try:
                position = Position.from_dict(cast(Mapping[str, Any], item))
            except (KeyError, TypeError, ValueError) as exc:
                self._logger.error("Failure in restore_positions: %s", exc)
                continue
            rebuilt[position.symbol.upper()] = position
        if not rebuilt:
            self._logger.info(
                "Condition met: restore_positions_empty",
                extra={"event": "position_manager_restore_empty"},
            )
            return
        self._positions = rebuilt
        self._logger.info(
            "Condition met: restore_positions_applied",
            extra={
                "event": "position_manager_restore_applied",
                "count": len(rebuilt),
            },
        )
        self.save_state()

    def _safe_get_net_qty(record: Mapping[str, object]) -> int:
        """
        🚨 CRITICAL FIX: Safely extract net quantity with explicit None checks.
    
        Python's 'or' chain evaluates 0 as falsy:
            0 or 65 = 65  ← WRONG!
    
        We need explicit None checks because 0 is a VALID quantity (position closed).
        """
        # Check net quantity fields FIRST with explicit None check
        for key in ("net_qty", "net_quantity", "netQuantity", "net"):
            val = record.get(key)
            if val is not None:  # Explicit None check - 0 is valid!
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    continue
    
        # Only fallback to 'quantity' if ALL net keys are genuinely missing
        # This is the dangerous fallback that caused the infinite loop
        qty_val = record.get("quantity")
        if qty_val is not None:
            try:
                return int(float(qty_val))
            except (ValueError, TypeError):
                pass
    
        return 0
    

    def synchronize_with_broker(
        self, broker_positions: Sequence[Mapping[str, object]]
    ) -> None:
        """
        Align local positions with broker-provided *broker_positions*.
        Optimized for robustness, atomic updates, and error handling.
        """
        self._logger.debug(
            "Entered synchronize_with_broker",
            extra={"event": "position_manager_sync"},
        )

        try:
            payloads = list(broker_positions or [])
        except Exception as exc:
            self._logger.error(
                "Failure in synchronize_with_broker input validation: %s",
                exc,
                extra={"event": "position_manager_sync_error"},
                exc_info=True,
            )
            return

        reconciled: Dict[str, Position] = {}

        # Helper to safely extract floats from multiple possible keys
        def _get_float(record: Mapping, keys: list[str], default: float = 0.0) -> float:
            for k in keys:
                if val := record.get(k):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return default

        for record in payloads:
            if not isinstance(record, Mapping):
                continue

            # 1. Symbol Normalization
            raw_symbol = (
                record.get("tradingsymbol")
                or record.get("symbol")
                or record.get("instrument")
                or ""
            )
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                continue

            # 2. Quantity Extraction (Fail-Safe)
            try:
                # Handle various broker keys for quantity
                qty_val = record.get("quantity") or record.get("net_quantity") or 0
                quantity = int(float(qty_val))
            except Exception as exc:
                self._logger.error(
                    "Failure decoding broker quantity for %s: %s",
                    symbol,
                    exc,
                    extra={"event": "position_manager_sync_qty_error"},
                    exc_info=True,
                )
                continue

            if quantity == 0:
                continue

            side: Side = "LONG" if quantity > 0 else "SHORT"
            abs_quantity = abs(quantity)

            # 3. Price & PnL Extraction
            entry_price = _get_float(record, ["average_price", "avg_price", "price", "buy_price"])
            current_price = _get_float(record, ["last_price", "ltp", "close", "sell_price"], default=entry_price)
            realized_pnl = _get_float(record, ["realized", "pnl", "m2m", "realised"])

            # Sanity check: Ensure prices are positive if possible
            if entry_price <= 0.0 and current_price > 0.0:
                entry_price = current_price
            if current_price <= 0.0 and entry_price > 0.0:
                current_price = entry_price

            # 4. Construct Position Object
            try:
                # Check if we have an existing position to preserve metadata
                existing = self._positions.get(symbol)

                if existing is None:
                    # Import NEW position
                    position = self._create_position(
                        symbol=symbol,
                        quantity=abs_quantity,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        realized_pnl=realized_pnl,
                        source="broker_sync",
                    )
                    self._logger.info(
                        f"Sync: Imported NEW {side} position: {symbol} x {abs_quantity}"
                    )
                else:
                    # Update EXISTING position
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

            except Exception as exc:
                self._logger.error(
                    "Failure applying broker snapshot for %s: %s",
                    symbol,
                    exc,
                    extra={"event": "position_manager_sync_apply_error"},
                    exc_info=True,
                )
                continue

        # 5. Atomic Reconciliation & Persistence
        with self._lock:
            old_keys = set(self._positions.keys())
            new_keys = set(reconciled.keys())
            
            # Identify what changed
            removed_symbols = sorted(old_keys - new_keys)
            added_symbols = sorted(new_keys - old_keys)
            
            # The Swap
            self._positions = reconciled

        # 6. Logging & Saving (Outside Lock)
        if removed_symbols:
            self._logger.info(
                f"Sync: Pruned {len(removed_symbols)} positions not found in broker: {removed_symbols}"
            )
        
        if not old_keys and not new_keys:
            self._logger.debug("Sync: Broker and local state both empty.")
            return

        try:
            self.save_state()
            self._logger.info(
                "Sync: Completed successfully.",
                extra={
                    "total_managed": len(self._positions),
                    "added": len(added_symbols),
                    "removed": len(removed_symbols)
                }
            )
        except Exception as exc:
            self._logger.error(
                "Failure persisting synchronized positions: %s",
                exc,
                extra={"event": "position_manager_sync_persist_error"},
                exc_info=True,
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
        manager = self._persistent_state
        if manager is None:
            return
        snapshot = [position.to_dict() for position in self._positions.values()]
        current_symbols: set[str] = set()
        for entry in snapshot:
            symbol_value = str(entry.get("symbol", "")).strip().upper()
            if symbol_value:
                current_symbols.add(symbol_value)
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
        for entry in snapshot:
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
        self._persist_order_snapshots(manager)
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

    def _persist_fill(self, order: Order, quantity: int, fill_price: float) -> None:
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
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": int(quantity),
            "fill_price": float(fill_price),
            "status": order.status,
            "timestamp": timestamp_iso,
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

    def _handle_filled_order(self, order: Order) -> None:
        symbol_key = order.symbol
        qty = order.quantity if order.filled_quantity == 0 else order.filled_quantity
        qty = abs(qty)
        if qty <= 0:
            self._logger.warning(
                "Ignoring zero-quantity fill for order %s", order.order_id
            )
            return

        fill_price = order.fill_price
        if fill_price is None:
            self._logger.warning("Missing fill price for order %s", order.order_id)
            return

        self._persist_fill(order, qty, fill_price)

        side = order.side
        if not self.has_position(symbol_key):
            position_side: Side = "LONG" if side == "BUY" else "SHORT"
            self.open_position(
                symbol=symbol_key,
                side=position_side,
                quantity=qty,
                entry_price=fill_price,
                order_id=order.order_id,
            )
            return

        position = self._positions[symbol_key]
        if (position.side == "LONG" and side == "SELL") or (
            position.side == "SHORT" and side == "BUY"
        ):
            self._reduce_or_close_position(position, qty, fill_price)
        else:
            self._scale_position(position, qty, fill_price)

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
        self._daily_realized_pnl += realized
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


__all__ = ["Order", "Position", "PositionManager", "ActiveContract"]
