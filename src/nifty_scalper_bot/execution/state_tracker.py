"""SQLite-backed execution state tracker for order and position lifecycles."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nifty_scalper_bot.utils.logging import get_logger


@dataclass(slots=True)
class OrderState:
    """Serialised state for a tracked order."""

    order_id: str
    timestamp: float
    symbol: str
    side: str
    quantity: int
    status: str
    fill_price: float | None
    intent: str
    parent_id: str | None = None


@dataclass(slots=True)
class LifecycleStage:
    """Lifecycle stage metadata for a position."""

    stage: str
    updated_at: float
    details: dict[str, Any] | None = None


@dataclass(slots=True)
class PositionState:
    """Serialised snapshot for a trading position."""

    symbol: str
    quantity: int
    entry_price: float
    entry_time: float
    unrealized_pnl: float
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    trail_active: bool
    lifecycle_stage: str


class StateTrackerSettings(BaseSettings):
    """Configuration payload for :class:`StateTracker`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_path: str = Field(default="data/execution_state.db")


class StateTracker:
    """Persist and expose order and position state for execution flows."""

    def __init__(self, settings: StateTrackerSettings | None = None) -> None:
        """Initialise the state tracker and underlying SQLite connection.

        Args:
            settings: Optional override for tracker configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = get_logger(__name__)
        self._logger.debug(
            "Entered StateTracker.__init__",
            extra={"event": "state_tracker_init"},
        )
        try:
            self._settings = settings or StateTrackerSettings()
            self._path = Path(self._settings.db_path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.__init__: %s",
                exc,
                extra={"event": "state_tracker_init_error"},
                exc_info=exc,
            )
            raise

    def _initialize_schema(self) -> None:
        """Create required tables when the database is empty.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker._initialize_schema",
            extra={"event": "state_tracker_init_schema"},
        )
        try:
            with self._lock:
                cursor = self._conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        fill_price REAL,
                        intent TEXT NOT NULL,
                        parent_id TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        quantity INTEGER NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        stop_loss REAL,
                        tp1 REAL,
                        tp2 REAL,
                        trail_active INTEGER NOT NULL,
                        lifecycle_stage TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lifecycle_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        details_json TEXT
                    )
                    """
                )
                self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker._initialize_schema: %s",
                exc,
                extra={"event": "state_tracker_schema_error"},
                exc_info=exc,
            )
            raise

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.close",
            extra={"event": "state_tracker_close"},
        )
        try:
            with self._lock:
                self._conn.close()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.close: %s",
                exc,
                extra={"event": "state_tracker_close_error"},
                exc_info=exc,
            )

    def add_order(self, order_data: dict[str, Any]) -> str:
        """Persist a new order row and return the canonical order identifier.

        Args:
            order_data: Mapping of order fields to persist.

        Returns:
            str: Order identifier recorded for downstream lookup.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.add_order",
            extra={"event": "state_tracker_add_order", "order_data": order_data},
        )
        try:
            incoming_id = order_data.get("order_id") or order_data.get("id")
            order_id = (
                str(incoming_id) if incoming_id else f"local_{int(time.time() * 1000)}"
            )
            timestamp_raw = order_data.get("timestamp")
            timestamp = (
                float(timestamp_raw)
                if timestamp_raw is not None
                else float(time.time())
            )
            fill_price_raw = order_data.get("fill_price")
            fill_price = None if fill_price_raw is None else float(fill_price_raw)
            parent_raw = order_data.get("parent_id")
            parent_id = None if parent_raw in (None, "") else str(parent_raw)
            payload = OrderState(
                order_id=order_id,
                timestamp=timestamp,
                symbol=str(order_data.get("symbol", "")),
                side=str(order_data.get("side", "")),
                quantity=int(order_data.get("quantity", 0)),
                status=str(order_data.get("status", "unknown")),
                fill_price=fill_price,
                intent=str(order_data.get("intent", "ENTRY")),
                parent_id=parent_id,
            )
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO orders(
                        order_id, timestamp, symbol, side, quantity,
                        status, fill_price, intent, parent_id
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        payload.order_id,
                        payload.timestamp,
                        payload.symbol,
                        payload.side,
                        payload.quantity,
                        payload.status,
                        payload.fill_price,
                        payload.intent,
                        payload.parent_id,
                    ),
                )
                self._conn.commit()
            self._logger.info(
                "state_tracker_order_added",
                extra={"order_id": payload.order_id, "symbol": payload.symbol},
            )
            return payload.order_id
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.add_order: %s",
                exc,
                extra={"event": "state_tracker_add_order_error"},
                exc_info=exc,
            )
            raise

    def get_orders_by_status(self, status: str) -> list[dict[str, Any]]:
        """Return orders matching the provided status token.

        Args:
            status: Canonical status string to filter by.

        Returns:
            list[dict[str, Any]]: Matching order rows ordered by timestamp.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.get_orders_by_status",
            extra={"event": "state_tracker_orders_by_status", "status": status},
        )
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM orders WHERE status=? ORDER BY timestamp DESC",
                    (status,),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.get_orders_by_status: %s",
                exc,
                extra={"event": "state_tracker_orders_by_status_error"},
                exc_info=exc,
            )
            return []

    def get_position_state(self, symbol: str) -> dict[str, Any] | None:
        """Return the persisted position state for *symbol* if available.

        Args:
            symbol: Trading symbol to query.

        Returns:
            dict[str, Any] | None: Stored position payload when present.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.get_position_state",
            extra={"event": "state_tracker_get_position", "symbol": symbol},
        )
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM positions WHERE symbol=?",
                    (symbol,),
                ).fetchone()
            return None if row is None else dict(row)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.get_position_state: %s",
                exc,
                extra={"event": "state_tracker_get_position_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def update_position(self, symbol: str, updates: dict[str, Any]) -> None:
        """Create, update, or delete a position row for *symbol*.

        Args:
            symbol: Trading symbol to mutate.
            updates: Key-value pairs describing the desired change.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.update_position",
            extra={
                "event": "state_tracker_update_position",
                "symbol": symbol,
                "keys": list(updates.keys()),
            },
        )
        try:
            if not symbol:
                return
            delete_flag = bool(updates.get("delete"))
            canonical_symbol = str(symbol)
            with self._lock:
                if delete_flag:
                    self._conn.execute(
                        "DELETE FROM positions WHERE symbol=?",
                        (canonical_symbol,),
                    )
                    self._conn.commit()
                    self._logger.info(
                        "state_tracker_position_deleted",
                        extra={"symbol": canonical_symbol},
                    )
                    return
                row = self._conn.execute(
                    "SELECT * FROM positions WHERE symbol=?",
                    (canonical_symbol,),
                ).fetchone()
                columns = {
                    "symbol",
                    "quantity",
                    "entry_price",
                    "entry_time",
                    "unrealized_pnl",
                    "stop_loss",
                    "tp1",
                    "tp2",
                    "trail_active",
                    "lifecycle_stage",
                }
                existing = dict(row) if row is not None else {}
                merged: dict[str, Any] = {**existing}
                for key, value in updates.items():
                    if key in columns and value is not None:
                        merged[key] = value
                if not merged:
                    return
                merged.setdefault("symbol", canonical_symbol)
                merged.setdefault("entry_time", time.time())
                merged.setdefault("entry_price", 0.0)
                merged.setdefault("quantity", 0)
                merged.setdefault("unrealized_pnl", 0.0)
                merged.setdefault("stop_loss", None)
                merged.setdefault("tp1", None)
                merged.setdefault("tp2", None)
                merged.setdefault("trail_active", False)
                merged.setdefault("lifecycle_stage", "UNKNOWN")
                entry_price_raw = merged.get("entry_price", 0.0)
                entry_price = float(entry_price_raw or 0.0)
                entry_time_raw = merged.get("entry_time", time.time())
                entry_time = float(entry_time_raw or time.time())
                unrealized_raw = merged.get("unrealized_pnl", 0.0)
                unrealized = float(unrealized_raw or 0.0)
                stop_raw = merged.get("stop_loss")
                stop_loss = None if stop_raw is None else float(stop_raw)
                tp1_raw = merged.get("tp1")
                tp1 = None if tp1_raw is None else float(tp1_raw)
                tp2_raw = merged.get("tp2")
                tp2 = None if tp2_raw is None else float(tp2_raw)
                position = PositionState(
                    symbol=str(merged["symbol"]),
                    quantity=int(merged.get("quantity", 0)),
                    entry_price=entry_price,
                    entry_time=entry_time,
                    unrealized_pnl=unrealized,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    trail_active=bool(merged.get("trail_active", False)),
                    lifecycle_stage=str(merged.get("lifecycle_stage", "UNKNOWN")),
                )
                payload = asdict(position)
                payload["trail_active"] = int(position.trail_active)
                self._conn.execute(
                    """
                    INSERT INTO positions(
                        symbol, quantity, entry_price, entry_time, unrealized_pnl,
                        stop_loss, tp1, tp2, trail_active, lifecycle_stage
                    ) VALUES(:symbol,:quantity,:entry_price,:entry_time,:unrealized_pnl,
                             :stop_loss,:tp1,:tp2,:trail_active,:lifecycle_stage)
                    ON CONFLICT(symbol) DO UPDATE SET
                        quantity=excluded.quantity,
                        entry_price=excluded.entry_price,
                        entry_time=excluded.entry_time,
                        unrealized_pnl=excluded.unrealized_pnl,
                        stop_loss=excluded.stop_loss,
                        tp1=excluded.tp1,
                        tp2=excluded.tp2,
                        trail_active=excluded.trail_active,
                        lifecycle_stage=excluded.lifecycle_stage
                    """,
                    payload,
                )
                self._conn.commit()
            self._logger.info(
                "state_tracker_position_upserted",
                extra={"symbol": canonical_symbol, "quantity": payload["quantity"]},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.update_position: %s",
                exc,
                extra={
                    "event": "state_tracker_update_position_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return all positions with non-zero quantity.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Open positions snapshot.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.get_open_positions",
            extra={"event": "state_tracker_get_open_positions"},
        )
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM positions WHERE quantity != 0",
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.get_open_positions: %s",
                exc,
                extra={"event": "state_tracker_get_open_positions_error"},
                exc_info=exc,
            )
            return []

    def record_lifecycle_event(
        self,
        position_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Append a lifecycle event for diagnostics and reconciliation.

        Args:
            position_id: Canonical symbol or synthetic identifier.
            event_type: Lifecycle event name.
            details: Optional JSON-serialisable payload for context.
            timestamp: Explicit event timestamp override.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.record_lifecycle_event",
            extra={
                "event": "state_tracker_record_lifecycle_event",
                "position_id": position_id,
                "event_type": event_type,
            },
        )
        try:
            encoded = json.dumps(details or {})
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO lifecycle_events(
                        position_id, timestamp, event_type, details_json
                    )
                    VALUES(?,?,?,?)
                    """,
                    (
                        position_id,
                        float(timestamp) if timestamp is not None else time.time(),
                        event_type,
                        encoded,
                    ),
                )
                self._conn.commit()
            self._logger.info(
                "state_tracker_lifecycle_event_recorded",
                extra={"position_id": position_id, "event_type": event_type},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.record_lifecycle_event: %s",
                exc,
                extra={"event": "state_tracker_record_lifecycle_event_error"},
                exc_info=exc,
            )

    def list_lifecycle_events(self, position_id: str) -> list[dict[str, Any]]:
        """Return lifecycle events captured for the provided *position_id*.

        Args:
            position_id: Symbol identifier to filter events.

        Returns:
            list[dict[str, Any]]: Ordered lifecycle events.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StateTracker.list_lifecycle_events",
            extra={
                "event": "state_tracker_list_lifecycle_events",
                "position_id": position_id,
            },
        )
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT position_id, timestamp, event_type, details_json
                    FROM lifecycle_events
                    WHERE position_id=?
                    ORDER BY timestamp ASC
                    """,
                    (position_id,),
                ).fetchall()
            events: list[dict[str, Any]] = []
            for row in rows:
                payload = dict(row)
                try:
                    payload["details_json"] = json.loads(
                        payload.get("details_json") or "{}"
                    )
                except json.JSONDecodeError:
                    payload["details_json"] = {}
                events.append(payload)
            return events
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.list_lifecycle_events: %s",
                exc,
                extra={"event": "state_tracker_list_lifecycle_events_error"},
                exc_info=exc,
            )
            return []

    def reconcile_with_broker(self, broker_positions: Iterable[dict[str, Any]]) -> None:
        """Align persisted positions with the provided broker snapshot.

        Args:
            broker_positions: Iterable of broker-reported positions.

        Returns:
            None.

        Raises:
            None.
        """

        broker_positions = list(broker_positions)
        self._logger.debug(
            "Entered StateTracker.reconcile_with_broker",
            extra={
                "event": "state_tracker_reconcile",
                "broker_count": len(broker_positions),
            },
        )
        try:
            current_positions = {
                pos["symbol"]: pos for pos in self.get_open_positions()
            }
            broker_map: dict[str, dict[str, Any]] = {}
            for entry in broker_positions:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("symbol", "")).strip()
                if not symbol:
                    continue
                broker_map[symbol] = entry
            for symbol, local in current_positions.items():
                broker_quantity = int(broker_map.get(symbol, {}).get("quantity", 0))
                if symbol not in broker_map or broker_quantity == 0:
                    self.update_position(symbol, {"delete": True})
                    self._logger.info(
                        "state_tracker_reconcile_removed",
                        extra={"symbol": symbol},
                    )
            for symbol, snapshot in broker_map.items():
                quantity = int(snapshot.get("quantity", 0))
                if quantity == 0:
                    self.update_position(symbol, {"delete": True})
                    continue
                local = current_positions.get(symbol, {})
                updates = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": float(snapshot.get("entry_price", 0.0)),
                    "entry_time": float(snapshot.get("entry_time", time.time())),
                    "unrealized_pnl": float(snapshot.get("unrealized_pnl", 0.0)),
                    "stop_loss": snapshot.get("stop_loss"),
                    "tp1": snapshot.get("tp1"),
                    "tp2": snapshot.get("tp2"),
                    "trail_active": bool(snapshot.get("trail_active", False)),
                    "lifecycle_stage": str(
                        (
                            snapshot.get(
                                "lifecycle_stage",
                                local.get("lifecycle_stage", "ACTIVE"),
                            )
                            if local
                            else snapshot.get("lifecycle_stage", "ACTIVE")
                        ),
                    ),
                }
                self.update_position(symbol, updates)
            self._logger.info(
                "state_tracker_reconcile_complete",
                extra={"broker_count": len(broker_map)},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in StateTracker.reconcile_with_broker: %s",
                exc,
                extra={"event": "state_tracker_reconcile_error"},
                exc_info=exc,
            )


__all__ = [
    "LifecycleStage",
    "OrderState",
    "PositionState",
    "StateTracker",
    "StateTrackerSettings",
]
