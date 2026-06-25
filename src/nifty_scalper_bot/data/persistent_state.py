"""Durable persistence for trades, positions, and shadow equity."""

from __future__ import annotations
import os
import json
import signal
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import FrameType
from typing import Any, Callable, Mapping, cast

try:  # pragma: no cover - optional dependency guard
    import sqlite3
except Exception:  # pragma: no cover - fallback when sqlite unavailable
    sqlite3 = None  # type: ignore[assignment]

from nifty_scalper_bot.infra.structured_logger import emit_diag
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.serialization import to_json_safe

TradeDict = dict[str, object]
FillDict = dict[str, object]
PositionDict = dict[str, object]
OrderDict = dict[str, object]
BracketDict = dict[str, object]
ShadowPositions = dict[str, float]
FINAL_ORDER_STATUSES: tuple[str, ...] = (
    "FILLED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
)
SignalHandler = Callable[[int, FrameType | None], Any] | int | None


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Return *value* converted to float, falling back to *default*."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: object | None) -> float | None:
    """Return ``None`` when conversion of *value* fails."""

    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class PersistentStateDB:
    """Lightweight SQLite facade for trading state snapshots.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        None.

    Raises:
        RuntimeError: If the database cannot be initialised.
    """

    def __init__(self, db_path: Path) -> None:
        self._logger = get_logger(__name__)
        self._lock = RLock()
        
        # ✅ FIX 1: Auto-create directory
        # Fallback to /tmp if the path appears to be in a readonly location
        if not db_path:
            db_path = Path(os.getenv("STATE_DB_PATH", "/tmp/state.db"))
        self._path = Path(db_path).resolve()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._logger.critical("Failed to create DB directory: %s", exc)
            raise RuntimeError(f"Cannot create DB directory: {exc}") from exc

        if sqlite3 is None:
            raise RuntimeError("SQLite unavailable for persistent storage")

        try:
            # ✅ FIX 2: Connection with Crash Recovery (Auto-Unlock)
            try:
                self._conn = sqlite3.connect(
                    str(self._path), 
                    check_same_thread=False, 
                    timeout=30.0
                )
            except sqlite3.OperationalError as e:
                # If DB is locked from a previous crash, nuke the WAL files and retry
                if "locked" in str(e).lower():
                    self._logger.warning(f"⚠️ DB locked on startup (stale WAL?), forcing cleanup: {e}")
                    
                    # Force delete temporary journal files
                    with suppress(Exception):
                        wal_path = self._path.with_name(self._path.name + "-wal")
                        shm_path = self._path.with_name(self._path.name + "-shm")
                        # Use unlink(missing_ok=True) if on Python 3.8+, or try/except
                        if wal_path.exists(): wal_path.unlink()
                        if shm_path.exists(): shm_path.unlink()
                    
                    # Retry connection
                    self._conn = sqlite3.connect(
                        str(self._path), 
                        check_same_thread=False, 
                        timeout=30.0
                    )
                else:
                    raise e
            
            # ✅ FIX 3: Performance Tuning for Scalping
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            
            self._create_tables()
            
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.__init__: %s", exc)
            # Cleanup connection if init failed
            if hasattr(self, '_conn'):
                with suppress(Exception):
                    self._conn.close()
            raise RuntimeError("Failed to initialise persistent database") from exc

    def _create_tables(self) -> None:
        """Create required tables when they are missing.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If table creation fails.
        """

        try:
            with self._conn:  # type: ignore[attr-defined]
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fills (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        order_id TEXT,
                        payload TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_ts TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shadow_equity (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        equity REAL NOT NULL,
                        payload TEXT,
                        updated_ts TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        updated_ts TEXT NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS brackets (
                        entry_id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_ts TEXT NOT NULL
                    )
                    """
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB._create_tables: %s", exc)
            raise RuntimeError("Failed to create persistence tables") from exc

    def save_trade(self, trade: TradeDict) -> None:
        """Persist *trade* payload atomically.

        Args:
            trade: Trade dictionary to be recorded.

        Returns:
            None.

        Raises:
            RuntimeError: If persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.save_trade",
            extra={"event": "persistent_db_save_trade"},
        )
        payload = json.dumps(to_json_safe(dict(trade)), ensure_ascii=False, default=str)
        ts = str(trade.get("timestamp") or datetime.now(timezone.utc).isoformat())
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        "INSERT INTO trades(ts, payload) VALUES(?, ?)",
                        (ts, payload),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.save_trade: %s", exc)
            raise RuntimeError("Failed to persist trade") from exc

    def save_fill(self, fill: FillDict) -> None:
        """Persist executed *fill* payload atomically.

        Args:
            fill: Mapping describing the executed fill event.

        Returns:
            None.

        Raises:
            RuntimeError: If persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.save_fill",
            extra={"event": "persistent_db_save_fill"},
        )
        payload = json.dumps(to_json_safe(dict(fill)), ensure_ascii=False, default=str)
        ts = str(fill.get("timestamp") or datetime.now(timezone.utc).isoformat())
        order_id_raw = fill.get("order_id")
        order_id = str(order_id_raw).strip() if order_id_raw is not None else None
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        "INSERT INTO fills(ts, order_id, payload) VALUES(?, ?, ?)",
                        (ts, order_id, payload),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.save_fill: %s", exc)
            raise RuntimeError("Failed to persist fill") from exc

    def load_trades(self) -> list[TradeDict]:
        """Load persisted trades in chronological order.

        Args:
            None.

        Returns:
            List of trade payloads.

        Raises:
            RuntimeError: If retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_trades",
            extra={"event": "persistent_db_load_trades"},
        )
        try:
            with self._lock:
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    "SELECT payload FROM trades ORDER BY id ASC"
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.load_trades: %s", exc)
            raise RuntimeError("Failed to load trades") from exc
        trades: list[TradeDict] = []
        for row in rows:
            try:
                decoded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_db_trade_decode_failed",
                    exc_info=True,
                )
                continue
            if isinstance(decoded, Mapping):
                trades.append(dict(decoded))
        return trades

    def load_fills(self, limit: int | None = None) -> list[FillDict]:
        """Load persisted fills ordered by insertion.

        Args:
            limit: Optional maximum number of fills to return.

        Returns:
            List of fill payloads.

        Raises:
            RuntimeError: If retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_fills",
            extra={"event": "persistent_db_load_fills", "limit": limit},
        )
        try:
            with self._lock:
                query = "SELECT payload FROM fills ORDER BY id ASC"
                params = ()
                if limit is not None and limit > 0:
                    query = f"{query} LIMIT ?"
                    params = (int(limit),)
                cursor = self._conn.execute(query, params)  # type: ignore[attr-defined]
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.load_fills: %s", exc)
            raise RuntimeError("Failed to load fills") from exc
        fills: list[FillDict] = []
        for row in rows:
            try:
                decoded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_db_fill_decode_failed",
                    exc_info=True,
                )
                continue
            if isinstance(decoded, Mapping):
                fills.append(dict(decoded))
        return fills

    def save_position(self, position: PositionDict) -> None:
        """Persist or remove *position* snapshot.

        Args:
            position: Position payload including ``symbol`` and ``quantity``.

        Returns:
            None.

        Raises:
            RuntimeError: If persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.save_position",
            extra={"event": "persistent_db_save_position"},
        )
        symbol = str(position.get("symbol", "")).strip().upper()
        if not symbol:
            self._logger.info(
                "Condition met: persistent_db_save_position_missing_symbol",
                extra={"event": "persistent_db_missing_symbol"},
            )
            return
        quantity = _coerce_float(position.get("quantity", 0.0))
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    if abs(quantity) <= 0.0:
                        self._conn.execute(
                            "DELETE FROM positions WHERE symbol=?",
                            (symbol,),
                        )
                    else:
                        encoded = json.dumps(to_json_safe(dict(position)), ensure_ascii=False, default=str)
                        ts = datetime.now(timezone.utc).isoformat()
                        self._conn.execute(
                            """
                            INSERT INTO positions(symbol, payload, updated_ts)
                            VALUES(?, ?, ?)
                            ON CONFLICT(symbol) DO UPDATE SET
                                payload=excluded.payload,
                                updated_ts=excluded.updated_ts
                            """,
                            (symbol, encoded, ts),
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.save_position: %s", exc)
            raise RuntimeError("Failed to persist position") from exc

    def save_shadow_equity(
        self, equity: float, payload: Mapping[str, object] | None = None
    ) -> None:
        """Persist latest *equity* snapshot.

        Args:
            equity: Aggregated equity value.
            payload: Optional serialised exposures or metadata.

        Returns:
            None.

        Raises:
            RuntimeError: If persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.save_shadow_equity",
            extra={"event": "persistent_db_save_shadow"},
        )
        snapshot = {
            "equity": float(equity),
            "payload": dict(payload) if payload is not None else {},
        }
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        """
                        INSERT INTO shadow_equity(id, equity, payload, updated_ts)
                        VALUES(1, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            equity=excluded.equity,
                            payload=excluded.payload,
                            updated_ts=excluded.updated_ts
                        """,
                        (
                            float(equity),
                            json.dumps(to_json_safe(snapshot["payload"]), ensure_ascii=False, default=str),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PersistentStateDB.save_shadow_equity: %s", exc
            )
            raise RuntimeError("Failed to persist shadow equity") from exc

    def _serialize_order(self, order: dict) -> str:
        """Serialize order dict with Enum, datetime, Decimal handling.
        
        ✅ PRODUCTION FIX: Handles OrderType, OrderStatus, and other Enums.
        """
        from enum import Enum
        from datetime import datetime, date
        from decimal import Decimal
        
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
        
        sanitized = _sanitize(order)
        return json.dumps(sanitized, ensure_ascii=False, default=str)

# ============================================================
# STEP 2: REPLACE THE save_order METHOD (starting at line ~451)
# ============================================================

    def save_order(self, order: OrderDict) -> None:
        """Persist or update an *order* snapshot.

        Args:
            order: Order payload including ``order_id`` and ``status``.

        Returns:
            None.

        Raises:
            RuntimeError: If persistence fails.
            
        ✅ PRODUCTION FIX: Uses _serialize_order for Enum handling.
        """

        self._logger.debug(
            "Entered PersistentStateDB.save_order",
            extra={"event": "persistent_db_save_order"},
        )
        
        # ✅ FIX: Defensive check for order_id
        order_id = str(order.get("order_id", "")).strip()
        if not order_id:
            # Try alternative keys
            order_id = str(order.get("broker_order_id", "") or order.get("id", "")).strip()
            if order_id:
                order["order_id"] = order_id
            else:
                self._logger.warning(
                    "save_order: Missing order_id, skipping persist",
                    extra={"event": "persistent_db_missing_order_id", "symbol": order.get("symbol")},
                )
                return
                
        status = str(order.get("status", "")).strip().upper() or "PENDING"
        
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        """
                        INSERT INTO orders(order_id, status, payload, updated_ts)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(order_id) DO UPDATE SET
                            status=excluded.status,
                            payload=excluded.payload,
                            updated_ts=excluded.updated_ts
                        """,
                        (
                            order_id,
                            status,
                            self._serialize_order(dict(order)),  # ✅ FIX: Use serializer
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.save_order: %s", exc)
            raise RuntimeError("Failed to persist order") from exc
            

    
    def load_open_orders(self) -> list[OrderDict]:
        """Return persisted orders that remain open.

        Args:
            None.

        Returns:
            List of order payloads still considered open.

        Raises:
            RuntimeError: If the query fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_open_orders",
            extra={"event": "persistent_db_load_open_orders"},
        )
        try:
            with self._lock:
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    """
                    SELECT payload FROM orders
                    WHERE UPPER(status) NOT IN (?, ?, ?, ?)
                    """,
                    FINAL_ORDER_STATUSES,
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.load_open_orders: %s", exc)
            raise RuntimeError("Failed to load open orders") from exc
        orders: list[OrderDict] = []
        for row in rows:
            try:
                decoded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_db_open_order_decode_failed",
                    exc_info=True,
                )
                continue
            if isinstance(decoded, Mapping):
                orders.append(dict(decoded))
        return orders

    def save_bracket(self, bracket: BracketDict) -> None:
        """Save bracket with defensive order_id handling."""
        # ✅ FIX: Ensure order_id exists
        if "order_id" not in bracket or not bracket.get("order_id"):
            # Try alternate field names
            bracket["order_id"] = (
                bracket.get("id") or 
                bracket.get("entry_id") or 
                bracket.get("bracket_id") or
                f"auto_{bracket.get('symbol', 'unknown')}_{int(time.time())}"
            )

        self._logger.debug(
            "Entered PersistentStateDB.save_bracket",
            extra={"event": "persistent_db_save_bracket"},
        )
        entry_id = str(bracket.get("entry_id", "")).strip()
        if not entry_id:
            self._logger.info(
                "Condition met: persistent_db_save_bracket_missing_id",
                extra={"event": "persistent_db_missing_entry_id"},
            )
            return
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        """
                        INSERT INTO brackets(entry_id, payload, updated_ts)
                        VALUES(?, ?, ?)
                        ON CONFLICT(entry_id) DO UPDATE SET
                            payload=excluded.payload,
                            updated_ts=excluded.updated_ts
                        """,
                        (
                            entry_id,
                            self._serialize_order(dict(bracket)),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.save_bracket: %s", exc)
            raise RuntimeError("Failed to persist bracket") from exc

    def remove_bracket(self, entry_id: str) -> None:
        """Remove persisted bracket identified by *entry_id*.

        Args:
            entry_id: Entry order identifier for the bracket.

        Returns:
            None.

        Raises:
            RuntimeError: If deletion fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.remove_bracket",
            extra={"event": "persistent_db_remove_bracket", "entry_id": entry_id},
        )
        try:
            with self._lock:
                with self._conn:  # type: ignore[attr-defined]
                    self._conn.execute(
                        "DELETE FROM brackets WHERE entry_id=?",
                        (entry_id.strip(),),
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.remove_bracket: %s", exc)
            raise RuntimeError("Failed to remove bracket") from exc

    def load_brackets(self) -> list[BracketDict]:
        """Load all persisted bracket snapshots.

        Args:
            None.

        Returns:
            List of bracket payloads.

        Raises:
            RuntimeError: If retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_brackets",
            extra={"event": "persistent_db_load_brackets"},
        )
        try:
            with self._lock:
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    "SELECT payload FROM brackets"
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.load_brackets: %s", exc)
            raise RuntimeError("Failed to load brackets") from exc
        brackets: list[BracketDict] = []
        for row in rows:
            try:
                decoded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_db_bracket_decode_failed",
                    exc_info=True,
                )
                continue
            if isinstance(decoded, Mapping):
                brackets.append(dict(decoded))
        return brackets

    def load_positions(self) -> list[PositionDict]:
        """Load persisted positions from SQLite.

        Args:
            None.

        Returns:
            List of position payloads.

        Raises:
            RuntimeError: If retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_positions",
            extra={"event": "persistent_db_load_positions"},
        )
        try:
            with self._lock:
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    "SELECT payload FROM positions"
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.load_positions: %s", exc)
            raise RuntimeError("Failed to load positions") from exc
        positions: list[PositionDict] = []
        for row in rows:
            try:
                decoded = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_db_position_decode_failed",
                    exc_info=True,
                )
                continue
            if isinstance(decoded, Mapping):
                positions.append(dict(decoded))
        return positions

    def load_shadow_equity(self) -> Mapping[str, object]:
        """Load persisted shadow equity snapshot.

        Args:
            None.

        Returns:
            Mapping containing ``equity`` and optional payload.

        Raises:
            RuntimeError: If retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.load_shadow_equity",
            extra={"event": "persistent_db_load_shadow"},
        )
        try:
            with self._lock:
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    "SELECT equity, payload FROM shadow_equity WHERE id=1"
                )
                row = cursor.fetchone()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PersistentStateDB.load_shadow_equity: %s", exc
            )
            raise RuntimeError("Failed to load shadow equity") from exc
        if not row:
            return {}
        exposures: dict[str, float] = {}
        try:
            payload_raw = row[1]
            if payload_raw is not None:
                decoded = json.loads(payload_raw)
                if isinstance(decoded, Mapping):
                    exposures = {
                        str(key): _coerce_float(value) for key, value in decoded.items()
                    }
        except (TypeError, json.JSONDecodeError):
            self._logger.debug(
                "persistent_db_shadow_decode_failed",
                exc_info=True,
            )
        equity_value = float(row[0]) if row[0] is not None else 0.0
        payload: dict[str, object] = {"equity": equity_value}
        if exposures:
            payload["positions"] = exposures
        return payload

    def flush(self) -> None:
        """Commit pending transactions to disk.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If commit fails.
        """

        self._logger.debug(
            "Entered PersistentStateDB.flush",
            extra={"event": "persistent_db_flush"},
        )
        try:
            with self._lock:
                self._conn.commit()  # type: ignore[attr-defined]
            emit_diag(
                self._logger,
                "persistent_db_flush",
                reason="ok",
                severity="info",
            )
        except Exception as exc:  # noqa: BLE001
            emit_diag(
                self._logger,
                "persistent_db_flush_error",
                reason="flush_error",
                severity="critical",
                alert=True,
                error=str(exc),
            )
            self._logger.error("Failure in PersistentStateDB.flush: %s", exc)
            raise RuntimeError("Failed to flush persistent database") from exc

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        try:
            with self._lock:
                self._conn.close()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in PersistentStateDB.close: %s", exc)


class PersistentStateManager:
    """Coordinate durable storage for trades, positions, and shadow equity.

    Args:
        base_path: Directory used to store persistence artefacts.

    Returns:
        None.

    Raises:
        RuntimeError: If SQLite initialisation fails.
    """

    def __init__(self, base_path: str | Path = "data") -> None:
        self._logger = get_logger(__name__)
        self._lock = RLock()
        
        # ✅ FIX: Robust directory creation with fallback
        self._base_path = Path(base_path).resolve()
        
        try:
            self._base_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"✅ Data directory ready: {self._base_path}")
        except OSError as exc:
            self._logger.error(f"❌ Failed to create base directory: {exc}")
            # Fallback to /tmp for Railway/Cloud
            import tempfile
            fallback = Path(tempfile.gettempdir()) / "nifty_bot_data"
            fallback.mkdir(parents=True, exist_ok=True)
            self._base_path = fallback
            self._logger.warning(f"⚠️ Using fallback directory: {self._base_path}")

        self._mode = "json"
        self._trade_cache: list[TradeDict] = []
        self._fill_cache: list[FillDict] = []
        self._position_cache: dict[str, PositionDict] = {}
        self._orders_cache: dict[str, OrderDict] = {}
        self._bracket_cache: dict[str, BracketDict] = {}
        self._shadow_state: dict[str, object] = {}
        self._signal_handlers: dict[int, SignalHandler] = {}
        self._db: PersistentStateDB | None = None
        self._auto_flush_interval = 5.0
        self._auto_flush_threshold = 10
        self._last_auto_flush = 0.0
        self._pending_flush_events = 0
        self._health_status = "initialising"
        self._last_write_wall_clock: datetime | None = None
        self._last_flush_wall_clock: datetime | None = None
        self._last_error: str | None = None
        self._trades_path = self._base_path / "trades.json"
        self._fills_path = self._base_path / "fills.json"
        self._positions_path = self._base_path / "positions.json"
        self._orders_path = self._base_path / "orders.json"
        self._brackets_path = self._base_path / "brackets.json"
        self._shadow_path = self._base_path / "shadow_state.json"
        self._logger.debug(
            "Entered PersistentStateManager.__init__",
            extra={"event": "persistent_state_init", "base_path": str(self._base_path)},
        )
        if sqlite3 is not None:
            try:
                self._initialise_sqlite()
                self._mode = "sqlite"
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in PersistentStateManager.__init__: %s",
                    exc,
                )
                self._mode = "json"
                self._db = None
        self._load_existing_state()
        self._register_signal(signal.SIGINT)
        self._register_signal(signal.SIGTERM)
        self._health_status = "ok"

    # ------------------------------------------------------------------
    def save_trade(self, trade: TradeDict) -> None:
        """Persist *trade* to the underlying storage engine.

        Args:
            trade: Mapping describing the recorded trade event.

        Returns:
            None.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.save_trade",
            extra={"event": "persistent_state_save_trade"},
        )
        payload = dict(trade)
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._trade_cache.append(payload)
        try:
            if self._mode == "sqlite":
                self._write_trade_sqlite(payload)
            else:
                self._write_trades_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_trade: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def save_fill(self, fill: FillDict) -> None:
        """Persist *fill* events to the configured storage engine.

        Args:
            fill: Mapping describing the executed fill payload.

        Returns:
            None.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.save_fill",
            extra={"event": "persistent_state_save_fill"},
        )
        payload = dict(fill)
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._fill_cache.append(payload)
        try:
            if self._mode == "sqlite":
                self._write_fill_sqlite(payload)
            else:
                self._write_fills_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_fill: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def save_position(self, position: PositionDict) -> None:
        """Persist a single *position* snapshot, removing it when flat.

        Args:
            position: Mapping representing a position state payload.

        Returns:
            None.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.save_position",
            extra={"event": "persistent_state_save_position"},
        )
        payload = dict(position)
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            self._logger.info(
                "Condition met: save_position_missing_symbol",
                extra={"event": "persistent_state_missing_symbol"},
            )
            return
        quantity = _coerce_float(payload.get("quantity", 0.0))
        payload["quantity"] = int(quantity)
        if "side" not in payload:
            payload["side"] = "LONG" if quantity >= 0 else "SHORT"
        entry_price_value = _optional_float(payload.get("entry_price"))
        if entry_price_value is None:
            for candidate in ("average_price", "price", "ltp", "last_price"):
                fallback = _optional_float(payload.get(candidate))
                if fallback is not None:
                    entry_price_value = fallback
                    break
        payload["entry_price"] = _coerce_float(entry_price_value or 0.0)
        current_price_value = _optional_float(payload.get("current_price"))
        if current_price_value is None:
            payload["current_price"] = payload["entry_price"]
        else:
            payload["current_price"] = _coerce_float(current_price_value)
        if "entry_time" not in payload or not str(payload["entry_time"]).strip():
            timestamp_source = payload.get("timestamp") or payload.get("created_at")
            if isinstance(timestamp_source, str) and timestamp_source.strip():
                payload["entry_time"] = timestamp_source
            else:
                payload["entry_time"] = datetime.now(timezone.utc).isoformat()
        else:
            payload["entry_time"] = str(payload["entry_time"])
        with self._lock:
            if abs(quantity) <= 0.0:
                self._position_cache.pop(symbol, None)
            else:
                payload["symbol"] = symbol
                self._position_cache[symbol] = payload
        try:
            if self._mode == "sqlite":
                self._write_position_sqlite(
                    symbol,
                    payload if abs(quantity) > 0.0 else None,
                )
            else:
                self._write_positions_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_position: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def save_shadow_equity(
        self, equity: float, positions: Mapping[str, float] | None = None
    ) -> None:
        """Persist the latest *equity* and optional per-symbol exposures.

        Args:
            equity: Aggregated paper equity used for drift detection.
            positions: Optional mapping of symbol exposure for recovery.

        Returns:
            None.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.save_shadow_equity",
            extra={"event": "persistent_state_save_shadow"},
        )
        snapshot: dict[str, object] = {"equity": float(equity)}
        if positions is not None:
            snapshot["positions"] = {
                str(key): _coerce_float(value) for key, value in positions.items()
            }
        with self._lock:
            self._shadow_state = snapshot
        try:
            if self._mode == "sqlite":
                self._write_shadow_sqlite(snapshot)
            else:
                self._write_shadow_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_shadow_equity: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def save_order(self, order: OrderDict) -> None:
        """Persist or update an individual *order* snapshot.

        Args:
            order: Order payload containing identifiers and status fields.

        Returns:
            None.

        Raises:
            Exception: Propagated when persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.save_order",
            extra={"event": "persistent_state_save_order"},
        )
        payload = dict(order)
        order_id = str(payload.get("order_id", "")).strip()
        if not order_id:
            self._logger.info(
                "Condition met: save_order_missing_id",
                extra={"event": "persistent_state_order_missing_id"},
            )
            return
        status = str(payload.get("status", "")).strip().upper() or "PENDING"
        payload["status"] = status
        with self._lock:
            if status in FINAL_ORDER_STATUSES:
                self._orders_cache.pop(order_id, None)
            else:
                self._orders_cache[order_id] = payload
        try:
            if self._mode == "sqlite" and self._db is not None:
                self._db.save_order(payload)
            else:
                self._write_orders_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_order: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def load_open_orders(self) -> list[OrderDict]:
        """Return persisted orders that remain open.

        Args:
            None.

        Returns:
            List of open order payloads.

        Raises:
            Exception: Propagated when retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.load_open_orders",
            extra={"event": "persistent_state_load_orders"},
        )
        if self._mode == "sqlite" and self._db is not None:
            try:
                orders = self._db.load_open_orders()
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in load_open_orders: %s", exc)
                orders = []
            normalized: dict[str, OrderDict] = {}
            for item in orders:
                if not isinstance(item, Mapping):
                    continue
                order_id = str(item.get("order_id", "")).strip()
                if order_id:
                    normalized[order_id] = dict(item)
            with self._lock:
                self._orders_cache = normalized
            return [dict(entry) for entry in normalized.values()]
        with self._lock:
            return [dict(entry) for entry in self._orders_cache.values()]

    def save_bracket(self, bracket: BracketDict) -> None:
        """Persist *bracket* state for recovery."""

        self._logger.debug(
            "Entered PersistentStateManager.save_bracket",
            extra={"event": "persistent_state_save_bracket"},
        )
        payload = dict(bracket)
        entry_id = str(payload.get("entry_id", "")).strip()
        if not entry_id:
            self._logger.info(
                "Condition met: save_bracket_missing_entry",
                extra={"event": "persistent_state_bracket_missing"},
            )
            return
        with self._lock:
            self._bracket_cache[entry_id] = payload
        try:
            if self._mode == "sqlite" and self._db is not None:
                self._db.save_bracket(payload)
            else:
                self._write_brackets_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in save_bracket: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def remove_bracket(self, entry_id: str) -> None:
        """Remove persisted bracket identified by *entry_id*.

        Args:
            entry_id: Entry identifier associated with the bracket.

        Returns:
            None.

        Raises:
            Exception: Propagated when deletion fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.remove_bracket",
            extra={"event": "persistent_state_remove_bracket", "entry_id": entry_id},
        )
        with self._lock:
            self._bracket_cache.pop(entry_id, None)
        try:
            if self._mode == "sqlite" and self._db is not None:
                self._db.remove_bracket(entry_id)
            else:
                self._write_brackets_json()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in remove_bracket: %s", exc)
            self._record_failure(exc)
            raise
        self._record_write_success()
        self._maybe_auto_flush()

    def load_brackets(self) -> list[BracketDict]:
        """Return persisted bracket states.

        Args:
            None.

        Returns:
            List of bracket payloads.

        Raises:
            Exception: Propagated when retrieval fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.load_brackets",
            extra={"event": "persistent_state_load_brackets"},
        )
        if self._mode == "sqlite" and self._db is not None:
            try:
                payloads = self._db.load_brackets()
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in load_brackets: %s", exc)
                payloads = []
            normalized: dict[str, BracketDict] = {}
            for item in payloads:
                if not isinstance(item, Mapping):
                    continue
                entry_id = str(item.get("entry_id", "")).strip()
                if entry_id:
                    normalized[entry_id] = dict(item)
            with self._lock:
                self._bracket_cache = normalized
            return [dict(entry) for entry in normalized.values()]
        with self._lock:
            return [dict(entry) for entry in self._bracket_cache.values()]

    def load_trades(self) -> list[TradeDict]:
        """Return a snapshot of persisted trade history.

        Args:
            None.

        Returns:
            List of trade payloads in chronological order.

        Raises:
            None.
        """

        with self._lock:
            return [dict(entry) for entry in self._trade_cache]

    def load_fills(self) -> list[FillDict]:
        """Return a snapshot of persisted fills.

        Args:
            None.

        Returns:
            List of fill payloads in chronological order.

        Raises:
            None.
        """

        with self._lock:
            return [dict(entry) for entry in self._fill_cache]

    def load_positions(self) -> list[PositionDict]:
        """Return persisted position snapshots.

        Args:
            None.

        Returns:
            List of saved position payloads.

        Raises:
            None.
        """

        with self._lock:
            return [dict(entry) for entry in self._position_cache.values()]

    def load_shadow_state(self) -> tuple[float | None, ShadowPositions]:
        """Return persisted shadow equity and exposures.

        Args:
            None.

        Returns:
            Tuple of optional equity value and exposure mapping.

        Raises:
            None.
        """

        with self._lock:
            equity = self._shadow_state.get("equity")
            positions_raw = self._shadow_state.get("positions", {})
        parsed_equity = _optional_float(equity)
        positions: ShadowPositions = {}
        if isinstance(positions_raw, Mapping):
            for key, value in positions_raw.items():
                maybe_float = _optional_float(value)
                if maybe_float is not None:
                    positions[str(key)] = maybe_float
        return parsed_equity, positions

    def flush(self) -> None:
        """Force pending state to durable storage.

        Args:
            None.

        Returns:
            None.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered PersistentStateManager.flush",
            extra={"event": "persistent_state_flush"},
        )
        with self._lock:
            try:
                if self._mode == "sqlite" and self._db is not None:
                    self._db.flush()
                else:
                    self._write_trades_json()
                    self._write_fills_json()
                    self._write_positions_json()
                    self._write_orders_json()
                    self._write_brackets_json()
                    self._write_shadow_json()
                self._pending_flush_events = 0
                self._last_auto_flush = time.monotonic()
                self._record_flush_success()
                emit_diag(
                    self._logger,
                    "persistent_state_flush_success",
                    reason="ok",
                    severity="info",
                    mode=self._mode,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_failure(exc)
                emit_diag(
                    self._logger,
                    "persistent_state_flush_failure",
                    reason="flush_error",
                    severity="critical",
                    alert=True,
                    mode=self._mode,
                    error=str(exc),
                )
                self._logger.error("Failure in flush: %s", exc)
                raise

    def _maybe_auto_flush(self) -> None:
        """Flush state when buffers exceed configured thresholds.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        now = time.monotonic()
        should_flush = False
        with self._lock:
            self._pending_flush_events += 1
            if self._last_auto_flush <= 0.0:
                self._last_auto_flush = now
            if self._pending_flush_events >= self._auto_flush_threshold:
                should_flush = True
            elif (now - self._last_auto_flush) >= self._auto_flush_interval:
                should_flush = True
        if not should_flush:
            return
        try:
            self.flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _maybe_auto_flush: %s", exc)

    def close(self) -> None:
        """Flush and close any open storage handles.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PersistentStateManager.close",
            extra={"event": "persistent_state_close"},
        )
        try:
            self.flush()
        except Exception:  # pragma: no cover - best effort
            self._logger.debug("persistent_state_flush_failure", exc_info=True)
        if self._db is not None:
            try:
                self._db.close()
            except Exception:  # pragma: no cover - best effort
                self._logger.debug("persistent_state_close_failure", exc_info=True)
            self._db = None
        for sig, previous in self._signal_handlers.items():
            try:
                signal.signal(sig, previous)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - best effort
                continue
        self._signal_handlers.clear()

    # ------------------------------------------------------------------
    def _initialise_sqlite(self) -> None:
        if sqlite3 is None:
            raise RuntimeError("SQLite unavailable")
        db_path = self._base_path / "persistent_state.db"
        self._db = PersistentStateDB(db_path)

    def _load_existing_state(self) -> None:
        self._logger.debug(
            "Entered PersistentStateManager._load_existing_state",
            extra={"event": "persistent_state_load"},
        )
        try:
            if self._mode == "sqlite" and self._db is not None:
                self._load_sqlite_state()
            else:
                self._load_json_state()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_existing_state: %s", exc)
            raise

    def _record_write_success(self) -> None:
        """Record metadata for the most recent successful write.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_write_wall_clock = now
            self._health_status = "ok"
            self._last_error = None

    def _record_flush_success(self) -> None:
        """Record metadata for the most recent successful flush.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_flush_wall_clock = now
            self._health_status = "ok"
            self._last_error = None

    def _record_failure(self, error: Exception) -> None:
        """Record failure metadata following an exception.

        Args:
            error: Exception raised during persistence activity.

        Returns:
            None.

        Raises:
            None.
        """

        with self._lock:
            self._last_error = str(error)
            self._health_status = "error"

    def _load_sqlite_state(self) -> None:
        if self._db is None:
            raise RuntimeError("SQLite database unavailable")
        try:
            trades = self._db.load_trades()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state trades: %s", exc)
            trades = []
        try:
            fills_payloads = self._db.load_fills()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state fills: %s", exc)
            fills_payloads = []
        try:
            positions_payloads = self._db.load_positions()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state positions: %s", exc)
            positions_payloads = []
        try:
            shadow_payload = self._db.load_shadow_equity()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state shadow: %s", exc)
            shadow_payload = {}
        try:
            orders_payloads = self._db.load_open_orders()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state orders: %s", exc)
            orders_payloads = []
        try:
            brackets_payloads = self._db.load_brackets()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _load_sqlite_state brackets: %s", exc)
            brackets_payloads = []

        positions: dict[str, PositionDict] = {}
        for payload in positions_payloads:
            if not isinstance(payload, Mapping):
                continue
            symbol = str(payload.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            entry = dict(payload)
            entry["symbol"] = symbol
            positions[symbol] = entry

        orders: dict[str, OrderDict] = {}
        for payload in orders_payloads:
            if not isinstance(payload, Mapping):
                continue
            order_id = str(payload.get("order_id", "")).strip()
            if not order_id:
                continue
            orders[order_id] = dict(payload)

        brackets: dict[str, BracketDict] = {}
        for payload in brackets_payloads:
            if not isinstance(payload, Mapping):
                continue
            entry_id = str(payload.get("entry_id", "")).strip()
            if not entry_id:
                continue
            brackets[entry_id] = dict(payload)

        shadow: dict[str, object] = {}
        if isinstance(shadow_payload, Mapping):
            equity_value = shadow_payload.get("equity")
            if equity_value is not None:
                shadow["equity"] = _coerce_float(equity_value)
            positions_source: Mapping[str, object] | None = None
            raw_positions = shadow_payload.get("positions")
            if isinstance(raw_positions, Mapping):
                positions_source = raw_positions
            else:
                payload_raw = shadow_payload.get("payload")
                if isinstance(payload_raw, Mapping):
                    nested_positions = payload_raw.get("positions")
                    if isinstance(nested_positions, Mapping):
                        positions_source = nested_positions
                    else:
                        positions_source = payload_raw
            if isinstance(positions_source, Mapping):
                shadow["positions"] = {
                    str(key): _coerce_float(value)
                    for key, value in positions_source.items()
                }

        fills: list[FillDict] = []
        for payload in fills_payloads:
            if isinstance(payload, Mapping):
                fills.append(dict(payload))

        with self._lock:
            self._trade_cache = [dict(entry) for entry in trades]
            self._fill_cache = fills
            self._position_cache = positions
            self._orders_cache = orders
            self._bracket_cache = brackets
            self._shadow_state = shadow

    def _load_json_state(self) -> None:
        trades: list[TradeDict] = []
        fills: list[FillDict] = []
        positions: dict[str, PositionDict] = {}
        orders: dict[str, OrderDict] = {}
        brackets: dict[str, BracketDict] = {}
        shadow: dict[str, object] = {}
        if self._trades_path.exists():
            try:
                trades_data = json.loads(self._trades_path.read_text(encoding="utf-8"))
                if isinstance(trades_data, list):
                    trades = [
                        dict(entry)
                        for entry in trades_data
                        if isinstance(entry, Mapping)
                    ]
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_trades_json_decode_failed", exc_info=True
                )
        if self._fills_path.exists():
            try:
                fills_data = json.loads(self._fills_path.read_text(encoding="utf-8"))
                if isinstance(fills_data, list):
                    fills = [
                        dict(entry)
                        for entry in fills_data
                        if isinstance(entry, Mapping)
                    ]
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_fills_json_decode_failed", exc_info=True
                )
        if self._positions_path.exists():
            try:
                positions_data = json.loads(
                    self._positions_path.read_text(encoding="utf-8")
                )
                if isinstance(positions_data, Mapping):
                    for key, value in positions_data.items():
                        if isinstance(value, Mapping):
                            payload = dict(value)
                            payload["symbol"] = str(key).strip().upper()
                            positions[payload["symbol"]] = payload
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_positions_json_decode_failed", exc_info=True
                )
        if self._orders_path.exists():
            try:
                orders_data = json.loads(self._orders_path.read_text(encoding="utf-8"))
                if isinstance(orders_data, Mapping):
                    for key, value in orders_data.items():
                        if isinstance(value, Mapping):
                            orders[str(key)] = dict(value)
                elif isinstance(orders_data, list):
                    for entry in orders_data:
                        if isinstance(entry, Mapping):
                            order_id = str(entry.get("order_id", "")).strip()
                            if order_id:
                                orders[order_id] = dict(entry)
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_orders_json_decode_failed", exc_info=True
                )
        if self._brackets_path.exists():
            try:
                brackets_data = json.loads(
                    self._brackets_path.read_text(encoding="utf-8")
                )
                if isinstance(brackets_data, Mapping):
                    for key, value in brackets_data.items():
                        if isinstance(value, Mapping):
                            brackets[str(key)] = dict(value)
                elif isinstance(brackets_data, list):
                    for entry in brackets_data:
                        if isinstance(entry, Mapping):
                            entry_id = str(entry.get("entry_id", "")).strip()
                            if entry_id:
                                brackets[entry_id] = dict(entry)
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_brackets_json_decode_failed", exc_info=True
                )
        if self._shadow_path.exists():
            try:
                shadow_data = json.loads(self._shadow_path.read_text(encoding="utf-8"))
                if isinstance(shadow_data, Mapping):
                    shadow = dict(shadow_data)
            except (OSError, json.JSONDecodeError):
                self._logger.debug(
                    "persistent_state_shadow_json_decode_failed", exc_info=True
                )
        with self._lock:
            self._trade_cache = trades
            self._fill_cache = fills
            self._position_cache = positions
            self._orders_cache = orders
            self._bracket_cache = brackets
            self._shadow_state = shadow

    def _write_trade_sqlite(self, payload: TradeDict) -> None:
        if self._db is None:
            raise RuntimeError("SQLite database unavailable")
        self._db.save_trade(payload)

    def _write_fill_sqlite(self, payload: FillDict) -> None:
        if self._db is None:
            raise RuntimeError("SQLite database unavailable")
        self._db.save_fill(payload)

    def _write_position_sqlite(self, symbol: str, payload: PositionDict | None) -> None:
        if self._db is None:
            raise RuntimeError("SQLite database unavailable")
        if payload is None:
            self._db.save_position({"symbol": symbol, "quantity": 0})
            return
        self._db.save_position(payload)

    def _write_shadow_sqlite(self, payload: Mapping[str, object]) -> None:
        if self._db is None:
            raise RuntimeError("SQLite database unavailable")
        equity_value = _coerce_float(payload.get("equity", 0.0))
        extra_payload: Mapping[str, object] | None = None
        positions_raw = payload.get("positions")
        if isinstance(positions_raw, Mapping):
            extra_payload = positions_raw
        self._db.save_shadow_equity(equity_value, extra_payload)

    def _ensure_base_dir(self) -> None:
        """Ensure base directory exists before writing (safety net)."""
        try:
            if not self._base_path.exists():
                self._base_path.mkdir(parents=True, exist_ok=True)
                self._logger.debug(f"Created missing base directory: {self._base_path}")
        except Exception as exc:
            self._logger.error(f"Directory check failed: {exc}", exc_info=True)
            raise

    def _write_trades_json(self) -> None:
        self._ensure_base_dir()  # ✅ ADD
        data = json.dumps(self._trade_cache, indent=2, sort_keys=True)
        self._trades_path.write_text(data, encoding="utf-8")

    def _write_fills_json(self) -> None:
        self._ensure_base_dir()  # ✅ ADD
        data = json.dumps(self._fill_cache, indent=2, sort_keys=True)
        self._fills_path.write_text(data, encoding="utf-8")

    def _write_positions_json(self) -> None:
        """Write positions cache to JSON with Enum handling.
        
        ✅ PRODUCTION FIX: Added sanitization for Enum, datetime, Decimal types.
        """
        self._ensure_base_dir()
        
        def _sanitize(obj):
            """Recursively convert non-JSON-serializable types."""
            from enum import Enum
            from datetime import datetime, date
            from decimal import Decimal
            
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
        
        sanitized = _sanitize(self._position_cache)
        data = json.dumps(sanitized, indent=2, sort_keys=True, default=str)
        self._positions_path.write_text(data, encoding="utf-8")

    def _write_orders_json(self) -> None:
        """Write orders cache to JSON with Enum handling.
        
        ✅ PRODUCTION FIX: Added sanitization for Enum, datetime, Decimal types.
        """
        self._ensure_base_dir()
        
        def _sanitize(obj):
            """Recursively convert non-JSON-serializable types."""
            from enum import Enum
            from datetime import datetime, date
            from decimal import Decimal
            
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
        
        sanitized = _sanitize(self._orders_cache)
        data = json.dumps(sanitized, indent=2, sort_keys=True, default=str)
        self._orders_path.write_text(data, encoding="utf-8")

    def _write_brackets_json(self) -> None:
        """Write brackets cache to JSON with Enum handling.
        
        ✅ PRODUCTION FIX: Added sanitization for consistency.
        """
        self._ensure_base_dir()
        
        def _sanitize(obj):
            from enum import Enum
            from datetime import datetime, date
            from decimal import Decimal
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
            return obj
        
        sanitized = _sanitize(self._bracket_cache)
        data = json.dumps(sanitized, indent=2, sort_keys=True, default=str)
        self._brackets_path.write_text(data, encoding="utf-8")

    def _write_shadow_json(self) -> None:
        """Write shadow state to JSON with type handling.
        
        ✅ PRODUCTION FIX: Added sanitization for consistency.
        """
        self._ensure_base_dir()
        
        def _sanitize(obj):
            from enum import Enum
            from datetime import datetime, date
            from decimal import Decimal
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
            return obj
        
        sanitized = _sanitize(self._shadow_state)
        data = json.dumps(sanitized, indent=2, sort_keys=True, default=str)
        self._shadow_path.write_text(data, encoding="utf-8")

    def _register_signal(self, sig: int) -> None:
        try:
            previous = cast(SignalHandler, signal.getsignal(sig))
        except Exception:  # pragma: no cover - platform differences
            return

        def _handler(signum: int, frame: FrameType | None) -> None:
            self._logger.info(
                "Flushing persistent state on signal",
                extra={"event": "persistent_state_signal", "signal": signum},
            )
            try:
                self.flush()
            except Exception:  # pragma: no cover - best effort
                self._logger.debug(
                    "persistent_state_signal_flush_failed", exc_info=True
                )
            if callable(previous):
                previous(signum, frame)

        try:
            signal.signal(sig, _handler)
            self._signal_handlers[sig] = previous
        except Exception:  # pragma: no cover - unsupported signals
            return

    # ------------------------------------------------------------------
    def telemetry(self) -> dict[str, object]:
        """Return telemetry payload describing persistence health.

        Args:
            None.

        Returns:
            Dictionary containing mode, health, queue depth, and timestamps.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PersistentStateManager.telemetry",
            extra={"event": "persistent_state_telemetry"},
        )
        with self._lock:
            last_write = self._last_write_wall_clock
            last_flush = self._last_flush_wall_clock
            pending = int(self._pending_flush_events)
            payload: dict[str, object] = {
                "mode": self._mode,
                "health": self._health_status,
                "pending_events": pending,
                "pending_queue_depth": pending,
                "auto_flush_threshold": int(self._auto_flush_threshold),
                "auto_flush_interval": float(self._auto_flush_interval),
            }
            if last_write is not None:
                payload["last_write_at"] = last_write.isoformat()
                payload["last_write_epoch"] = last_write.timestamp()
            if last_flush is not None:
                payload["last_flush_at"] = last_flush.isoformat()
                payload["last_flush_epoch"] = last_flush.timestamp()
            if self._last_error:
                payload["last_error"] = self._last_error
        return payload

    # ------------------------------------------------------------------
    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        with suppress(Exception):
            self.close()


__all__ = [
    "PersistentStateDB",
    "PersistentStateManager",
    "TradeDict",
    "PositionDict",
    "OrderDict",
    "BracketDict",
    "ShadowPositions",
]
