"""Asynchronous trade journaling for low-latency execution paths."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nifty_scalper_bot.journal.trade_ledger import (
    ensure_trade_ledger_schema,
    materialize_trade_events,
)

LOGGER = logging.getLogger(__name__)

_TRADE_LEDGER_BACKFILL_MIGRATION = "trade_ledger_historical_backfill_v1"

_CANONICAL_EVENT_NAMES = {
    "TRADE_DECISION": "signal.evaluated",
    "ORDER_BLOCKED_DUPLICATE": "candidate.blocked",
    "ORDER_SUBMIT_ATTEMPT": "order.submit_attempt",
    "ORDER_SUBMITTED": "order.acknowledged",
    "ORDER_FILL_CONFIRMED": "entry.filled",
    "ORDER_REJECTED_FATAL": "broker.rejected",
    "BRACKET_GUARD_REGISTERED": "bracket.armed",
    "BRACKET_ARMED": "bracket.armed",
    "TRAIL_UPDATED": "trail.updated",
    "EXIT_TRIGGERED": "exit.triggered",
    "EXIT_SUBMITTED": "exit.submitted",
    "EXIT_FILLED": "exit.filled",
    "BRACKET_CLOSED": "trade.closed",
}


def _canonical_event_name(event_type: str) -> str:
    mapped = _CANONICAL_EVENT_NAMES.get(event_type)
    if mapped:
        return mapped
    return event_type.strip().lower().replace("_", ".") or "unknown"


class TradeJournal:
    """Queue-driven async SQLite trade event journal."""

    _SENTINEL: dict[str, Any] = {"event_type": "__STOP__"}

    def __init__(
        self,
        db_path: str,
        *,
        max_queue_size: int = 10_000,
        batch_size: int = 50,
        flush_interval_s: float = 0.1,
        max_retries: int = 5,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._fallback_path = self._db_path.with_name(
            f"{self._db_path.stem}_fallback.jsonl"
        )

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_s = max(0.01, float(flush_interval_s))
        self._max_retries = max(1, int(max_retries))

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

        self._dropped_events = 0

        # ✅ crash-safe shutdown
        atexit.register(self.stop)

    # -------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="trade-journal-worker",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        self._stop_event.set()

        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        # 🔴 FINAL SAFETY DRAIN
        try:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item and item.get("event_type") != "__STOP__":
                    self._write_fallback(item)
        except Exception:
            pass

        self._started = False

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def log_event(self, event: dict[str, Any]) -> None:
        payload = self._normalize_event(event)

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped_events += 1

            # 🔴 NEVER LOSE EVENT
            try:
                self._write_fallback(payload)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "fallback_write_failed dropped=%d err=%s",
                    self._dropped_events,
                    exc,
                    exc_info=exc,
                )

            if self._dropped_events % 100 == 1:
                LOGGER.warning(
                    "queue_full fallback_used dropped=%d",
                    self._dropped_events,
                )

    def get_stats(self) -> dict[str, int]:
        return {
            "queue_size": self._queue.qsize(),
            "dropped_events": self._dropped_events,
        }

    @property
    def db_path(self) -> Path:
        """Return the canonical SQLite journal path for read-only consumers."""
        return self._db_path

    # -------------------------------------------------------
    # Internal
    # -------------------------------------------------------
    def _normalize_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        meta = event.get("meta")
        if not isinstance(meta, Mapping):
            meta = {}

        event_type = str(event.get("event_type") or "UNKNOWN")
        meta_dict = dict(meta)
        trace_id = str(meta_dict.get("trace_id") or "") or None
        signal_id = str(meta_dict.get("signal_id") or trace_id or "") or None
        return {
            "event_type": event_type,
            "event_name": str(
                meta_dict.get("event_name") or _canonical_event_name(event_type)
            ),
            "timestamp": float(event.get("timestamp") or time.time()),
            "symbol": str(event.get("symbol") or ""),
            "side": str(event.get("side") or ""),
            "qty": int(event.get("qty") or 0),
            "price": float(event.get("price") or 0.0),
            "order_id": str(event.get("order_id")) if event.get("order_id") else None,
            "trade_id": str(meta_dict.get("trade_id") or "") or None,
            "signal_id": signal_id,
            "trace_id": trace_id,
            "strategy": str(meta_dict.get("strategy") or "") or None,
            "reason_code": str(
                meta_dict.get("reason_code")
                or meta_dict.get("block_reason")
                or meta_dict.get("final_reason")
                or ""
            )
            or None,
            "build_sha": str(
                meta_dict.get("build_sha")
                or os.getenv("GIT_SHA")
                or os.getenv("BUILD_SHA")
                or ""
            )
            or None,
            "meta": meta_dict,
        }

    # -------------------------------------------------------
    # Worker
    # -------------------------------------------------------
    def _worker_loop(self) -> None:
        conn: sqlite3.Connection | None = None
        batch: list[dict[str, Any]] = []

        try:
            conn = self._ensure_connection(None)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "TradeJournal initial connection failed: %s",
                exc,
                exc_info=exc,
            )

        deadline = time.monotonic() + self._flush_interval_s

        while True:
            try:
                timeout = max(0.0, deadline - time.monotonic())
                item = self._queue.get(timeout=timeout)

                if item is self._SENTINEL or item.get("event_type") == "__STOP__":
                    if batch:
                        conn = self._flush_batch(batch, conn)
                        batch.clear()
                    self._drain_remaining(batch, conn)
                    return

                batch.append(item)

                if len(batch) >= self._batch_size:
                    conn = self._flush_batch(batch, conn)
                    batch.clear()
                    deadline = time.monotonic() + self._flush_interval_s

            except queue.Empty:
                if batch:
                    conn = self._flush_batch(batch, conn)
                    batch.clear()
                deadline = time.monotonic() + self._flush_interval_s

            except Exception as exc:  # noqa: BLE001
                LOGGER.critical(
                    "TradeJournal worker failure: %s",
                    exc,
                    exc_info=exc,
                )
                self._write_fallback(
                    {
                        "event_type": "WORKER_ERROR",
                        "timestamp": time.time(),
                        "symbol": "",
                        "side": "",
                        "qty": 0,
                        "price": 0.0,
                        "meta": {"error": str(exc)},
                    }
                )

    def _drain_remaining(
        self,
        batch: list[dict[str, Any]],
        conn: sqlite3.Connection | None,
    ) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is self._SENTINEL:
                continue

            batch.append(item)

            if len(batch) >= self._batch_size:
                conn = self._flush_batch(batch, conn)
                batch.clear()

        if batch:
            self._flush_batch(batch, conn)

    # -------------------------------------------------------
    # DB Layer
    # -------------------------------------------------------
    def _backfill_trade_ledger(self, conn: sqlite3.Connection) -> None:
        """Replay stored journal events into the derived ledger exactly once."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal_migrations (
                name TEXT PRIMARY KEY,
                completed_at REAL NOT NULL
            )
            """)
        if conn.execute(
            "SELECT 1 FROM trade_journal_migrations WHERE name = ?",
            (_TRADE_LEDGER_BACKFILL_MIGRATION,),
        ).fetchone():
            return

        materialized = 0
        skipped = 0
        last_id = 0
        try:
            conn.execute("BEGIN")
            while True:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, event_type, event_name, symbol, side,
                           qty, price, order_id, trade_id, signal_id, trace_id,
                           strategy, reason_code, build_sha, meta_json, event_json
                    FROM trade_events
                    WHERE id > ?
                    ORDER BY id
                    LIMIT 500
                    """,
                    (last_id,),
                ).fetchall()
                if not rows:
                    break

                events: list[dict[str, Any]] = []
                for row in rows:
                    last_id = int(row[0])
                    try:
                        raw = json.loads(str(row[16]))
                        if not isinstance(raw, Mapping):
                            raise TypeError("event_json is not an object")
                        event = dict(raw)
                        stored_meta = json.loads(str(row[15] or "{}"))
                        if not isinstance(stored_meta, Mapping):
                            stored_meta = {}
                        event_meta = event.get("meta")
                        if not isinstance(event_meta, Mapping):
                            event_meta = {}
                        meta = dict(stored_meta)
                        meta.update(event_meta)
                        for key, value in zip(
                            (
                                "event_name",
                                "trade_id",
                                "signal_id",
                                "trace_id",
                                "strategy",
                                "reason_code",
                                "build_sha",
                            ),
                            row[3:4] + row[9:15],
                            strict=True,
                        ):
                            if value not in (None, ""):
                                meta.setdefault(key, value)
                        event["meta"] = meta
                        for key, value in zip(
                            (
                                "timestamp",
                                "event_type",
                                "symbol",
                                "side",
                                "qty",
                                "price",
                                "order_id",
                            ),
                            row[1:3] + row[4:9],
                            strict=True,
                        ):
                            if value is not None:
                                event.setdefault(key, value)
                        events.append(self._normalize_event(event))
                    except Exception:  # noqa: BLE001
                        skipped += 1

                if events:
                    materialize_trade_events(conn, events)
                    materialized += len(events)

            conn.execute(
                """
                INSERT INTO trade_journal_migrations (name, completed_at)
                VALUES (?, ?)
                """,
                (_TRADE_LEDGER_BACKFILL_MIGRATION, time.time()),
            )
            conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            LOGGER.error(
                "Trade ledger historical backfill failed: %s",
                exc,
                exc_info=exc,
            )
            return

        LOGGER.info(
            "TRADE_LEDGER_BACKFILL_COMPLETE events=%d skipped=%d",
            materialized,
            skipped,
        )

    def _ensure_connection(
        self,
        conn: sqlite3.Connection | None,
    ) -> sqlite3.Connection:
        if conn is not None:
            return conn

        conn = sqlite3.connect(
            str(self._db_path),
            timeout=60.0,
            check_same_thread=False,
            isolation_level=None,  # 🔴 manual transactions
        )

        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                event_name TEXT,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                order_id TEXT,
                trade_id TEXT,
                signal_id TEXT,
                trace_id TEXT,
                strategy TEXT,
                reason_code TEXT,
                build_sha TEXT,
                meta_json TEXT,
                event_json TEXT NOT NULL
            )
            """)
        existing = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(trade_events)")
        }
        extra_columns = {
            "event_name": "TEXT",
            "trade_id": "TEXT",
            "signal_id": "TEXT",
            "trace_id": "TEXT",
            "strategy": "TEXT",
            "reason_code": "TEXT",
            "build_sha": "TEXT",
        }
        for column, sql_type in extra_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE trade_events ADD COLUMN {column} {sql_type}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_events_trade_id "
            "ON trade_events(trade_id, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_events_signal_id "
            "ON trade_events(signal_id, timestamp)"
        )
        ensure_trade_ledger_schema(conn)
        self._backfill_trade_ledger(conn)

        return conn

    def _flush_batch(
        self,
        batch: list[dict[str, Any]],
        conn: sqlite3.Connection | None,
    ) -> sqlite3.Connection | None:
        if not batch:
            return conn

        rows = []

        for event in batch:
            meta_json = json.dumps(
                event.get("meta", {}), separators=(",", ":"), default=str
            )
            event_json = json.dumps(event, separators=(",", ":"), default=str)

            rows.append(
                (
                    event["timestamp"],
                    event["event_type"],
                    event["event_name"],
                    event["symbol"],
                    event["side"],
                    event["qty"],
                    event["price"],
                    event["order_id"],
                    event["trade_id"],
                    event["signal_id"],
                    event["trace_id"],
                    event["strategy"],
                    event["reason_code"],
                    event["build_sha"],
                    meta_json,
                    event_json,
                )
            )

        for attempt in range(1, self._max_retries + 1):
            try:
                conn = self._ensure_connection(conn)

                conn.execute("BEGIN")
                conn.executemany(
                    """
                    INSERT INTO trade_events (
                        timestamp, event_type, event_name, symbol, side,
                        qty, price, order_id, trade_id, signal_id, trace_id,
                        strategy, reason_code, build_sha, meta_json, event_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                materialize_trade_events(conn, batch)
                conn.execute("COMMIT")

                return conn

            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < self._max_retries:
                    time.sleep(0.02 * attempt)
                    continue

                LOGGER.error("SQLite write failed: %s", exc)
                self._write_fallback_many(batch)

                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return None

            except Exception as exc:  # noqa: BLE001
                LOGGER.error("flush_batch failure: %s", exc, exc_info=exc)
                self._write_fallback_many(batch)

                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return None

        self._write_fallback_many(batch)
        return conn

    # -------------------------------------------------------
    # Fallback
    # -------------------------------------------------------
    def _write_fallback_many(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self._write_fallback(event)

    def _write_fallback(self, event: Mapping[str, Any]) -> None:
        try:
            with self._fallback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(event), default=str))
                f.write("\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("fallback_write_failure: %s", exc, exc_info=exc)
