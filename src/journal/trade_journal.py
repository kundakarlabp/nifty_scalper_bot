"""Asynchronous trade journaling for low-latency execution paths."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


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

    # -------------------------------------------------------
    # Internal
    # -------------------------------------------------------
    def _normalize_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        meta = event.get("meta")
        if not isinstance(meta, Mapping):
            meta = {}

        return {
            "event_type": str(event.get("event_type") or "UNKNOWN"),
            "timestamp": float(event.get("timestamp") or time.time()),
            "symbol": str(event.get("symbol") or ""),
            "side": str(event.get("side") or ""),
            "qty": int(event.get("qty") or 0),
            "price": float(event.get("price") or 0.0),
            "order_id": str(event.get("order_id")) if event.get("order_id") else None,
            "meta": dict(meta),
        }

    # -------------------------------------------------------
    # Worker
    # -------------------------------------------------------
    def _worker_loop(self) -> None:
        conn: sqlite3.Connection | None = None
        batch: list[dict[str, Any]] = []

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
    def _ensure_connection(
        self, conn: sqlite3.Connection | None
    ) -> sqlite3.Connection:
        if conn is not None:
            return conn

        conn = sqlite3.connect(
            str(self._db_path),
            timeout=2.0,
            check_same_thread=False,
            isolation_level=None,  # 🔴 manual transactions
        )

        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                order_id TEXT,
                meta_json TEXT,
                event_json TEXT NOT NULL
            )
            """
        )

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
            meta_json = json.dumps(event.get("meta", {}), separators=(",", ":"), default=str)
            event_json = json.dumps(event, separators=(",", ":"), default=str)

            rows.append(
                (
                    event["timestamp"],
                    event["event_type"],
                    event["symbol"],
                    event["side"],
                    event["qty"],
                    event["price"],
                    event["order_id"],
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
                        timestamp, event_type, symbol, side,
                        qty, price, order_id, meta_json, event_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
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
