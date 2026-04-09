"""Asynchronous trade journaling for low-latency execution paths."""

from __future__ import annotations

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

    _SENTINEL: dict[str, Any] = {'event_type': '__STOP__'}

    def __init__(
        self,
        db_path: str,
        *,
        max_queue_size: int = 10_000,
        batch_size: int = 50,
        flush_interval_s: float = 0.1,
        max_retries: int = 5,
    ) -> None:
        """Args: cfg. Returns: None. Raises: None."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path = self._db_path.with_name(
            f'{self._db_path.stem}_fallback.jsonl'
        )

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_s = max(0.01, float(flush_interval_s))
        self._max_retries = max(1, int(max_retries))

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._dropped_events = 0

    def start(self) -> None:
        """Args: none. Returns: None. Raises: None."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name='trade-journal-worker',
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Args: none. Returns: None. Raises: None."""
        if not self._started:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._started = False

    def log_event(self, event: dict[str, Any]) -> None:
        """Args: event. Returns: None. Raises: None."""
        payload = self._normalize_event(event)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events % 100 == 1:
                LOGGER.warning(
                    'trade_journal_queue_full dropped=%d', self._dropped_events
                )

    def _normalize_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Args: event. Returns: dict. Raises: None."""
        meta = event.get('meta')
        if not isinstance(meta, Mapping):
            meta = {}
        symbol = event.get('symbol')
        side = event.get('side')
        order_id = event.get('order_id')
        normalized: dict[str, Any] = {
            'event_type': str(event.get('event_type') or 'UNKNOWN'),
            'timestamp': float(event.get('timestamp') or time.time()),
            'symbol': str(symbol) if symbol is not None else '',
            'side': str(side) if side is not None else '',
            'qty': int(event.get('qty') or 0),
            'price': float(event.get('price') or 0.0),
            'order_id': str(order_id) if order_id is not None else None,
            'meta': dict(meta),
        }
        return normalized

    def _worker_loop(self) -> None:
        """Args: none. Returns: None. Raises: None."""
        conn: sqlite3.Connection | None = None
        batch: list[dict[str, Any]] = []
        deadline = time.monotonic() + self._flush_interval_s

        while True:
            try:
                timeout = max(0.0, deadline - time.monotonic())
                item = self._queue.get(timeout=timeout)
                if item is self._SENTINEL or item.get('event_type') == '__STOP__':
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
                LOGGER.error(
                    'Failure in TradeJournal._worker_loop: %s',
                    exc,
                    exc_info=exc,
                )
                self._write_fallback(
                    {
                        'event_type': 'JOURNAL_WORKER_ERROR',
                        'timestamp': time.time(),
                        'symbol': '',
                        'side': '',
                        'qty': 0,
                        'price': 0.0,
                        'meta': {'error': str(exc)},
                    }
                )

    def _drain_remaining(
        self,
        batch: list[dict[str, Any]],
        conn: sqlite3.Connection | None,
    ) -> None:
        """Args: batch, conn. Returns: None. Raises: None."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is self._SENTINEL or item.get('event_type') == '__STOP__':
                continue
            batch.append(item)
            if len(batch) >= self._batch_size:
                conn = self._flush_batch(batch, conn)
                batch.clear()
        if batch:
            self._flush_batch(batch, conn)
            batch.clear()

    def _ensure_connection(self, conn: sqlite3.Connection | None) -> sqlite3.Connection:
        """Args: conn. Returns: sqlite3.Connection. Raises: sqlite3.Error."""
        if conn is not None:
            return conn
        new_conn = sqlite3.connect(
            str(self._db_path), timeout=1.0, check_same_thread=False
        )
        new_conn.execute('PRAGMA journal_mode=WAL;')
        new_conn.execute('PRAGMA synchronous=NORMAL;')
        new_conn.execute(
            '''
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
            '''
        )
        return new_conn

    def _flush_batch(
        self,
        batch: list[dict[str, Any]],
        conn: sqlite3.Connection | None,
    ) -> sqlite3.Connection | None:
        """Args: batch, conn. Returns: conn. Raises: None."""
        if not batch:
            return conn

        rows = [
            (
                float(event.get('timestamp') or time.time()),
                str(event.get('event_type') or 'UNKNOWN'),
                str(event.get('symbol') or ''),
                str(event.get('side') or ''),
                int(event.get('qty') or 0),
                float(event.get('price') or 0.0),
                event.get('order_id'),
                json.dumps(event.get('meta', {}), separators=(',', ':'), default=str),
                json.dumps(event, separators=(',', ':'), default=str),
            )
            for event in batch
        ]

        for attempt in range(1, self._max_retries + 1):
            try:
                conn = self._ensure_connection(conn)
                conn.executemany(
                    '''
                    INSERT INTO trade_events (
                        timestamp,
                        event_type,
                        symbol,
                        side,
                        qty,
                        price,
                        order_id,
                        meta_json,
                        event_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    rows,
                )
                conn.commit()
                return conn
            except sqlite3.OperationalError as exc:
                locked = 'locked' in str(exc).lower() or 'busy' in str(exc).lower()
                if locked and attempt < self._max_retries:
                    time.sleep(0.02 * attempt)
                    continue
                LOGGER.error('TradeJournal SQLite write failed: %s', exc)
                self._write_fallback_many(batch)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return None
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    'Failure in TradeJournal._flush_batch: %s',
                    exc,
                    exc_info=exc,
                )
                self._write_fallback_many(batch)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return None

        self._write_fallback_many(batch)
        return conn

    def _write_fallback_many(self, events: list[dict[str, Any]]) -> None:
        """Args: events. Returns: None. Raises: None."""
        for event in events:
            self._write_fallback(event)

    def _write_fallback(self, event: Mapping[str, Any]) -> None:
        """Args: event. Returns: None. Raises: None."""
        try:
            with self._fallback_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(event), default=str))
                handle.write('\n')
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in TradeJournal._write_fallback: %s',
                exc,
                exc_info=exc,
            )
