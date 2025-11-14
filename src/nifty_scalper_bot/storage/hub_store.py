"""Durable backing store for :class:`~nifty_scalper_bot.data.data_hub.DataHub`."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger

log = get_logger(__name__)


class HubStore:
    """Simple SQLite-backed snapshot and WAL store for hub state."""

    def __init__(self, path: str | Path = "data/hub.db") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wal (
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    entry_key TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot (
                    kind TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_ts REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def append_wal(self, kind: str, entry_key: str, payload: MappingPayload) -> None:
        """Append an entry to the write-ahead log."""

        encoded = json.dumps(payload)
        with self._lock:
            self._conn.execute(
                "INSERT INTO wal(ts, kind, entry_key, payload) VALUES(?,?,?,?)",
                (time.time(), kind, entry_key, encoded),
            )
            self._conn.commit()

    def load_wal(self) -> list[WalEntry]:
        """Return WAL entries ordered by timestamp."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, entry_key, payload FROM wal ORDER BY ts"
            ).fetchall()
        entries: list[WalEntry] = []
        for row in rows:
            ts, kind, entry_key, payload = row
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                log.warning("hub_store_invalid_wal_json", extra={"kind": kind})
                continue
            entries.append(
                {
                    "ts": float(ts),
                    "kind": str(kind),
                    "key": str(entry_key),
                    "payload": decoded,
                }
            )
        return entries

    def save_snapshot(self, kind: str, payload: MappingPayload) -> None:
        """Persist the latest snapshot for *kind*."""

        encoded = json.dumps(payload)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO snapshot(kind, payload, updated_ts)
                VALUES(?, ?, ?)
                ON CONFLICT(kind) DO UPDATE SET
                    payload=excluded.payload,
                    updated_ts=excluded.updated_ts
                """,
                (kind, encoded, time.time()),
            )
            self._conn.commit()

    def load_snapshot(self, kind: str) -> MappingPayload | None:
        """Load the latest snapshot for *kind* if present."""

        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM snapshot WHERE kind=?", (kind,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            log.warning("hub_store_invalid_snapshot_json", extra={"kind": kind})
            return None

    def purge_wal(self, *, before_ts: float | None = None) -> None:
        """Delete WAL entries up to *before_ts* (or all if ``None``)."""

        with self._lock:
            if before_ts is None:
                self._conn.execute("DELETE FROM wal")
            else:
                self._conn.execute("DELETE FROM wal WHERE ts <= ?", (before_ts,))
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            self._conn.close()

    def __enter__(self) -> "HubStore":  # pragma: no cover - convenience only
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:  # pragma: no cover - convenience only
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


MappingPayload = Mapping[str, Any]
WalEntry = dict[str, Any]


__all__ = ["HubStore", "WalEntry"]
