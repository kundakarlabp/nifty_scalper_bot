"""Asynchronous trade journaling for low-latency execution paths."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

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


_TRADE_LEDGER_STATES = {
    "signal.evaluated": ("SIGNAL_EVALUATED", 10),
    "candidate.blocked": ("BLOCKED", 100),
    "order.submit_attempt": ("ENTRY_SUBMITTING", 20),
    "order.acknowledged": ("ENTRY_SUBMITTED", 30),
    "entry.filled": ("ENTRY_FILLED", 40),
    "bracket.armed": ("OPEN", 50),
    "trail.updated": ("OPEN", 50),
    "exit.triggered": ("EXIT_TRIGGERED", 60),
    "exit.submitted": ("EXIT_SUBMITTED", 70),
    "exit.filled": ("EXIT_FILLED", 80),
    "broker.rejected": ("REJECTED", 100),
    "trade.closed": ("CLOSED", 100),
}


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _optional_positive_int(value: Any) -> int | None:
    try:
        resolved = int(float(value))
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _mapping_json(value: Any) -> str | None:
    if not isinstance(value, Mapping) or not value:
        return None
    return json.dumps(dict(value), separators=(",", ":"), default=str)


def _first_text(*values: Any) -> str | None:
    for value in values:
        resolved = _optional_text(value)
        if resolved is not None:
            return resolved
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        resolved = _optional_float(value)
        if resolved is not None:
            return resolved
    return None


_TRADE_LEDGER_UPSERT_SQL = """
INSERT INTO trade_ledger (
    trade_id, signal_id, trace_id, strategy, symbol, side,
    state, state_rank, entry_order_id, exit_order_id, quantity,
    entry_price, stop_price, target_price, exit_price,
    gross_pnl, estimated_costs, net_pnl, r_multiple, mfe_r, mae_r,
    holding_seconds, exit_reason, close_source, ledger_complete,
    decision_at, entry_submitted_at, entry_filled_at, bracket_armed_at,
    exit_triggered_at, exit_submitted_at, exit_filled_at, closed_at,
    created_at, updated_at, last_event_name, build_sha,
    costs_json, execution_quality_json, outcome_json
) VALUES (
    :trade_id, :signal_id, :trace_id, :strategy, :symbol, :side,
    :state, :state_rank, :entry_order_id, :exit_order_id, :quantity,
    :entry_price, :stop_price, :target_price, :exit_price,
    :gross_pnl, :estimated_costs, :net_pnl, :r_multiple, :mfe_r, :mae_r,
    :holding_seconds, :exit_reason, :close_source, :ledger_complete,
    :decision_at, :entry_submitted_at, :entry_filled_at, :bracket_armed_at,
    :exit_triggered_at, :exit_submitted_at, :exit_filled_at, :closed_at,
    :created_at, :updated_at, :last_event_name, :build_sha,
    :costs_json, :execution_quality_json, :outcome_json
)
ON CONFLICT(trade_id) DO UPDATE SET
    signal_id = COALESCE(trade_ledger.signal_id, excluded.signal_id),
    trace_id = COALESCE(trade_ledger.trace_id, excluded.trace_id),
    strategy = COALESCE(trade_ledger.strategy, excluded.strategy),
    symbol = COALESCE(trade_ledger.symbol, excluded.symbol),
    side = COALESCE(trade_ledger.side, excluded.side),
    state = CASE
        WHEN excluded.state_rank > trade_ledger.state_rank
          OR (
              excluded.state_rank = trade_ledger.state_rank
              AND excluded.updated_at >= trade_ledger.updated_at
          )
        THEN excluded.state
        ELSE trade_ledger.state
    END,
    state_rank = MAX(trade_ledger.state_rank, excluded.state_rank),
    entry_order_id = COALESCE(trade_ledger.entry_order_id, excluded.entry_order_id),
    exit_order_id = CASE
        WHEN excluded.exit_order_id IS NOT NULL
          AND (
              trade_ledger.exit_order_id IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.exit_order_id
        ELSE trade_ledger.exit_order_id
    END,
    quantity = CASE
        WHEN excluded.quantity IS NOT NULL
          AND (
              trade_ledger.quantity IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.quantity
        ELSE trade_ledger.quantity
    END,
    entry_price = CASE
        WHEN excluded.entry_price IS NOT NULL
          AND (
              trade_ledger.entry_price IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.entry_price
        ELSE trade_ledger.entry_price
    END,
    stop_price = CASE
        WHEN excluded.stop_price IS NOT NULL
          AND (
              trade_ledger.stop_price IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.stop_price
        ELSE trade_ledger.stop_price
    END,
    target_price = CASE
        WHEN excluded.target_price IS NOT NULL
          AND (
              trade_ledger.target_price IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.target_price
        ELSE trade_ledger.target_price
    END,
    exit_price = CASE
        WHEN excluded.exit_price IS NOT NULL
          AND (
              trade_ledger.exit_price IS NULL
              OR excluded.state_rank > trade_ledger.state_rank
              OR (
                  excluded.state_rank = trade_ledger.state_rank
                  AND excluded.updated_at >= trade_ledger.updated_at
              )
          )
        THEN excluded.exit_price
        ELSE trade_ledger.exit_price
    END,
    gross_pnl = COALESCE(excluded.gross_pnl, trade_ledger.gross_pnl),
    estimated_costs = COALESCE(excluded.estimated_costs, trade_ledger.estimated_costs),
    net_pnl = COALESCE(excluded.net_pnl, trade_ledger.net_pnl),
    r_multiple = COALESCE(excluded.r_multiple, trade_ledger.r_multiple),
    mfe_r = COALESCE(excluded.mfe_r, trade_ledger.mfe_r),
    mae_r = COALESCE(excluded.mae_r, trade_ledger.mae_r),
    holding_seconds = COALESCE(excluded.holding_seconds, trade_ledger.holding_seconds),
    exit_reason = COALESCE(excluded.exit_reason, trade_ledger.exit_reason),
    close_source = COALESCE(excluded.close_source, trade_ledger.close_source),
    ledger_complete = COALESCE(excluded.ledger_complete, trade_ledger.ledger_complete),
    decision_at = COALESCE(trade_ledger.decision_at, excluded.decision_at),
    entry_submitted_at = COALESCE(
        trade_ledger.entry_submitted_at, excluded.entry_submitted_at
    ),
    entry_filled_at = COALESCE(trade_ledger.entry_filled_at, excluded.entry_filled_at),
    bracket_armed_at = COALESCE(
        trade_ledger.bracket_armed_at, excluded.bracket_armed_at
    ),
    exit_triggered_at = COALESCE(
        trade_ledger.exit_triggered_at, excluded.exit_triggered_at
    ),
    exit_submitted_at = COALESCE(
        trade_ledger.exit_submitted_at, excluded.exit_submitted_at
    ),
    exit_filled_at = COALESCE(trade_ledger.exit_filled_at, excluded.exit_filled_at),
    closed_at = COALESCE(trade_ledger.closed_at, excluded.closed_at),
    created_at = MIN(trade_ledger.created_at, excluded.created_at),
    updated_at = MAX(trade_ledger.updated_at, excluded.updated_at),
    last_event_name = CASE
        WHEN excluded.updated_at >= trade_ledger.updated_at
        THEN excluded.last_event_name
        ELSE trade_ledger.last_event_name
    END,
    build_sha = CASE
        WHEN excluded.build_sha IS NOT NULL
          AND (
              trade_ledger.build_sha IS NULL
              OR excluded.updated_at >= trade_ledger.updated_at
          )
        THEN excluded.build_sha
        ELSE trade_ledger.build_sha
    END,
    costs_json = COALESCE(excluded.costs_json, trade_ledger.costs_json),
    execution_quality_json = COALESCE(
        excluded.execution_quality_json, trade_ledger.execution_quality_json
    ),
    outcome_json = COALESCE(excluded.outcome_json, trade_ledger.outcome_json)
"""


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_ledger (
                trade_id TEXT PRIMARY KEY,
                signal_id TEXT,
                trace_id TEXT,
                strategy TEXT,
                symbol TEXT,
                side TEXT,
                state TEXT NOT NULL,
                state_rank INTEGER NOT NULL DEFAULT 0,
                entry_order_id TEXT,
                exit_order_id TEXT,
                quantity INTEGER,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                exit_price REAL,
                gross_pnl REAL,
                estimated_costs REAL,
                net_pnl REAL,
                r_multiple REAL,
                mfe_r REAL,
                mae_r REAL,
                holding_seconds REAL,
                exit_reason TEXT,
                close_source TEXT,
                ledger_complete INTEGER,
                decision_at REAL,
                entry_submitted_at REAL,
                entry_filled_at REAL,
                bracket_armed_at REAL,
                exit_triggered_at REAL,
                exit_submitted_at REAL,
                exit_filled_at REAL,
                closed_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_event_name TEXT NOT NULL,
                build_sha TEXT,
                costs_json TEXT,
                execution_quality_json TEXT,
                outcome_json TEXT
            )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_ledger_signal_id "
            "ON trade_ledger(signal_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_ledger_state_closed "
            "ON trade_ledger(state, closed_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_ledger_updated_at "
            "ON trade_ledger(updated_at)"
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
        ledger_rows = []

        for event in batch:
            meta_json = json.dumps(
                event.get("meta", {}), separators=(",", ":"), default=str
            )
            event_json = json.dumps(event, separators=(",", ":"), default=str)
            ledger_row = self._trade_ledger_row(event)
            if ledger_row is not None:
                ledger_rows.append(ledger_row)

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
                if ledger_rows:
                    conn.executemany(_TRADE_LEDGER_UPSERT_SQL, ledger_rows)
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

    def _trade_ledger_row(
        self,
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Build one read-optimized row from the canonical lifecycle event."""
        trade_id = _optional_text(event.get("trade_id"))
        if trade_id is None:
            return None

        meta = event.get("meta")
        meta = dict(meta) if isinstance(meta, Mapping) else {}
        outcome_value = meta.get("completed_trade")
        outcome = dict(outcome_value) if isinstance(outcome_value, Mapping) else {}
        costs_value = outcome.get("estimated_costs")
        costs = dict(costs_value) if isinstance(costs_value, Mapping) else {}
        quality_value = outcome.get("execution_quality")
        quality = dict(quality_value) if isinstance(quality_value, Mapping) else {}

        event_name = _optional_text(event.get("event_name")) or "unknown"
        state, state_rank = _TRADE_LEDGER_STATES.get(
            event_name,
            ("OBSERVED", 0),
        )
        timestamp = _optional_float(event.get("timestamp")) or time.time()
        order_id = _optional_text(event.get("order_id"))

        entry_order_id = _first_text(meta.get("entry_order_id"))
        if entry_order_id is None and event_name in {
            "order.acknowledged",
            "entry.filled",
        }:
            entry_order_id = order_id

        exit_order_id = _first_text(meta.get("exit_order_id"))
        if exit_order_id is None and event_name in {"exit.submitted", "exit.filled"}:
            exit_order_id = order_id

        quantity = _optional_positive_int(outcome.get("quantity"))
        if quantity is None and event_name in {
            "order.acknowledged",
            "entry.filled",
            "bracket.armed",
        }:
            quantity = _optional_positive_int(
                meta.get("filled_qty") or event.get("qty")
            )

        entry_price = _optional_float(outcome.get("entry_price"))
        if entry_price is None and event_name == "entry.filled":
            entry_price = _optional_float(event.get("price"))
        if entry_price is None and event_name == "bracket.armed":
            entry_price = _first_float(meta.get("fill_price"), event.get("price"))

        stop_price = _first_float(
            outcome.get("final_stop_price"),
            meta.get("new_sl"),
            meta.get("stop_price"),
        )
        target_price = _optional_float(meta.get("target_price"))

        exit_price = _optional_float(outcome.get("exit_price"))
        if exit_price is None and event_name == "exit.filled":
            exit_price = _first_float(
                meta.get("fill_price"),
                meta.get("exit_price"),
                event.get("price"),
            )

        ledger_complete = None
        if "ledger_complete" in outcome:
            ledger_complete = int(bool(outcome.get("ledger_complete")))
        elif "ledger_complete" in meta:
            ledger_complete = int(bool(meta.get("ledger_complete")))

        signal_id = _first_text(event.get("signal_id"), outcome.get("signal_id"))
        trace_id = _first_text(event.get("trace_id"), outcome.get("trace_id"))
        strategy = _first_text(
            event.get("strategy"),
            outcome.get("strategy"),
            outcome.get("strategy_name"),
        )

        return {
            "trade_id": trade_id,
            "signal_id": signal_id,
            "trace_id": trace_id,
            "strategy": strategy,
            "symbol": _first_text(event.get("symbol"), outcome.get("symbol")),
            "side": _first_text(event.get("side"), outcome.get("side")),
            "state": state,
            "state_rank": state_rank,
            "entry_order_id": entry_order_id,
            "exit_order_id": exit_order_id,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_price": exit_price,
            "gross_pnl": _first_float(outcome.get("gross_pnl"), meta.get("pnl")),
            "estimated_costs": _first_float(costs.get("total")),
            "net_pnl": _first_float(outcome.get("net_pnl"), meta.get("net_pnl")),
            "r_multiple": _optional_float(outcome.get("r_multiple")),
            "mfe_r": _optional_float(outcome.get("mfe_r")),
            "mae_r": _optional_float(outcome.get("mae_r")),
            "holding_seconds": _optional_float(outcome.get("holding_seconds")),
            "exit_reason": _first_text(
                outcome.get("exit_reason"),
                meta.get("reason"),
            ),
            "close_source": _first_text(
                outcome.get("close_source"),
                meta.get("close_source"),
            ),
            "ledger_complete": ledger_complete,
            "decision_at": timestamp if event_name == "signal.evaluated" else None,
            "entry_submitted_at": (
                timestamp if event_name == "order.acknowledged" else None
            ),
            "entry_filled_at": timestamp if event_name == "entry.filled" else None,
            "bracket_armed_at": timestamp if event_name == "bracket.armed" else None,
            "exit_triggered_at": timestamp if event_name == "exit.triggered" else None,
            "exit_submitted_at": timestamp if event_name == "exit.submitted" else None,
            "exit_filled_at": timestamp if event_name == "exit.filled" else None,
            "closed_at": timestamp if event_name == "trade.closed" else None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_event_name": event_name,
            "build_sha": _optional_text(event.get("build_sha")),
            "costs_json": _mapping_json(costs),
            "execution_quality_json": _mapping_json(quality),
            "outcome_json": _mapping_json(outcome),
        }

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
