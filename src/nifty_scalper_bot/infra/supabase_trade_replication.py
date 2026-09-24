"""Off-hot-path replication of canonical trade observability to Supabase.

SQLite remains the source of truth. This module owns no execution state and no
event queue; it periodically reads committed TradeJournal rows and advances one
local checkpoint only after the remote batch is acknowledged.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_REPLICATION_URL = (
    "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1/nifty-trade-ingest"
)
_DEFAULT_SOURCE = "lightsail"
_IST = ZoneInfo("Asia/Kolkata")
_REPLICATION_SINK = "supabase_trade_observability"

Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def replication_interval_seconds() -> float:
    try:
        return max(
            10.0,
            float(os.getenv("SUPABASE_TRADE_REPLICATION_INTERVAL_SECONDS", "30")),
        )
    except (TypeError, ValueError):
        return 30.0


def build_supabase_trade_replicator(
    db_path: str | Path,
) -> "SupabaseTradeReplicator | None":
    """Build the optional remote sink from runtime configuration."""
    if not _env_enabled("SUPABASE_TRADE_REPLICATION_ENABLED", default=False):
        return None
    endpoint = (
        os.getenv("SUPABASE_TRADE_REPLICATION_URL", DEFAULT_REPLICATION_URL).strip()
        or DEFAULT_REPLICATION_URL
    )
    source = os.getenv("SUPABASE_TRADE_REPLICATION_SOURCE", _DEFAULT_SOURCE).strip()
    try:
        batch_size = max(
            1,
            min(500, int(os.getenv("SUPABASE_TRADE_REPLICATION_BATCH_SIZE", "100"))),
        )
    except (TypeError, ValueError):
        batch_size = 100
    try:
        timeout_seconds = max(
            1.0,
            min(
                30.0,
                float(os.getenv("SUPABASE_TRADE_REPLICATION_TIMEOUT_SECONDS", "5")),
            ),
        )
    except (TypeError, ValueError):
        timeout_seconds = 5.0
    return SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url=endpoint,
        source=source or _DEFAULT_SOURCE,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )


class SupabaseTradeReplicator:
    """Replicate committed trade events and their materialized ledger rows."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        endpoint_url: str,
        source: str = _DEFAULT_SOURCE,
        batch_size: int = 100,
        timeout_seconds: float = 5.0,
        transport: Transport | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._endpoint_url = str(endpoint_url).strip()
        self._source = str(source).strip() or _DEFAULT_SOURCE
        self._batch_size = max(1, min(500, int(batch_size)))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._transport = transport or _post_json

    def replicate_once(self) -> dict[str, int]:
        """Replicate one bounded batch; checkpoint only after remote success."""
        if not self._db_path.exists():
            return {"events": 0, "ledger": 0, "checkpoint": 0}

        event_rows, ledger_rows, checkpoint = self._load_batch()
        if not event_rows:
            return {"events": 0, "ledger": 0, "checkpoint": checkpoint}

        events = [self._event_payload(row) for row in event_rows]
        ledger = [self._ledger_payload(row) for row in ledger_rows]
        response = self._transport(
            self._endpoint_url,
            {
                "source": self._source,
                "events": events,
                "ledger": ledger,
            },
            self._timeout_seconds,
        )
        if response.get("ok") is not True:
            detail = response.get("error") or response
            raise RuntimeError(f"Supabase trade replication rejected: {detail}")

        last_event_id = int(event_rows[-1]["id"])
        self._store_checkpoint(last_event_id)
        LOGGER.info(
            "SUPABASE_TRADE_REPLICATION_SUCCESS events=%d ledger=%d checkpoint=%d",
            len(events),
            len(ledger),
            last_event_id,
            extra={
                "event": "SUPABASE_TRADE_REPLICATION_SUCCESS",
                "events": len(events),
                "ledger": len(ledger),
                "checkpoint": last_event_id,
            },
        )
        return {
            "events": len(events),
            "ledger": len(ledger),
            "checkpoint": last_event_id,
        }

    def _load_batch(
        self,
    ) -> tuple[list[sqlite3.Row], list[sqlite3.Row], int]:
        with sqlite3.connect(str(self._db_path), timeout=1.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=1000")
            self._ensure_checkpoint_schema(conn)
            checkpoint = self._checkpoint(conn)
            try:
                events = list(
                    conn.execute(
                        """
                        SELECT *
                        FROM trade_events
                        WHERE id > ?
                        ORDER BY id
                        LIMIT ?
                        """,
                        (checkpoint, self._batch_size),
                    )
                )
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    return [], [], checkpoint
                raise

            trade_ids: set[str] = set()
            for row in events:
                trade_id = dict(row).get("trade_id")
                if trade_id not in (None, ""):
                    trade_ids.add(str(trade_id))
            sorted_trade_ids = sorted(trade_ids)
            if not sorted_trade_ids:
                return events, [], checkpoint

            placeholders = ",".join("?" for _ in sorted_trade_ids)
            try:
                ledger = list(
                    conn.execute(
                        f"""
                        SELECT *
                        FROM trade_ledger
                        WHERE trade_id IN ({placeholders})
                        ORDER BY trade_id
                        """,
                        sorted_trade_ids,
                    )
                )
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    ledger = []
                else:
                    raise
            return events, ledger, checkpoint

    @staticmethod
    def _ensure_checkpoint_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replication_state (
                sink TEXT PRIMARY KEY,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                last_success_at REAL
            )
            """)
        conn.execute(
            """
            INSERT OR IGNORE INTO replication_state (sink, last_event_id)
            VALUES (?, 0)
            """,
            (_REPLICATION_SINK,),
        )

    @staticmethod
    def _checkpoint(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT last_event_id FROM replication_state WHERE sink = ?",
            (_REPLICATION_SINK,),
        ).fetchone()
        return max(0, int(row[0] if row is not None else 0))

    def _store_checkpoint(self, event_id: int) -> None:
        with sqlite3.connect(str(self._db_path), timeout=1.0) as conn:
            conn.execute("PRAGMA busy_timeout=1000")
            self._ensure_checkpoint_schema(conn)
            conn.execute(
                """
                UPDATE replication_state
                SET last_event_id = MAX(last_event_id, ?),
                    last_success_at = strftime('%s', 'now')
                WHERE sink = ?
                """,
                (int(event_id), _REPLICATION_SINK),
            )

    def _event_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        timestamp = float(data["timestamp"])
        event_json = _json_object(data.get("event_json"))
        event_name = (
            _optional_text(data.get("event_name"))
            or _optional_text(event_json.get("event_name"))
            or _optional_text(data.get("event_type"))
            or _optional_text(event_json.get("event_type"))
            or "UNKNOWN"
        )
        return {
            "source": self._source,
            "source_event_id": int(data["id"]),
            "trading_date": _trading_date(timestamp),
            "event_at": _iso_utc(timestamp),
            "event_name": event_name,
            "trade_id": _optional_text(data.get("trade_id")),
            "signal_id": _optional_text(data.get("signal_id")),
            "trace_id": _optional_text(data.get("trace_id")),
            "symbol": _optional_text(data.get("symbol")),
            "side": _optional_text(data.get("side")),
            "qty": _optional_int(data.get("qty")),
            "price": _optional_float(data.get("price")),
            "order_id": _optional_text(data.get("order_id")),
            "strategy": _optional_text(data.get("strategy")),
            "reason_code": _optional_text(data.get("reason_code")),
            "build_sha": _optional_text(data.get("build_sha")),
            "payload": event_json,
        }

    @staticmethod
    def _ledger_payload(row: sqlite3.Row) -> dict[str, Any]:
        timestamp = _first_float(
            row["decision_at"],
            row["entry_filled_at"],
            row["closed_at"],
            row["created_at"],
            row["updated_at"],
        )
        payload = {
            "trade_id": str(row["trade_id"]),
            "trading_date": _trading_date(timestamp),
            "signal_id": _optional_text(row["signal_id"]),
            "trace_id": _optional_text(row["trace_id"]),
            "strategy": _optional_text(row["strategy"]),
            "symbol": str(row["symbol"] or ""),
            "side": _optional_text(row["side"]),
            "signal_at": _optional_iso(row["decision_at"]),
            "order_submitted_at": _optional_iso(row["entry_submitted_at"]),
            "entry_filled_at": _optional_iso(row["entry_filled_at"]),
            "entry_price": _optional_float(row["entry_price"]),
            "quantity": _optional_int(row["quantity"]),
            "initial_stop": _optional_float(row["initial_stop_price"]),
            "initial_target": _optional_float(row["initial_target_price"]),
            "exit_triggered_at": _optional_iso(row["exit_triggered_at"]),
            "exit_filled_at": _optional_iso(row["exit_filled_at"]),
            "exit_price": _optional_float(row["exit_price"]),
            "exit_reason": _optional_text(row["exit_reason"]),
            "gross_pnl": _optional_float(row["gross_pnl"]),
            "estimated_costs": _optional_float(row["estimated_costs"]),
            "net_pnl": _optional_float(row["net_pnl"]),
            "status": str(row["state"] or "OBSERVED"),
            "state_rank": _optional_int(row["state_rank"]),
            "entry_order_id": _optional_text(row["entry_order_id"]),
            "exit_order_id": _optional_text(row["exit_order_id"]),
            "current_stop": _optional_float(row["stop_price"]),
            "r_multiple": _optional_float(row["r_multiple"]),
            "mfe_r": _optional_float(row["mfe_r"]),
            "mae_r": _optional_float(row["mae_r"]),
            "holding_seconds": _optional_float(row["holding_seconds"]),
            "close_source": _optional_text(row["close_source"]),
            "ledger_complete": (
                bool(row["ledger_complete"])
                if row["ledger_complete"] is not None
                else None
            ),
            "closed_at": _optional_iso(row["closed_at"]),
            "build_sha": _optional_text(row["build_sha"]),
            "costs": _json_object(row["costs_json"]),
            "execution_quality": _json_object(row["execution_quality_json"]),
            "outcome": _json_object(row["outcome_json"]),
            "updated_at": _iso_utc(float(row["updated_at"])),
            "created_at": _iso_utc(float(row["created_at"])),
        }
        return {key: value for key, value in payload.items() if value is not None}


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    response = requests.post(
        url,
        json=dict(payload),
        timeout=timeout_seconds,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "nifty-scalper-trade-replicator/1",
            "X-Nifty-Source": _DEFAULT_SOURCE,
        },
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, Mapping):
        raise RuntimeError("Supabase trade replication returned a non-object response")
    return body


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def _optional_iso(value: Any) -> str | None:
    number = _optional_float(value)
    return _iso_utc(number) if number is not None else None


def _trading_date(timestamp: float | None) -> str:
    resolved = float(timestamp or 0.0)
    if resolved <= 0:
        resolved = datetime.now(timezone.utc).timestamp()
    return (
        datetime.fromtimestamp(resolved, timezone.utc)
        .astimezone(_IST)
        .date()
        .isoformat()
    )


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float:
    for value in values:
        resolved = _optional_float(value)
        if resolved is not None and resolved > 0:
            return resolved
    return 0.0


__all__ = [
    "DEFAULT_REPLICATION_URL",
    "SupabaseTradeReplicator",
    "build_supabase_trade_replicator",
    "replication_interval_seconds",
]
