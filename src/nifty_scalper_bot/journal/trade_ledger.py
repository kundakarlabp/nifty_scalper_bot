"""Transactional read model for canonical trade lifecycle events.

This module is deliberately not an execution or accounting authority. It derives
one query-friendly row per trade_id from events already accepted by TradeJournal
and writes that row in the same SQLite transaction.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any

_TRADE_STATES = {
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


def _monotonic_merge_sql(column: str) -> str:
    """Prefer a populated higher-rank value or a non-older same-rank value."""
    return (
        f"CASE WHEN excluded.{column} IS NOT NULL AND ("
        f"trade_ledger.{column} IS NULL "
        "OR excluded.state_rank > trade_ledger.state_rank "
        "OR (excluded.state_rank = trade_ledger.state_rank "
        "AND excluded.updated_at >= trade_ledger.updated_at)) "
        f"THEN excluded.{column} ELSE trade_ledger.{column} END"
    )


_UPSERT_SQL = f"""
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
    gross_pnl = {_monotonic_merge_sql("gross_pnl")},
    estimated_costs = {_monotonic_merge_sql("estimated_costs")},
    net_pnl = {_monotonic_merge_sql("net_pnl")},
    r_multiple = {_monotonic_merge_sql("r_multiple")},
    mfe_r = {_monotonic_merge_sql("mfe_r")},
    mae_r = {_monotonic_merge_sql("mae_r")},
    holding_seconds = {_monotonic_merge_sql("holding_seconds")},
    exit_reason = {_monotonic_merge_sql("exit_reason")},
    close_source = {_monotonic_merge_sql("close_source")},
    ledger_complete = CASE
        WHEN trade_ledger.ledger_complete = 1 THEN 1
        ELSE {_monotonic_merge_sql("ledger_complete")}
    END,
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
    costs_json = {_monotonic_merge_sql("costs_json")},
    execution_quality_json = {_monotonic_merge_sql("execution_quality_json")},
    outcome_json = {_monotonic_merge_sql("outcome_json")}
"""


def ensure_trade_ledger_schema(conn: sqlite3.Connection) -> None:
    """Create the derived ledger table and read-path indexes."""
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


def materialize_trade_events(
    conn: sqlite3.Connection,
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Upsert derived trade rows from already-normalized journal events."""
    rows = [_build_trade_row(event) for event in events]
    materialized = [row for row in rows if row is not None]
    if materialized:
        conn.executemany(_UPSERT_SQL, materialized)


def _build_trade_row(event: Mapping[str, Any]) -> dict[str, Any] | None:
    trade_id = _text(event.get("trade_id"))
    if trade_id is None:
        return None

    meta_value = event.get("meta")
    meta = dict(meta_value) if isinstance(meta_value, Mapping) else {}
    outcome_value = meta.get("completed_trade")
    outcome = dict(outcome_value) if isinstance(outcome_value, Mapping) else {}
    costs_value = outcome.get("estimated_costs")
    costs = dict(costs_value) if isinstance(costs_value, Mapping) else {}
    quality_value = outcome.get("execution_quality")
    quality = dict(quality_value) if isinstance(quality_value, Mapping) else {}

    event_name = _text(event.get("event_name")) or "unknown"
    state, state_rank = _TRADE_STATES.get(event_name, ("OBSERVED", 0))
    timestamp = _number(event.get("timestamp")) or time.time()
    order_id = _text(event.get("order_id"))

    entry_order_id = _first_text(meta.get("entry_order_id"))
    if entry_order_id is None and event_name in {
        "order.acknowledged",
        "entry.filled",
    }:
        entry_order_id = order_id

    exit_order_id = _first_text(meta.get("exit_order_id"))
    if exit_order_id is None and event_name in {"exit.submitted", "exit.filled"}:
        exit_order_id = order_id

    quantity = _positive_int(outcome.get("quantity"))
    if quantity is None and event_name in {
        "order.acknowledged",
        "entry.filled",
        "bracket.armed",
    }:
        quantity = _positive_int(meta.get("filled_qty") or event.get("qty"))

    entry_price = _number(outcome.get("entry_price"))
    if entry_price is None and event_name == "entry.filled":
        entry_price = _number(event.get("price"))
    if entry_price is None and event_name == "bracket.armed":
        entry_price = _first_number(meta.get("fill_price"), event.get("price"))

    stop_price = _first_number(
        outcome.get("final_stop_price"),
        meta.get("new_sl"),
        meta.get("stop_price"),
    )
    target_price = _number(meta.get("target_price"))

    exit_price = _number(outcome.get("exit_price"))
    if exit_price is None and event_name == "exit.filled":
        exit_price = _first_number(
            meta.get("fill_price"),
            meta.get("exit_price"),
            event.get("price"),
        )

    ledger_complete = None
    if "ledger_complete" in outcome:
        ledger_complete = int(bool(outcome.get("ledger_complete")))
    elif "ledger_complete" in meta:
        ledger_complete = int(bool(meta.get("ledger_complete")))

    return {
        "trade_id": trade_id,
        "signal_id": _first_text(event.get("signal_id"), outcome.get("signal_id")),
        "trace_id": _first_text(event.get("trace_id"), outcome.get("trace_id")),
        "strategy": _first_text(
            event.get("strategy"),
            outcome.get("strategy"),
            outcome.get("strategy_name"),
        ),
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
        "gross_pnl": _first_number(outcome.get("gross_pnl"), meta.get("pnl")),
        "estimated_costs": _first_number(costs.get("total")),
        "net_pnl": _first_number(outcome.get("net_pnl"), meta.get("net_pnl")),
        "r_multiple": _number(outcome.get("r_multiple")),
        "mfe_r": _number(outcome.get("mfe_r")),
        "mae_r": _number(outcome.get("mae_r")),
        "holding_seconds": _number(outcome.get("holding_seconds")),
        "exit_reason": _first_text(outcome.get("exit_reason"), meta.get("reason")),
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
        "build_sha": _text(event.get("build_sha")),
        "costs_json": _mapping_json(costs),
        "execution_quality_json": _mapping_json(quality),
        "outcome_json": _mapping_json(outcome),
    }


def _text(value: Any) -> str | None:
    resolved = str(value or "").strip()
    return resolved or None


def _number(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _positive_int(value: Any) -> int | None:
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
        resolved = _text(value)
        if resolved is not None:
            return resolved
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        resolved = _number(value)
        if resolved is not None:
            return resolved
    return None


__all__ = ["ensure_trade_ledger_schema", "materialize_trade_events"]
