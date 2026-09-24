from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

import pytest

from nifty_scalper_bot.infra.supabase_trade_replication import (
    SupabaseTradeReplicator,
    build_supabase_trade_replicator,
    replication_interval_seconds,
)
from nifty_scalper_bot.journal.trade_journal import TradeJournal


def _write_lifecycle(db_path: Any) -> None:
    journal = TradeJournal(str(db_path))
    correlation = {
        "trade_id": "trade-1",
        "signal_id": "signal-1",
        "trace_id": "trace-1",
        "strategy": "VWAP",
    }
    events = [
        {
            "event_type": "ORDER_FILL_CONFIRMED",
            "timestamp": 1_790_000_000.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 65,
            "price": 100.0,
            "order_id": "entry-1",
            "meta": correlation,
        },
        {
            "event_type": "BRACKET_ARMED",
            "timestamp": 1_790_000_001.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 65,
            "price": 100.0,
            "order_id": "entry-1",
            "meta": {
                **correlation,
                "entry_order_id": "entry-1",
                "filled_qty": 65,
                "fill_price": 100.0,
                "stop_price": 90.0,
                "target_price": 120.0,
            },
        },
        {
            "event_type": "TRAIL_UPDATED",
            "timestamp": 1_790_000_002.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {**correlation, "new_sl": 104.0},
        },
        {
            "event_type": "BRACKET_CLOSED",
            "timestamp": 1_790_000_003.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {
                **correlation,
                "entry_order_id": "entry-1",
                "exit_order_id": "exit-1",
                "completed_trade": {
                    **correlation,
                    "symbol": "NFO:NIFTYCE",
                    "side": "BUY",
                    "quantity": 65,
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "gross_pnl": 650.0,
                    "estimated_costs": {"total": 50.0},
                    "net_pnl": 600.0,
                    "final_stop_price": 104.0,
                    "r_multiple": 0.92,
                    "mfe_r": 1.3,
                    "mae_r": 0.2,
                    "holding_seconds": 3.0,
                    "exit_reason": "HARD_TP_BREACH",
                    "close_source": "broker_fill",
                    "ledger_complete": True,
                    "execution_quality": {"entry_slippage_points": 0.1},
                },
            },
        },
    ]
    normalized = [journal._normalize_event(event) for event in events]
    conn = journal._flush_batch(normalized, None)
    assert conn is not None
    conn.close()


def test_replication_is_idempotent_and_preserves_risk_semantics(tmp_path: Any) -> None:
    db_path = tmp_path / "trades.db"
    _write_lifecycle(db_path)
    calls: list[dict[str, Any]] = []

    def transport(
        _url: str,
        payload: Mapping[str, Any],
        _timeout: float,
    ) -> Mapping[str, Any]:
        calls.append(dict(payload))
        return {"ok": True}

    replicator = SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url="https://example.test/ingest",
        transport=transport,
    )

    first = replicator.replicate_once()
    second = replicator.replicate_once()

    assert first == {"events": 4, "ledger": 1, "checkpoint": 4}
    assert second == {"events": 0, "ledger": 0, "checkpoint": 4}
    assert len(calls) == 1

    payload = calls[0]
    assert payload["source"] == "lightsail"
    events = payload["events"]
    assert [event["source_event_id"] for event in events] == [1, 2, 3, 4]
    assert events[-1]["event_name"] == "trade.closed"

    ledger = payload["ledger"]
    assert len(ledger) == 1
    trade = ledger[0]
    assert trade["trade_id"] == "trade-1"
    assert trade["initial_stop"] == 90.0
    assert trade["initial_target"] == 120.0
    assert trade["current_stop"] == 104.0
    assert trade["status"] == "CLOSED"
    assert trade["gross_pnl"] == 650.0
    assert trade["estimated_costs"] == 50.0
    assert trade["net_pnl"] == 600.0
    assert trade["ledger_complete"] is True


def test_failed_remote_batch_does_not_advance_checkpoint(tmp_path: Any) -> None:
    db_path = tmp_path / "trades.db"
    _write_lifecycle(db_path)

    replicator = SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url="https://example.test/ingest",
        transport=lambda *_args: {"ok": False, "error": "temporary"},
    )
    with pytest.raises(RuntimeError, match="temporary"):
        replicator.replicate_once()

    with sqlite3.connect(db_path) as conn:
        progress = conn.execute("""
            SELECT last_event_id, last_ledger_updated_at, last_ledger_trade_id
            FROM replication_state
            WHERE sink = 'supabase_trade_observability'
            """).fetchone()
    assert progress == (0, 0.0, "")

    calls = 0

    def success(
        _url: str,
        _payload: Mapping[str, Any],
        _timeout: float,
    ) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    retry = SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url="https://example.test/ingest",
        transport=success,
    )
    assert retry.replicate_once()["checkpoint"] == 4
    assert calls == 1


def test_replication_accepts_legacy_trade_event_schema(tmp_path: Any) -> None:
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE trade_events (
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
            """)
        conn.execute(
            """
            INSERT INTO trade_events (
                timestamp, event_type, symbol, side, qty, price,
                order_id, meta_json, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1_790_000_000.0,
                "ORDER_FILL_CONFIRMED",
                "NFO:NIFTYCE",
                "BUY",
                65,
                100.0,
                "entry-legacy",
                '{"trade_id":"TRD_legacy-1","signal_id":"legacy-1",'
                '"trace_id":"trace-legacy-1","strategy":"VWAP"}',
                '{"event_type":"ORDER_FILL_CONFIRMED"}',
            ),
        )

    calls: list[dict[str, Any]] = []

    def transport(
        _url: str,
        payload: Mapping[str, Any],
        _timeout: float,
    ) -> Mapping[str, Any]:
        calls.append(dict(payload))
        return {"ok": True}

    replicator = SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url="https://example.test/ingest",
        transport=transport,
    )

    assert replicator.replicate_once() == {
        "events": 1,
        "ledger": 0,
        "checkpoint": 1,
    }
    assert len(calls) == 1
    event = calls[0]["events"][0]
    assert event["event_name"] == "ORDER_FILL_CONFIRMED"
    assert event["trade_id"] == "TRD_legacy-1"
    assert event["signal_id"] == "legacy-1"
    assert event["trace_id"] == "trace-legacy-1"
    assert event["strategy"] == "VWAP"

    with sqlite3.connect(db_path) as conn:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(trade_events)")
        }
    assert "event_name" not in columns
    assert "trade_id" not in columns


def test_ledger_backfill_runs_after_event_checkpoint_already_advanced(
    tmp_path: Any,
) -> None:
    db_path = tmp_path / "trades.db"
    _write_lifecycle(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE replication_state (
                sink TEXT PRIMARY KEY,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                last_success_at REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO replication_state (sink, last_event_id)
            VALUES ('supabase_trade_observability', 4)
            """
        )

    calls: list[dict[str, Any]] = []

    def transport(
        _url: str,
        payload: Mapping[str, Any],
        _timeout: float,
    ) -> Mapping[str, Any]:
        calls.append(dict(payload))
        return {"ok": True}

    replicator = SupabaseTradeReplicator(
        db_path=db_path,
        endpoint_url="https://example.test/ingest",
        transport=transport,
    )

    first = replicator.replicate_once()
    second = replicator.replicate_once()

    assert first == {"events": 0, "ledger": 1, "checkpoint": 4}
    assert second == {"events": 0, "ledger": 0, "checkpoint": 4}
    assert len(calls) == 1
    assert calls[0]["events"] == []
    assert calls[0]["ledger"][0]["trade_id"] == "trade-1"

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT last_event_id, last_ledger_updated_at, last_ledger_trade_id
            FROM replication_state
            WHERE sink = 'supabase_trade_observability'
            """
        ).fetchone()

    assert row == (4, 1_790_000_003.0, "trade-1")


def test_missing_database_is_a_noop(tmp_path: Any) -> None:
    called = False

    def transport(
        _url: str,
        _payload: Mapping[str, Any],
        _timeout: float,
    ) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {"ok": True}

    replicator = SupabaseTradeReplicator(
        db_path=tmp_path / "missing.db",
        endpoint_url="https://example.test/ingest",
        transport=transport,
    )

    assert replicator.replicate_once() == {
        "events": 0,
        "ledger": 0,
        "checkpoint": 0,
    }
    assert called is False


def test_replication_builder_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    db_path = tmp_path / "trades.db"
    monkeypatch.delenv("SUPABASE_TRADE_REPLICATION_ENABLED", raising=False)
    assert build_supabase_trade_replicator(db_path) is None

    monkeypatch.setenv("SUPABASE_TRADE_REPLICATION_ENABLED", "true")
    monkeypatch.setenv("SUPABASE_TRADE_REPLICATION_INTERVAL_SECONDS", "45")
    replicator = build_supabase_trade_replicator(db_path)

    assert replicator is not None
    assert replication_interval_seconds() == 45.0
