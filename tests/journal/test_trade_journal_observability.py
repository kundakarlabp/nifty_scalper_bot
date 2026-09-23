from __future__ import annotations

import json
import sqlite3

import pytest

from nifty_scalper_bot.journal.trade_journal import TradeJournal


def test_normalize_event_adds_canonical_correlation_fields(tmp_path) -> None:
    journal = TradeJournal(str(tmp_path / "journal.db"))
    event = journal._normalize_event(
        {
            "event_type": "ORDER_SUBMITTED",
            "timestamp": 1.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 75,
            "price": 142.5,
            "order_id": "broker-1",
            "meta": {
                "trade_id": "TRD_sig-1",
                "signal_id": "sig-1",
                "trace_id": "trace-1",
                "strategy": "VWAP",
            },
        }
    )

    assert event["event_name"] == "order.acknowledged"
    assert event["trade_id"] == "TRD_sig-1"
    assert event["signal_id"] == "sig-1"
    assert event["trace_id"] == "trace-1"
    assert event["strategy"] == "VWAP"


def test_trade_decision_uses_trace_as_signal_correlation(tmp_path) -> None:
    journal = TradeJournal(str(tmp_path / "journal.db"))
    event = journal._normalize_event(
        {
            "event_type": "TRADE_DECISION",
            "meta": {
                "trace_id": "trace-1",
                "final_reason": "candidate_not_ready",
            },
        }
    )

    assert event["event_name"] == "signal.evaluated"
    assert event["signal_id"] == "trace-1"
    assert event["reason_code"] == "candidate_not_ready"


@pytest.mark.parametrize(
    ("event_type", "event_name"),
    [
        ("BRACKET_ARMED", "bracket.armed"),
        ("TRAIL_UPDATED", "trail.updated"),
        ("EXIT_TRIGGERED", "exit.triggered"),
        ("EXIT_SUBMITTED", "exit.submitted"),
        ("EXIT_FILLED", "exit.filled"),
        ("BRACKET_CLOSED", "trade.closed"),
    ],
)
def test_explicit_post_entry_event_name_is_preserved(
    tmp_path, event_type: str, event_name: str
) -> None:
    journal = TradeJournal(str(tmp_path / "journal.db"))

    event = journal._normalize_event(
        {
            "event_type": event_type,
            "meta": {
                "event_name": event_name,
                "trade_id": "TRD_sig-1",
                "signal_id": "sig-1",
                "trace_id": "trace-1",
            },
        }
    )

    assert event["event_name"] == event_name
    assert event["trade_id"] == "TRD_sig-1"
    assert event["signal_id"] == "sig-1"
    assert event["trace_id"] == "trace-1"


def test_existing_trade_events_table_is_migrated_in_place(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
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

    journal = TradeJournal(str(db_path))
    conn = journal._ensure_connection(None)
    try:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(trade_events)")
        }
    finally:
        conn.close()

    assert {
        "event_name",
        "trade_id",
        "signal_id",
        "trace_id",
        "strategy",
        "reason_code",
        "build_sha",
    } <= columns



def test_trade_lifecycle_materializes_one_authoritative_ledger_row(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    correlation = {
        "trade_id": "TRD_sig-1",
        "signal_id": "sig-1",
        "trace_id": "trace-1",
        "strategy": "VWAP",
    }
    events = [
        {
            "event_type": "ORDER_SUBMITTED",
            "timestamp": 10.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 130,
            "price": 99.0,
            "order_id": "ENTRY-1",
            "meta": correlation,
        },
        {
            "event_type": "ORDER_FILL_CONFIRMED",
            "timestamp": 11.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 130,
            "price": 100.0,
            "order_id": "ENTRY-1",
            "meta": correlation,
        },
        {
            "event_type": "BRACKET_ARMED",
            "timestamp": 12.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 130,
            "price": 100.0,
            "order_id": "ENTRY-1",
            "meta": {
                **correlation,
                "entry_order_id": "ENTRY-1",
                "filled_qty": 130,
                "fill_price": 100.0,
                "stop_price": 90.0,
                "target_price": 120.0,
            },
        },
        {
            "event_type": "TRAIL_UPDATED",
            "timestamp": 13.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {**correlation, "new_sl": 101.0},
        },
        {
            "event_type": "EXIT_TRIGGERED",
            "timestamp": 14.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {**correlation, "reason": "HARD_TP_BREACH"},
        },
        {
            "event_type": "EXIT_SUBMITTED",
            "timestamp": 15.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "order_id": "EXIT-1",
            "meta": {
                **correlation,
                "exit_order_id": "EXIT-1",
                "reason": "HARD_TP_BREACH",
            },
        },
        {
            "event_type": "EXIT_FILLED",
            "timestamp": 16.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "order_id": "EXIT-1",
            "price": 110.0,
            "meta": {
                **correlation,
                "exit_order_id": "EXIT-1",
                "filled_qty": 130,
                "fill_price": 110.0,
            },
        },
        {
            "event_type": "BRACKET_CLOSED",
            "timestamp": 17.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {
                **correlation,
                "entry_order_id": "ENTRY-1",
                "exit_order_id": "EXIT-1",
                "completed_trade": {
                    **correlation,
                    "bracket_id": "ENTRY-1",
                    "symbol": "NFO:NIFTYCE",
                    "side": "BUY",
                    "quantity": 130,
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "gross_pnl": 1300.0,
                    "estimated_costs": {
                        "brokerage": 40.0,
                        "total": 88.5,
                    },
                    "net_pnl": 1211.5,
                    "final_stop_price": 101.0,
                    "r_multiple": 0.932,
                    "mfe_r": 1.2,
                    "mae_r": 0.4,
                    "holding_seconds": 6.0,
                    "exit_reason": "HARD_TP_BREACH",
                    "close_source": "broker_fill",
                    "ledger_complete": True,
                    "execution_quality": {
                        "entry_slippage_points": 0.2,
                        "exit_slippage_points": 0.1,
                    },
                },
            },
        },
    ]

    normalized = [journal._normalize_event(event) for event in events]
    conn = journal._flush_batch(normalized, None)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        read_conn.row_factory = sqlite3.Row
        rows = read_conn.execute("SELECT * FROM trade_ledger").fetchall()

    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == "TRD_sig-1"
    assert row["signal_id"] == "sig-1"
    assert row["trace_id"] == "trace-1"
    assert row["strategy"] == "VWAP"
    assert row["state"] == "CLOSED"
    assert row["entry_order_id"] == "ENTRY-1"
    assert row["exit_order_id"] == "EXIT-1"
    assert row["quantity"] == 130
    assert row["entry_price"] == 100.0
    assert row["stop_price"] == 101.0
    assert row["target_price"] == 120.0
    assert row["exit_price"] == 110.0
    assert row["gross_pnl"] == 1300.0
    assert row["estimated_costs"] == 88.5
    assert row["net_pnl"] == 1211.5
    assert row["ledger_complete"] == 1
    assert row["entry_submitted_at"] == 10.0
    assert row["entry_filled_at"] == 11.0
    assert row["bracket_armed_at"] == 12.0
    assert row["exit_triggered_at"] == 14.0
    assert row["exit_submitted_at"] == 15.0
    assert row["exit_filled_at"] == 16.0
    assert row["closed_at"] == 17.0
    assert row["last_event_name"] == "trade.closed"
    assert json.loads(row["costs_json"])["total"] == 88.5
    assert json.loads(row["outcome_json"])["gross_pnl"] == 1300.0


def test_closed_trade_is_not_regressed_by_stale_lifecycle_event(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    close_event = journal._normalize_event(
        {
            "event_type": "BRACKET_CLOSED",
            "timestamp": 20.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {
                "trade_id": "trade-1",
                "completed_trade": {
                    "quantity": 65,
                    "entry_price": 100.0,
                    "exit_price": 109.0,
                    "gross_pnl": 585.0,
                    "estimated_costs": {"total": 70.0},
                    "net_pnl": 515.0,
                    "final_stop_price": 105.0,
                    "ledger_complete": True,
                },
            },
        }
    )
    stale_trail = journal._normalize_event(
        {
            "event_type": "TRAIL_UPDATED",
            "timestamp": 10.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {
                "trade_id": "trade-1",
                "new_sl": 99.0,
            },
        }
    )

    conn = journal._flush_batch([close_event], None)
    conn = journal._flush_batch([stale_trail], conn)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        row = read_conn.execute(
            """
            SELECT state, stop_price, gross_pnl, net_pnl,
                   updated_at, last_event_name
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("trade-1",),
        ).fetchone()

    assert row == ("CLOSED", 105.0, 585.0, 515.0, 20.0, "trade.closed")


def test_event_without_trade_id_does_not_create_trade_ledger_row(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    event = journal._normalize_event(
        {
            "event_type": "TRADE_DECISION",
            "timestamp": 1.0,
            "meta": {"trace_id": "trace-only"},
        }
    )

    conn = journal._flush_batch([event], None)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        event_count = read_conn.execute(
            "SELECT COUNT(*) FROM trade_events"
        ).fetchone()[0]
        ledger_count = read_conn.execute(
            "SELECT COUNT(*) FROM trade_ledger"
        ).fetchone()[0]

    assert event_count == 1
    assert ledger_count == 0
