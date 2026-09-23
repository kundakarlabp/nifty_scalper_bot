from __future__ import annotations

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
def test_post_entry_events_have_canonical_names(
    tmp_path, event_type: str, event_name: str
) -> None:
    journal = TradeJournal(str(tmp_path / "journal.db"))

    event = journal._normalize_event(
        {
            "event_type": event_type,
            "meta": {
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
