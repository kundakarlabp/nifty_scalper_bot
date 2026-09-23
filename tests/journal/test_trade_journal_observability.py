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
        event_count = read_conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()[
            0
        ]
        ledger_count = read_conn.execute(
            "SELECT COUNT(*) FROM trade_ledger"
        ).fetchone()[0]

    assert event_count == 1
    assert ledger_count == 0


def _normalized_close_event(
    journal: TradeJournal,
    *,
    timestamp: float,
    marker: str,
    **overrides,
):
    completed_trade = {
        "quantity": 65,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "gross_pnl": 650.0,
        "estimated_costs": {"total": 75.0, "marker": marker},
        "net_pnl": 575.0,
        "final_stop_price": 105.0,
        "r_multiple": 1.2,
        "mfe_r": 1.6,
        "mae_r": 0.3,
        "holding_seconds": 90.0,
        "exit_reason": "TARGET",
        "close_source": "broker_fill",
        "ledger_complete": True,
        "execution_quality": {"marker": marker},
        "marker": marker,
    }
    completed_trade.update(overrides)
    if "estimated_costs" not in overrides:
        completed_trade["estimated_costs"] = {"total": 75.0, "marker": marker}
    return journal._normalize_event(
        {
            "event_type": "BRACKET_CLOSED",
            "timestamp": timestamp,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "meta": {
                "trade_id": "trade-replay-1",
                "completed_trade": completed_trade,
            },
        }
    )


def test_stale_closed_event_cannot_overwrite_newer_terminal_outcome(
    tmp_path,
) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    newest = _normalized_close_event(journal, timestamp=20.0, marker="new")
    stale = _normalized_close_event(
        journal,
        timestamp=10.0,
        marker="old",
        gross_pnl=-325.0,
        estimated_costs={"total": 120.0, "marker": "old"},
        net_pnl=-445.0,
        r_multiple=-0.8,
        mfe_r=0.2,
        mae_r=1.1,
        holding_seconds=30.0,
        exit_reason="STALE",
        close_source="replay",
        ledger_complete=False,
    )

    conn = journal._flush_batch([newest], None)
    conn = journal._flush_batch([stale], conn)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        row = read_conn.execute(
            """
            SELECT gross_pnl, estimated_costs, net_pnl, r_multiple,
                   mfe_r, mae_r, holding_seconds, exit_reason, close_source,
                   ledger_complete, costs_json, execution_quality_json,
                   outcome_json, updated_at, last_event_name
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("trade-replay-1",),
        ).fetchone()

    assert row is not None
    assert row[:10] == (
        650.0,
        75.0,
        575.0,
        1.2,
        1.6,
        0.3,
        90.0,
        "TARGET",
        "broker_fill",
        1,
    )
    assert json.loads(row[10])["marker"] == "new"
    assert json.loads(row[11])["marker"] == "new"
    assert json.loads(row[12])["marker"] == "new"
    assert row[13:] == (20.0, "trade.closed")


def test_duplicate_closed_event_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    close_event = _normalized_close_event(journal, timestamp=20.0, marker="same")

    conn = journal._flush_batch([close_event], None)
    conn = journal._flush_batch([close_event], conn)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        rows = read_conn.execute(
            """
            SELECT gross_pnl, net_pnl, ledger_complete, updated_at, last_event_name
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("trade-replay-1",),
        ).fetchall()

    assert rows == [(650.0, 575.0, 1, 20.0, "trade.closed")]


def test_newer_corrected_closed_event_replaces_older_terminal_outcome(
    tmp_path,
) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    original = _normalized_close_event(
        journal,
        timestamp=20.0,
        marker="original",
        ledger_complete=False,
    )
    corrected = _normalized_close_event(
        journal,
        timestamp=21.0,
        marker="corrected",
        gross_pnl=630.0,
        estimated_costs={"total": 80.0, "marker": "corrected"},
        net_pnl=550.0,
        r_multiple=1.1,
        mfe_r=1.5,
        mae_r=0.4,
        holding_seconds=92.0,
        exit_reason="TARGET_CORRECTED",
        ledger_complete=True,
    )

    conn = journal._flush_batch([original], None)
    conn = journal._flush_batch([corrected], conn)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        row = read_conn.execute(
            """
            SELECT gross_pnl, estimated_costs, net_pnl, r_multiple,
                   mfe_r, mae_r, holding_seconds, exit_reason,
                   ledger_complete, outcome_json, updated_at
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("trade-replay-1",),
        ).fetchone()

    assert row is not None
    assert row[:9] == (
        630.0,
        80.0,
        550.0,
        1.1,
        1.5,
        0.4,
        92.0,
        "TARGET_CORRECTED",
        1,
    )
    assert json.loads(row[9])["marker"] == "corrected"
    assert row[10] == 21.0


def test_ledger_complete_true_never_regresses_to_false(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(str(db_path))
    complete = _normalized_close_event(journal, timestamp=20.0, marker="complete")
    newer_incomplete = _normalized_close_event(
        journal,
        timestamp=21.0,
        marker="newer",
        gross_pnl=640.0,
        net_pnl=560.0,
        ledger_complete=False,
    )

    conn = journal._flush_batch([complete], None)
    conn = journal._flush_batch([newer_incomplete], conn)
    assert conn is not None
    conn.close()

    with sqlite3.connect(db_path) as read_conn:
        row = read_conn.execute(
            """
            SELECT ledger_complete, gross_pnl, net_pnl, updated_at
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("trade-replay-1",),
        ).fetchone()

    assert row == (1, 640.0, 560.0, 21.0)


def _create_legacy_trade_events_db(db_path, events) -> None:
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
        conn.executemany(
            """
            INSERT INTO trade_events (
                timestamp, event_type, symbol, side, qty, price,
                order_id, meta_json, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event["timestamp"],
                    event["event_type"],
                    event.get("symbol"),
                    event.get("side"),
                    event.get("qty"),
                    event.get("price"),
                    event.get("order_id"),
                    json.dumps(event.get("meta", {})),
                    json.dumps(event),
                )
                for event in events
            ],
        )


def _historical_trade_events(trade_id: str = "TRD_historical-1"):
    correlation = {
        "trade_id": trade_id,
        "signal_id": "historical-1",
        "trace_id": "trace-historical-1",
        "strategy": "VWAP",
    }
    return [
        {
            "event_type": "ORDER_FILL_CONFIRMED",
            "event_name": "entry.filled",
            "timestamp": 10.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 65,
            "price": 100.0,
            "order_id": "ENTRY-HIST-1",
            "trade_id": trade_id,
            "signal_id": correlation["signal_id"],
            "trace_id": correlation["trace_id"],
            "strategy": correlation["strategy"],
            "meta": correlation,
        },
        {
            "event_type": "BRACKET_CLOSED",
            "event_name": "trade.closed",
            "timestamp": 20.0,
            "symbol": "NFO:NIFTYCE",
            "side": "BUY",
            "qty": 65,
            "price": 110.0,
            "order_id": "ENTRY-HIST-1",
            "trade_id": trade_id,
            "signal_id": correlation["signal_id"],
            "trace_id": correlation["trace_id"],
            "strategy": correlation["strategy"],
            "meta": {
                **correlation,
                "completed_trade": {
                    "quantity": 65,
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "gross_pnl": 650.0,
                    "estimated_costs": {"total": 75.0},
                    "net_pnl": 575.0,
                    "exit_reason": "TARGET",
                    "close_source": "broker_fill",
                    "ledger_complete": True,
                },
            },
        },
    ]


def test_journal_start_backfills_historical_trade_events_without_new_event(
    tmp_path,
) -> None:
    db_path = tmp_path / "journal.db"
    _create_legacy_trade_events_db(db_path, _historical_trade_events())

    journal = TradeJournal(str(db_path))
    journal.start()
    journal.stop()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT state, entry_price, exit_price, gross_pnl,
                   estimated_costs, net_pnl, ledger_complete
            FROM trade_ledger
            WHERE trade_id = ?
            """,
            ("TRD_historical-1",),
        ).fetchone()
        marker = conn.execute(
            "SELECT value FROM trade_ledger_meta WHERE key = ?",
            ("historical_backfill_v1",),
        ).fetchone()

    assert row == ("CLOSED", 100.0, 110.0, 650.0, 75.0, 575.0, 1)
    assert marker == ("done",)


def test_historical_trade_backfill_runs_only_once(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    _create_legacy_trade_events_db(db_path, _historical_trade_events())

    journal = TradeJournal(str(db_path))
    journal.start()
    journal.stop()

    late_event = _historical_trade_events("TRD_should-not-rescan")[0]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO trade_events (
                timestamp, event_type, symbol, side, qty, price,
                order_id, meta_json, event_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                late_event["timestamp"],
                late_event["event_type"],
                late_event["symbol"],
                late_event["side"],
                late_event["qty"],
                late_event["price"],
                late_event["order_id"],
                json.dumps(late_event["meta"]),
                json.dumps(late_event),
            ),
        )

    journal.start()
    journal.stop()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM trade_ledger WHERE trade_id = ?",
            ("TRD_should-not-rescan",),
        ).fetchone()[0]

    assert count == 0
