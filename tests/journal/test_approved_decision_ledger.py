import sqlite3

from nifty_scalper_bot.journal.trade_journal import TradeJournal
from nifty_scalper_bot.journal.trade_ledger import (
    _build_trade_row,
    ensure_trade_ledger_schema,
    materialize_trade_events,
)


def test_approved_decision_is_linked_and_keeps_decision_time() -> None:
    event = {
        "event_name": "candidate.approved",
        "timestamp": 1_790_232_000.0,
        "trade_id": "TRD_executed-signal",
        "signal_id": "executed-signal",
        "trace_id": "runner-trace",
        "symbol": "NFO:NIFTY26SEP23250CE",
        "meta": {"signal_score": 7.8},
    }

    row = _build_trade_row(event)

    assert row is not None
    assert row["trade_id"] == event["trade_id"]
    assert row["signal_id"] == event["signal_id"]
    assert row["decision_at"] == event["timestamp"]
    assert row["state"] == "SIGNAL_EVALUATED"
    assert _build_trade_row({**event, "trade_id": None}) is None


def test_approved_decision_after_fill_preserves_fill_and_score(tmp_path) -> None:
    journal = TradeJournal(str(tmp_path / "trades.db"))
    filled = journal._normalize_event(
        {
            "event_type": "ORDER_FILL_CONFIRMED",
            "timestamp": 1_790_232_001.0,
            "symbol": "NFO:NIFTY26SEP23250CE",
            "qty": 65,
            "price": 114.7,
            "meta": {"signal_id": "executed-signal", "trade_id": "TRD_executed-signal"},
        }
    )
    approved = journal._normalize_event(
        {
            "event_type": "TRADE_DECISION",
            "timestamp": 1_790_232_002.0,
            "symbol": "NFO:NIFTY26SEP23250CE",
            "meta": {
                "event_name": "candidate.approved",
                "signal_id": "executed-signal",
                "trade_id": "TRD_executed-signal",
                "trace_id": "runner-trace",
                "signal_score": 7.8,
            },
        }
    )

    with sqlite3.connect(":memory:") as conn:
        ensure_trade_ledger_schema(conn)
        materialize_trade_events(conn, [filled, approved])
        row = conn.execute(
            "SELECT state, signal_id, decision_at, entry_filled_at "
            "FROM trade_ledger WHERE trade_id = 'TRD_executed-signal'"
        ).fetchone()

    assert row == ("ENTRY_FILLED", "executed-signal", 1_790_232_002.0, 1_790_232_001.0)
    assert approved["meta"]["signal_score"] == 7.8
