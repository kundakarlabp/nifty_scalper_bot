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
        "meta": {
            "approval_path": "single_trigger_context_confirmed",
            "setup_pass": True,
        },
    }

    row = _build_trade_row(event)

    assert row is not None
    assert row["trade_id"] == event["trade_id"]
    assert row["signal_id"] == event["signal_id"]
    assert row["decision_at"] == event["timestamp"]
    assert row["state"] == "SIGNAL_EVALUATED"
    assert _build_trade_row({**event, "trade_id": None}) is None


def test_approved_decision_after_fill_preserves_structural_provenance(
    tmp_path,
) -> None:
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
                "approval_path": "single_trigger_context_confirmed",
                "setup_pass": True,
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
    assert approved["meta"]["approval_path"] == "single_trigger_context_confirmed"
    assert approved["meta"]["setup_pass"] is True


def test_causal_bracket_decision_precedes_delayed_approval_in_any_order() -> None:
    base = 1_790_232_000.0
    common = {
        "trade_id": "TRD_causal-sequence",
        "signal_id": "causal-sequence",
        "symbol": "NFO:NIFTY26SEP23250CE",
    }
    bracket = {
        **common,
        "event_name": "bracket.registered",
        "timestamp": base + 0.8,
        "meta": {"decision_ts": base},
    }
    fill = {
        **common,
        "event_name": "entry.filled",
        "timestamp": base + 1.5,
        "order_id": "order-1",
        "price": 114.7,
        "qty": 65,
    }
    late_approval = {
        **common,
        "event_name": "candidate.approved",
        "timestamp": base + 1.7,
        "meta": {"approval_path": "single_trigger_context_confirmed"},
    }

    for sequence in (
        [bracket, fill, late_approval],
        [late_approval, fill, bracket],
    ):
        with sqlite3.connect(":memory:") as conn:
            ensure_trade_ledger_schema(conn)
            materialize_trade_events(conn, sequence)
            row = conn.execute(
                "SELECT decision_at, entry_filled_at FROM trade_ledger "
                "WHERE trade_id = ?",
                (common["trade_id"],),
            ).fetchone()
        assert row == (base, base + 1.5)


def test_bracket_decision_timestamp_must_be_finite_and_causally_recent() -> None:
    event = {
        "event_name": "bracket.registered",
        "trade_id": "TRD_bad-decision",
        "timestamp": 1_790_232_050.0,
    }
    for untrusted in (None, "nan", "inf", 1_790_232_051.0, 1_790_231_700.0):
        row = _build_trade_row({**event, "meta": {"decision_ts": untrusted}})
        assert row is not None
        assert row["decision_at"] is None

    approved = _build_trade_row({**event, "event_name": "candidate.approved"})
    assert approved is not None
    assert approved["decision_at"] == event["timestamp"]
