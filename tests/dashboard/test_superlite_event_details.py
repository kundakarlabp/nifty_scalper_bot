from __future__ import annotations

from dashboard.superlite_events import csv_bytes, parse_event


def test_data_integrity_event_exports_structured_fields() -> None:
    row = parse_event(
        "[2026-07-06 11:02:15 IST] ERROR data_integrity_error "
        "event=data_integrity_error symbol=NFO:NIFTY2670724400CE "
        "reason=historical_validation_failed source=fetch_historical_safe "
        "attempt=1 required_bars=50 rows=44"
    )

    assert row is not None
    assert row["type"] == "ERROR"
    assert row["symbol"] == "NFO:NIFTY2670724400CE"
    assert row["reason"] == "historical_validation_failed"
    assert row["source"] == "fetch_historical_safe"
    assert row["attempt"] == "1"
    assert row["required_bars"] == "50"
    assert row["rows"] == "44"

    decoded = csv_bytes([row]).decode("utf-8-sig")
    header = decoded.splitlines()[0]
    assert "symbol" in header
    assert "reason" in header
    assert "source" in header
    assert "NFO:NIFTY2670724400CE" in decoded


def test_bare_data_integrity_event_is_marked_as_missing_details() -> None:
    row = parse_event("[2026-07-06 11:02:15 IST] ERROR data_integrity_error")

    assert row is not None
    assert row["type"] == "ERROR"
    assert row["reason"] == "missing_structured_details"
    assert row["source"] == "journal_message"
