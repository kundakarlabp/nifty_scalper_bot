from __future__ import annotations

from datetime import date, datetime, time, timezone

from dashboard import log_export


def test_market_window_is_converted_from_ist_to_epoch() -> None:
    since, until = log_export.window_epochs(
        date(2026, 6, 25),
        time(9, 15),
        time(15, 30),
    )
    assert datetime.fromtimestamp(since, tz=timezone.utc) == datetime(
        2026, 6, 25, 3, 45, tzinfo=timezone.utc
    )
    assert datetime.fromtimestamp(until, tz=timezone.utc) == datetime(
        2026, 6, 25, 10, 0, tzinfo=timezone.utc
    )


def test_journal_command_uses_absolute_epoch_not_host_local_time() -> None:
    command = log_export.journal_command(
        "niftybot",
        date(2026, 6, 25),
        time(9, 15),
        time(15, 30),
        "cat",
    )
    since_value = command[command.index("--since") + 1]
    until_value = command[command.index("--until") + 1]
    assert since_value.startswith("@")
    assert until_value.startswith("@")
    assert "09:15" not in command


def test_invalid_window_is_rejected() -> None:
    try:
        log_export.window_epochs(date(2026, 6, 25), time(15, 30), time(9, 15))
    except ValueError as exc:
        assert "earlier" in str(exc)
    else:
        raise AssertionError("invalid window was accepted")


def test_csv_export_preserves_unicode_and_columns() -> None:
    payload = log_export.csv_bytes(
        [
            {
                "timestamp_ist": "2026-06-25 09:30:00 IST",
                "type": "TRADE",
                "message": "FILLED pnl=₹12.50",
            }
        ]
    )
    decoded = payload.decode("utf-8-sig")
    assert decoded.splitlines()[0] == "timestamp_ist,type,message"
    assert "FILLED pnl=₹12.50" in decoded


def test_filter_events_supports_type_and_text() -> None:
    rows = [
        {"timestamp_ist": "x", "type": "TRADE", "message": "FILLED NIFTY"},
        {"timestamp_ist": "y", "type": "ERROR", "message": "broker timeout"},
    ]
    assert log_export.filter_events(rows, "TRADE", "nifty") == [rows[0]]

def test_actionable_export_uses_canonical_parser_and_redacts_credentials(
    monkeypatch,
) -> None:
    raw = (
        "[2026-08-18 15:06:06 IST] RUNNER_ON_TICK_ERROR "
        "symbol=NFO:X error=boom api_key=test-placeholder\n"
    ).encode()
    monkeypatch.setattr(
        log_export,
        "run_journal",
        lambda _: log_export.JournalResult(raw, 1, False),
    )
    rows, result = log_export.read_actionable_events(
        "niftybot", date(2026, 8, 18), time(9, 15), time(15, 30)
    )
    assert result.count == 1
    assert rows[0]["type"] == "ERROR"
    assert rows[0]["symbol"] == "NFO:X"
    assert "test-placeholder" not in rows[0]["message"]
    assert "symbol" in log_export.csv_bytes(rows).decode("utf-8-sig").splitlines()[0]


def test_raw_journal_export_redacts_credentials(monkeypatch) -> None:
    raw = b"2026-08-18T10:01:00+05:30 RUNNER api_key=test-placeholder\n"
    monkeypatch.setattr(
        log_export,
        "run_journal",
        lambda _: log_export.JournalResult(raw, 1, False),
    )
    result = log_export.read_raw_logs(
        "niftybot", date(2026, 8, 18), time(9, 15), time(15, 30)
    )
    assert result.error is None
    assert b"api_key=[REDACTED]" in result.data
    assert b"test-placeholder" not in result.data
