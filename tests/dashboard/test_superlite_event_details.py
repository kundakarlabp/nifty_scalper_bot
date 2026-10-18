from __future__ import annotations

from pathlib import Path

from dashboard.superlite_events import csv_bytes, deduplicate_events, parse_event

ROOT = Path(__file__).resolve().parents[2]


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


def test_expected_candidate_rejection_is_not_process_error():
    event = parse_event(
        "[2026-06-25 15:00:00 IST] CANDIDATE_REJECTED symbol=X reason=tick_stale"
    )
    assert event is not None
    assert event["type"] == "SIGNAL"


def test_actual_runner_error_remains_error():
    event = parse_event(
        "[2026-06-25 15:00:00 IST] RUNNER_ON_TICK_ERROR symbol=X error=boom"
    )
    assert event is not None
    assert event["type"] == "ERROR"


def test_duplicate_terminal_result_is_removed():
    rows = [
        {
            "timestamp_ist": "2026-06-25 15:00:00 IST",
            "type": "SIGNAL",
            "message": (
                "SIGNAL_EXECUTION_RESULT accepted=False "
                "reason=no_candidate trace_id=t1"
            ),
        },
        {
            "timestamp_ist": "2026-06-25 15:00:01 IST",
            "type": "TRADE",
            "message": (
                "SIGNAL_EXECUTION_RESULT accepted=False "
                "reason=no_candidate trace_id=t1"
            ),
        },
    ]
    assert len(deduplicate_events(rows)) == 1


def test_superlite_none_failure_reason_is_not_error():
    event = parse_event(
        "[2026-06-26 00:00:00 IST] RUNNER_HISTORY_SYNC_RESULT "
        "symbol=NFO:X success=False failure_reason=None"
    )
    assert event is not None
    assert event["type"] == "SYSTEM"


def test_no_combined_signal_is_a_system_decision_not_a_signal() -> None:
    line = (
        "[2026-08-06 14:40:38 IST] "
        "strategy_manager_no_combined_signal symbol=NFO:NIFTY2681124650CE "
        "no_vote_reason_counts={'orb_not_ready':1}"
    )

    event = parse_event(line)

    assert event is not None and event["type"] == "SYSTEM"


def test_non_gating_option_context_history_miss_is_system_not_error():
    line = (
        "[2026-07-08 01:13:00 IST] ✅ CANONICAL_HISTORY_RESULT "
        "symbol=NFO:NIFTY2671424450PE role=option_context phase=startup "
        "reason=startup_hydration required_bars=50 target_bars=50 mdm_bars=1 "
        "runner_bars=1 indicator_bars=1 minimum_ready=False target_ready=False "
        "failure_reason=broker_fetch_not_allowed"
    )

    event = parse_event(line)
    assert event is not None
    assert event["type"] == "SYSTEM"
    assert event["failure_reason"] == "broker_fetch_not_allowed"


def test_non_gating_runner_history_short_is_system_not_error():
    line = (
        "[2026-08-18 18:16:19 IST] ✅ RUNNER_HISTORY_SYNC_RESULT "
        "symbol=NFO:NIFTY2681824100CE role=option_context reason=startup_hydration "
        "required_bars=50 mdm_after=1 runner_after=1 indicator_after=1 "
        "success=False failure_reason=source_history_short"
    )

    event = parse_event(line)
    assert event is not None
    assert event["type"] == "SYSTEM"
    assert event["failure_reason"] == "source_history_short"


def test_selected_option_runner_history_short_remains_error():
    line = (
        "[2026-08-18 18:16:19 IST] RUNNER_HISTORY_SYNC_RESULT "
        "symbol=NFO:NIFTY2681824150CE role=selected_option reason=startup_hydration "
        "required_bars=75 mdm_after=1 runner_after=1 indicator_after=1 "
        "success=False failure_reason=source_history_short"
    )

    event = parse_event(line)

    assert event is not None and event["type"] == "ERROR"


def test_selected_option_history_failure_remains_error():
    line = (
        "[2026-07-08 01:13:00 IST] CANONICAL_HISTORY_RESULT "
        "symbol=NFO:NIFTY2671424400CE role=selected_option required_bars=30 "
        "mdm_bars=1 runner_bars=1 indicator_bars=1 "
        "failure_reason=broker_fetch_not_allowed"
    )

    event = parse_event(line)
    assert event is not None
    assert event["type"] == "ERROR"


def test_superlite_indicator_error_is_visible():
    event = parse_event(
        "[2026-06-26 00:00:00 IST] INDICATOR_COMPUTE_ERROR symbol=NFO:X"
    )
    assert event is not None
    assert event["type"] == "ERROR"


def test_superlite_services_are_independent_and_bounded():
    admin_path = ROOT / "deploy/systemd/niftybot-admin.service"
    review_path = ROOT / "deploy/systemd/niftybot-streamlit.service"
    admin = admin_path.read_text(encoding="utf-8")
    review = review_path.read_text(encoding="utf-8")
    assert "nifty_scalper_bot.superlite_admin:app" in admin
    assert "MemoryMax=180M" in admin
    assert "dashboard/superlite_console.py" in review
    assert "MemoryMax=320M" in review


def test_readiness_and_candle_warnings_are_not_trade_events() -> None:
    readiness = parse_event(
        "[2026-07-01 09:10:00 IST] ORDER_READINESS_REJECTED "
        "reason=selected_option_quote_missing"
    )
    candle = parse_event(
        "[2026-07-01 09:11:00 IST] Dropping out-of-order candle symbol=NFO:X"
    )
    assert readiness is not None and readiness["type"] != "TRADE"
    assert candle is not None and candle["type"] != "TRADE"


def test_strategy_no_vote_failed_reason_is_not_process_error() -> None:
    line = (
        "[2026-08-18 15:06:06 IST] STRATEGY_NO_VOTE "
        "symbol=NFO:NIFTY2681824200CE reason=smc_quality_gate_failed"
    )

    event = parse_event(line)

    assert event is not None and event["type"] == "SYSTEM"


def test_no_combined_signal_failed_reason_is_not_process_error() -> None:
    line = (
        "[2026-08-18 15:06:06 IST] strategy_manager_no_combined_signal "
        "symbol=NFO:NIFTY2681824200CE "
        "no_vote_reason_counts={'smc_quality_gate_failed':1}"
    )

    event = parse_event(line)

    assert event is not None and event["type"] == "SYSTEM"


def test_hard_error_precedence_survives_strategy_rejection_markers() -> None:
    line = (
        "[2026-08-18 15:06:06 IST] RUNNER_ON_TICK_ERROR "
        "symbol=NFO:NIFTY2681824200CE reason=smc_quality_gate_failed "
        "context=STRATEGY_NO_VOTE error=boom"
    )

    event = parse_event(line)

    assert event is not None and event["type"] == "ERROR"


def test_trade_event_is_kept() -> None:
    event = parse_event("[2026-06-23 09:35:00 IST] ORDER_SENT NIFTY CE")
    assert event is not None
    assert event["type"] == "TRADE"


def test_heartbeat_is_dropped() -> None:
    assert parse_event("[2026-06-23 09:35:00 IST] HEARTBEAT ok") is None


def test_signal_event_is_classified() -> None:
    event = parse_event("[2026-06-23 09:35:00 IST] SIGNAL_GENERATED BUY")
    assert event is not None
    assert event["type"] == "SIGNAL"
