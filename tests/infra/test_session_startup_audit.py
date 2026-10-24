"""Regression coverage for conservative session-recovery log summaries."""

from __future__ import annotations

from nifty_scalper_bot.infra.session_startup_audit import audit_session_startups


def line(time: str, event: str) -> str:
    return f"[2026-10-23 {time} IST] {event}"


def test_four_realistic_live_restarts_are_counted_without_claiming_safety() -> None:
    log = "\n".join(
        (
            line("09:49:33", "STARTUP_PHASE phase=create_context"),
            line("09:49:35", "STARTUP_WS_START_DONE"),
            line("09:49:36", "BROKER_ORDER_RECONCILE_OK seen=2"),
            line("09:50:04", "RECONCILE_COMPLETE"),
            line("09:50:04", "STARTUP_MODE_TRANSITION from=DATA_WARMUP to=LIVE_READY"),
            line("10:31:34", "STARTUP_PHASE phase=create_context"),
            line("10:31:36", "STARTUP_WS_START_DONE"),
            line("10:32:05", "RECONCILE_COMPLETE"),
            line("10:32:05", "STARTUP_MODE_TRANSITION from=DATA_WARMUP to=LIVE_READY"),
            line("14:02:50", "STARTUP_PHASE phase=create_context"),
            line("14:02:52", "STARTUP_WS_START_DONE"),
            line("14:03:21", "RECONCILE_COMPLETE"),
            line("14:03:21", "STARTUP_MODE_TRANSITION from=DATA_WARMUP to=LIVE_READY"),
            line("15:12:33", "STARTUP_PHASE phase=create_context"),
            line("15:12:35", "STARTUP_WS_START_DONE"),
            line("15:13:04", "RECONCILE_COMPLETE"),
            line("15:13:04", "STARTUP_MODE_TRANSITION from=DATA_WARMUP to=LIVE_READY"),
        )
    )
    summary = audit_session_startups(log)
    assert summary["observed_startup_count"] == 4
    assert summary["ws_started_count"] == 4
    assert summary["reconcile_complete_count"] == 4
    assert summary["live_ready_count"] == 4
    assert summary["startups_after_0930_count"] == 4
    assert summary["startups_without_live_ready_count"] == 0
    assert summary["first_startup_at_ist"] == "09:49:33"
    assert summary["last_startup_at_ist"] == "15:12:33"
    assert summary["last_live_ready_at_ist"] == "15:13:04"
    assert summary["max_startup_to_ready_seconds"] == 31
    assert summary["protective_stop_verification"] == "not_established_from_log_markers"


def test_missing_recovery_evidence_remains_explicit() -> None:
    summary = audit_session_startups(
        line("10:00:00", "STARTUP_PHASE phase=create_context")
    )
    assert summary["observed_startup_count"] == 1
    assert summary["ws_started_count"] == 0
    assert summary["live_ready_count"] == 0
    assert summary["startups_without_live_ready_count"] == 1
    assert summary["max_startup_to_ready_seconds"] is None
    assert summary["last_live_ready_at_ist"] is None


def test_unrelated_ready_and_invalid_loglines_do_not_fake_recovery() -> None:
    summary = audit_session_startups(
        "\n".join(
            (
                line("09:20:00", "STARTUP_MODE_TRANSITION to=LIVE_READY"),
                "[not-a-date] STARTUP_PHASE phase=create_context",
                line("09:31:00", "STARTUP_PHASE phase=create_context"),
                line("09:31:03", "LIVE_READY from=DATA_WARMUP"),
                line("09:31:04", "STARTUP_MODE_TRANSITION to=DATA_WARMUP"),
                "api_key=TOP_SECRET_DO_NOT_OUTPUT",
            )
        )
    )
    assert summary["observed_startup_count"] == 1
    assert summary["live_ready_count"] == 1
    assert summary["startups_without_live_ready_count"] == 1
    assert summary["max_startup_to_ready_seconds"] is None
    assert "TOP_SECRET" not in str(summary)


def test_no_logs_means_unknown_evidence_not_success() -> None:
    summary = audit_session_startups("")
    assert summary["observed_startup_count"] == 0
    assert summary["reconcile_complete_count"] == 0
    assert summary["max_startup_to_ready_seconds"] is None
    assert summary["protective_stop_verification"] == "not_established_from_log_markers"
