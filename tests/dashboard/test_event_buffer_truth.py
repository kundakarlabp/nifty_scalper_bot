from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "dashboard" / "event_buffer.py"
SPEC = importlib.util.spec_from_file_location("dashboard_event_buffer_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SUPERLITE_PATH = ROOT / "dashboard" / "superlite_events.py"
SUPERLITE_SPEC = importlib.util.spec_from_file_location("dashboard_superlite_event_truth", SUPERLITE_PATH)
SUPERLITE = importlib.util.module_from_spec(SUPERLITE_SPEC)
assert SUPERLITE_SPEC.loader is not None
SUPERLITE_SPEC.loader.exec_module(SUPERLITE)


def test_expected_candidate_rejection_is_not_process_error():
    event = MODULE.parse_event(
        "[2026-06-25 15:00:00 IST] CANDIDATE_REJECTED symbol=X reason=tick_stale"
    )
    assert event is not None
    assert event["type"] == "SIGNAL"


def test_actual_runner_error_remains_error():
    event = MODULE.parse_event(
        "[2026-06-25 15:00:00 IST] RUNNER_ON_TICK_ERROR symbol=X error=boom"
    )
    assert event is not None
    assert event["type"] == "ERROR"


def test_duplicate_terminal_result_is_removed():
    rows = [
        {
            "timestamp_ist": "2026-06-25 15:00:00 IST",
            "type": "SIGNAL",
            "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_candidate trace_id=t1",
        },
        {
            "timestamp_ist": "2026-06-25 15:00:01 IST",
            "type": "TRADE",
            "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_candidate trace_id=t1",
        },
    ]
    assert len(MODULE.deduplicate_events(rows)) == 1


def test_superlite_none_failure_reason_is_not_error():
    event = SUPERLITE.parse_event(
        "[2026-06-26 00:00:00 IST] RUNNER_HISTORY_SYNC_RESULT "
        "symbol=NFO:X success=False failure_reason=None"
    )
    assert event is not None
    assert event["type"] == "SYSTEM"


def test_superlite_indicator_error_is_visible():
    event = SUPERLITE.parse_event(
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
