from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "event_buffer.py"
SPEC = importlib.util.spec_from_file_location("dashboard_event_buffer_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_expected_candidate_rejection_is_not_process_error():
    event = MODULE.parse_event("[2026-06-25 15:00:00 IST] CANDIDATE_REJECTED symbol=X reason=tick_stale")
    assert event is not None
    assert event["type"] == "SIGNAL"


def test_actual_runner_error_remains_error():
    event = MODULE.parse_event("[2026-06-25 15:00:00 IST] RUNNER_ON_TICK_ERROR symbol=X error=boom")
    assert event is not None
    assert event["type"] == "ERROR"


def test_duplicate_terminal_result_is_removed():
    rows = [
        {"timestamp_ist": "2026-06-25 15:00:00 IST", "type": "SIGNAL", "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_execution_ready_candidate trace_id=t1"},
        {"timestamp_ist": "2026-06-25 15:00:01 IST", "type": "TRADE", "message": "SIGNAL_EXECUTION_RESULT accepted=False reason=no_execution_ready_candidate trace_id=t1"},
    ]
    assert len(MODULE.deduplicate_events(rows)) == 1
