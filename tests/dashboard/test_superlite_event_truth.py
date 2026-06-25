from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "dashboard" / "superlite_events.py"
SPEC = importlib.util.spec_from_file_location("superlite_events_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_empty_failure_reason_is_not_process_error() -> None:
    event = MODULE.parse_event(
        "[2026-06-26 00:00:00 IST] RUNNER_HISTORY_SYNC_RESULT "
        "symbol=NFO:X success=False failure_reason=None"
    )
    assert event is not None
    assert event["type"] == "SYSTEM"


def test_indicator_compute_error_remains_visible() -> None:
    event = MODULE.parse_event(
        "[2026-06-26 00:00:00 IST] INDICATOR_COMPUTE_ERROR symbol=NFO:X error=boom"
    )
    assert event is not None
    assert event["type"] == "ERROR"


def test_replayed_exact_event_is_deduplicated() -> None:
    row = {
        "timestamp_ist": "2026-06-26 00:00:00 IST",
        "type": "SYSTEM",
        "message": "BROKER_READY",
    }
    assert MODULE.deduplicate_events([row, dict(row)]) == [row]
