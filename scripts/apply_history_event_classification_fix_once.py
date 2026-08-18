from pathlib import Path


def replace_exact(path: str, old: str, new: str, expected: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} matches, got {count} for {old!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


for path in ("dashboard/event_buffer.py", "dashboard/superlite_events.py"):
    replace_exact(
        path,
        'SOFT_HISTORY_FAILURE_REASONS = {"broker_fetch_not_allowed"}',
        'SOFT_HISTORY_FAILURE_REASONS = {"broker_fetch_not_allowed", "source_history_short"}',
    )
    replace_exact(
        path,
        '        "CANONICAL_HISTORY_RESULT" in upper\n        and role in SOFT_HISTORY_ROLES',
        '        _history_diagnostic(upper)\n        and role in SOFT_HISTORY_ROLES',
    )


test_path = "tests/dashboard/test_event_buffer_truth.py"
anchor = '''\ndef test_selected_option_history_failure_remains_error():\n'''
regression = '''\ndef test_non_gating_runner_history_short_is_system_not_error():\n    line = (\n        "[2026-08-18 18:16:19 IST] ✅ RUNNER_HISTORY_SYNC_RESULT "\n        "symbol=NFO:NIFTY2681824100CE role=option_context reason=startup_hydration "\n        "required_bars=50 mdm_after=1 runner_after=1 indicator_after=1 "\n        "success=False failure_reason=source_history_short"\n    )\n\n    event = MODULE.parse_event(line)\n    assert event is not None\n    assert event["type"] == "SYSTEM"\n\n    superlite_event = SUPERLITE.parse_event(line)\n    assert superlite_event is not None\n    assert superlite_event["type"] == "SYSTEM"\n    assert superlite_event["failure_reason"] == "source_history_short"\n\n\ndef test_selected_option_runner_history_short_remains_error():\n    line = (\n        "[2026-08-18 18:16:19 IST] RUNNER_HISTORY_SYNC_RESULT "\n        "symbol=NFO:NIFTY2681824150CE role=selected_option reason=startup_hydration "\n        "required_bars=75 mdm_after=1 runner_after=1 indicator_after=1 "\n        "success=False failure_reason=source_history_short"\n    )\n\n    event = MODULE.parse_event(line)\n    superlite_event = SUPERLITE.parse_event(line)\n\n    assert event is not None and event["type"] == "ERROR"\n    assert superlite_event is not None and superlite_event["type"] == "ERROR"\n\n'''
replace_exact(test_path, anchor, regression + anchor)
