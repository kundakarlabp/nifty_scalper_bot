"""Regression contracts for conservative routine release timing and provenance."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deploy/lightsail_release.sh"


def _window_exit(weekday: int, hour_minute: str) -> int:
    script = RELEASE.read_text(encoding="utf-8")
    start = script.index("routine_market_window_ist() {")
    end = script.index("\n}\n", start) + 2
    function = script[start:end]
    mocked_clock = """
date() {
  case "${@: -1}" in
    '+%u') printf '%s\\n' "$MOCK_WEEKDAY" ;;
    '+%H%M') printf '%s\\n' "$MOCK_HHMM" ;;
    *) return 1 ;;
  esac
}
"""
    env = {**os.environ, "MOCK_WEEKDAY": str(weekday), "MOCK_HHMM": hour_minute}
    return subprocess.run(
        ["bash", "-c", mocked_clock + function + "\nroutine_market_window_ist"],
        env=env,
        check=False,
    ).returncode


def test_routine_release_window_covers_nifty_session_and_opening_warmup() -> None:
    assert _window_exit(4, "0900") == 0
    assert _window_exit(4, "1529") == 0
    assert _window_exit(4, "1544") == 0
    assert _window_exit(4, "1545") != 0
    assert _window_exit(4, "0859") != 0
    assert _window_exit(6, "1100") != 0


def test_routine_market_deferral_is_before_checkout_and_protects_ci_gate() -> None:
    script = RELEASE.read_text(encoding="utf-8")
    defer = script.index("write_status deferred_market_window")
    ci = script.index('verify_release_ci.py" "$AFTER"')
    checkout = script.index('git reset --hard --quiet "$AFTER"')
    assert defer < ci < checkout
    assert 'ALLOW_INTRADAY_RELEASE=false' in script
    assert '--emergency-intraday) ALLOW_INTRADAY_RELEASE=true' in script
    assert 'service_healthy_confirmed "$BEFORE"' in script
    assert 'intraday_flat_broker_confirmed' in script
    assert 'acquire_release_entry_hold' in script
    assert 'record_release_restart requested "$BEFORE" "$AFTER"' in script
    assert 'record_release_restart verified_healthy "$BEFORE" "$AFTER"' in script
    assert (
        subprocess.run(["bash", "-n", str(RELEASE)], check=False).returncode
        == 0
    )
