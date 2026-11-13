"""Guard repeated healthy AUTO releases without market-hour scheduling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deploy/lightsail_release.sh"


def _cooldown_exit(
    tmp_path: Path, *, last: str, now: str, interval: str = "900"
) -> int:
    script = RELEASE.read_text(encoding="utf-8")
    start = script.index("recent_successful_release() {")
    end = script.index("\n}\n", start) + 2
    function = script[start:end]
    marker = tmp_path / "last_success_epoch"
    marker.write_text(last, encoding="utf-8")
    mocked_clock = "date() { printf '%s\\n' \"$MOCK_NOW\"; }\n"
    env = {
        **os.environ,
        "MOCK_NOW": now,
        "AUTO_RELEASE_LAST_SUCCESS_FILE": str(marker),
        "AUTO_RELEASE_COOLDOWN_SECONDS": interval,
    }
    return subprocess.run(
        ["bash", "-c", mocked_clock + function + "\nrecent_successful_release"],
        env=env,
        check=False,
    ).returncode


def test_healthy_auto_release_cooldown_is_bounded_and_monotonic(
    tmp_path: Path,
) -> None:
    assert _cooldown_exit(tmp_path, last="1700000000", now="1700000001") == 0
    assert _cooldown_exit(tmp_path, last="1700000000", now="1700000899") == 0
    assert _cooldown_exit(tmp_path, last="1700000000", now="1700000900") != 0
    assert _cooldown_exit(tmp_path, last="1700000100", now="1700000000") != 0
    assert _cooldown_exit(tmp_path, last="invalid", now="1700000001") != 0


def test_no_market_clock_restriction_and_force_preserves_safety_gates() -> None:
    script = RELEASE.read_text(encoding="utf-8")
    defer = script.index("write_status deferred_release_cooldown")
    ci = script.index('verify_release_ci.py" "$AFTER"')
    checkout = script.index('git reset --hard --quiet "$AFTER"')
    assert defer < ci < checkout
    assert "routine_market_window_ist" not in script
    assert "TZ=Asia/Kolkata date +%H%M" not in script
    assert "TZ=Asia/Kolkata date +%u" not in script
    assert 'if [ "$FORCE_RESTART" != true ]' in script
    assert 'service_healthy_confirmed "$BEFORE"' in script
    assert "intraday_flat_broker_confirmed" in script
    assert "acquire_release_entry_hold" in script
    assert 'record_release_restart requested "$BEFORE" "$AFTER"' in script
    assert 'record_release_restart verified_healthy "$BEFORE" "$AFTER"' in script
    assert 'record_release_restart rolled_back "$AFTER" "$BEFORE"' in script
    assert subprocess.run(["bash", "-n", str(RELEASE)], check=False).returncode == 0
