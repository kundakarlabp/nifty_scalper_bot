"""Token rotation must reach the regular validated release without exposure deferral."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_token_rotation_is_not_trapped_behind_old_broker_flat_guard() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(encoding="utf-8")
    assert "PY_FLAT_GUARD" not in release
    assert "auto_release_should_defer" not in release
    assert "deferred_broker_exposure" not in release
    assert 'if [ -f "$RESTART_PENDING_FILE" ]; then FORCE_RESTART=true; fi' in release
    assert 'sudo systemctl restart "$SERVICE"' in release


def test_token_rotation_remains_validated_and_never_claims_early_success() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(encoding="utf-8")
    assert 'write_status validating "validating' in release
    assert 'if wait_for_service "$AFTER"; then' in release
    assert 'rm -f "$RESTART_PENDING_FILE"' in release
    assert "write_status rolled_back" in release
    assert "write_status deferred_ci" in release
