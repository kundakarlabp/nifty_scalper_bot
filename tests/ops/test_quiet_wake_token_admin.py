"""Prove a scheduled QUIET wake does not depend on an expired Kite token.

A loaded engine, unknown runtime or mismatched build must still require
independent broker-flat/terminal-order proof before any release handoff.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deploy/lightsail_release.sh"
SHA = "f674601e87b7ccb3f590d1c240dbaa6604f26907"


def _unloaded_guard(live: object, release: object) -> int:
    """Execute only the extracted guard with stubbed localhost HTTP replies."""
    source = RELEASE.read_text(encoding="utf-8")
    start = source.index("quiet_control_plane_unloaded() {")
    end = source.index("\nauto_release_should_defer() {", start)
    function = source[start:end]
    script = (
        "curl() { "
        'case "$*" in '
        '*"/livez"*) printf "%s" "$LIVE_BODY" ;; '
        '*"/releasez"*) printf "%s" "$RELEASE_BODY" ;; '
        "*) return 1 ;; esac; }\n" + function + "\nquiet_control_plane_unloaded\n"
    )
    process = subprocess.run(
        ["bash", "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "BEFORE": SHA,
            "PORT": "8080",
            "VENV": str(Path(sys.executable).parent.parent),
            "LIVE_BODY": json.dumps(live),
            "RELEASE_BODY": json.dumps(release),
        },
    )
    return process.returncode


def _live() -> dict[str, object]:
    return {
        "quiet": True,
        "bot_loaded": False,
        "engine_http_responsive": True,
    }


def _release() -> dict[str, object]:
    return {"effective_sha": SHA[:12], "engine_expected": False}


def test_unloaded_quiet_can_wake_without_broker_http_or_token() -> None:
    assert _unloaded_guard(_live(), _release()) == 0


@pytest.mark.parametrize(
    ("live_change", "release_change"),
    [
        ({"quiet": False, "bot_loaded": True}, {}),
        ({"quiet": True, "bot_loaded": True}, {}),
        ({"engine_http_responsive": False}, {}),
        ({}, {"engine_expected": True}),
        ({}, {"effective_sha": "other-build"}),
        ({}, {"effective_sha": ""}),
    ],
)
def test_guard_never_waives_proof_for_active_or_uncertain_engine(
    live_change, release_change
) -> None:
    assert _unloaded_guard(_live() | live_change, _release() | release_change) != 0


def test_corrupt_local_evidence_does_not_authorize_restart() -> None:
    assert _unloaded_guard([], {}) != 0


def test_only_verified_quiet_guard_precedes_full_broker_proof() -> None:
    source = RELEASE.read_text(encoding="utf-8")
    defer = source.split("auto_release_should_defer() {", 1)[1]
    assert defer.index("if quiet_control_plane_unloaded; then") < defer.index(
        "intraday_flat_broker_confirmed && return 1"
    )
    assert defer.index('if [ "$RELEASE_HOLD_ACTIVE" != true ]') < defer.index(
        "if quiet_control_plane_unloaded; then"
    )
    assert "intraday_flat_broker_confirmed" in source


def test_request_token_is_available_beside_credentials_without_storing_it(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        admin_dashboard,
        "_read_env",
        lambda: {
            "ADMIN_ZERODHA_USER_ID": "AB1234",
            "KITE_API_KEY": "sample-key",
            "KITE_API_SECRET": "sample-secret",
            "EXECUTION_MODE": "SHADOW",
        },
    )
    app = FastAPI()
    app.include_router(admin_dashboard.router)
    with TestClient(app, base_url="https://65.0.255.84") as client:
        response = client.get("/admin")
        assert response.status_code == 200
        assert 'id="settings-request-token"' in response.text
        assert response.text.count('action="/admin/token"') == 2
        assert response.text.count('name="owner_state"') == 2
        assert "The request token is" in response.text
    assert all(key != "KITE_REQUEST_TOKEN" for _, key, _ in admin_dashboard.FIELDS)
