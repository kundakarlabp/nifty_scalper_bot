"""Regression tests for fail-closed Zerodha token-rotation release recovery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _guard_source() -> str:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(encoding="utf-8")
    return release.split("<<'PY_FLAT_GUARD'\n", 1)[1].split("\nPY_FLAT_GUARD", 1)[0]


# No broker network access. Exercise the actual release guard's Python body.
_STUB = """
import io
import json
import os
import sys
import time
import types
import urllib.request

mode = os.environ["TEST_AUTH"]
health = {
    "broker_authentication": mode,
    "broker_session_state": mode,
    "authentication_known": os.environ["TEST_KNOWN"] == "1",
    "live_orders_armed": os.environ["TEST_ARMED"] == "1",
    "release_entry_hold_active": os.environ["TEST_HOLD"] == "1",
    "broker": {
        "authenticated": (
            True if mode == "valid" else (False if mode == "invalid" else None)
        ),
        "auth_invalid": mode == "invalid",
    },
    "reconciliation": {
        "completed": os.environ["TEST_RECONCILED"] == "1",
        "failed": False,
        "unprotected_positions": [],
    },
}
live = {
    "engine_http_responsive": True,
    "bot_loaded": os.environ.get("TEST_BOT_LOADED", "1") == "1",
}
ready = {
    "status": (
        "degraded" if os.environ.get("TEST_STARTUP_FAILED", "0") == "1"
        else "starting"
    ),
    "primary_blocker": (
        "startup_failed" if os.environ.get("TEST_STARTUP_FAILED", "0") == "1"
        else "startup_incomplete"
    ),
    "error": os.environ.get(
        "TEST_STARTUP_ERROR", "LIVE startup requires validated broker balance"
    ),
}
release = {"effective_sha": os.environ.get("TEST_RELEASE_SHA", "a" * 12)}
if not live["bot_loaded"]:
    health.update({
        "status": "starting",
        "primary_blocker": "startup_incomplete",
        "live_orders_armed": False,
    })
def response(url, timeout=3):
    if url.endswith("/livez"):
        value = live
    elif url.endswith("/readyz"):
        value = ready
    elif url.endswith("/releasez"):
        value = release
    else:
        value = health
    return io.BytesIO(json.dumps(value).encode("utf-8"))
urllib.request.urlopen = response
time.sleep = lambda seconds: None

class FakeKite:
    def __init__(self, api_key, timeout):
        pass
    def set_access_token(self, token):
        pass
    def positions(self):
        if os.environ["TEST_API_FAIL"] == "1":
            raise RuntimeError("broker unavailable")
        return {
            "net": [{"quantity": int(os.environ["TEST_POSITION"])}],
            "day": [],
        }
    def orders(self):
        return [{"status": "OPEN"}] if os.environ["TEST_OPEN_ORDER"] == "1" else []
    def profile(self):
        return {"user_id": os.environ.get("TEST_PROFILE_USER", "AB1234")}
    def margins(self, segment):
        if os.environ.get("TEST_MARGIN_MISSING") == "1":
            return {}
        return {"available": {"live_balance": 1000.0}}

kiteconnect = types.ModuleType("kiteconnect")
kiteconnect.KiteConnect = FakeKite
sys.modules["kiteconnect"] = kiteconnect
"""


_FAILED_START = {"TEST_BOT_LOADED": "0", "TEST_STARTUP_FAILED": "1"}


@pytest.mark.parametrize(
    ("mode", "verified", "overrides", "expected_success"),
    [
        ("valid", False, {}, True),
        ("invalid", False, {}, False),
        ("unknown", False, {}, False),
        ("invalid", True, {}, True),
        ("unknown", True, {}, True),
        ("invalid", True, {"TEST_POSITION": "1"}, False),
        ("invalid", True, {"TEST_OPEN_ORDER": "1"}, False),
        ("invalid", True, {"TEST_ARMED": "1"}, False),
        ("invalid", True, {"TEST_RECONCILED": "0"}, False),
        ("invalid", True, {"TEST_API_FAIL": "1"}, False),
        ("invalid", True, {"TEST_HOLD": "0"}, False),
        ("invalid", True, {"TEST_BAD_DIGEST": "1"}, False),
        ("invalid", True, {"TEST_STALE_PROOF": "1"}, False),
        ("invalid", True, {"TEST_GENERIC_ONLY": "1"}, False),
        # A constructor failure cannot complete local reconciliation. Permit
        # recovery only with fresh, broker-bound proof and no trading stack.
        ("invalid", True, {**_FAILED_START, "TEST_RECONCILED": "0"}, True),
        ("invalid", False, _FAILED_START, False),
        ("invalid", True, {"TEST_BOT_LOADED": "0"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_POSITION": "1"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_OPEN_ORDER": "1"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_PROFILE_USER": "WRONG"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_MARGIN_MISSING": "1"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_STARTUP_ERROR": "other"}, False),
        ("invalid", True, {**_FAILED_START, "TEST_RELEASE_SHA": "b" * 12}, False),
        ("invalid", True, {**_FAILED_START, "TEST_API_FAIL": "1"}, False),
    ],
)
def test_queued_token_rotation_requires_verified_token_and_broker_flat(
    tmp_path: Path,
    mode: str,
    verified: bool,
    overrides: dict[str, str],
    expected_success: bool,
) -> None:
    marker = tmp_path / "restart_pending"
    marker.touch()
    proof_file = tmp_path / "verified_rotation"
    if verified:
        digest = hashlib.sha256(b"fake_token").hexdigest()
        if overrides.get("TEST_BAD_DIGEST") == "1":
            digest = "0" * 64
        stamp = time.time()
        if overrides.get("TEST_STALE_PROOF") == "1":
            stamp -= 15 * 3600
        proof_file.write_text(
            json.dumps({"token_sha256": digest, "verified_at": stamp}),
            encoding="utf-8",
        )
    if overrides.get("TEST_GENERIC_ONLY") == "1":
        proof_file.unlink(missing_ok=True)
    env = {
        **os.environ,
        "BOT_RELEASE_LOCAL_PORT": "8080",
        "BOT_RELEASE_KITE_API_KEY": "fake_key",
        "BOT_RELEASE_KITE_ACCESS_TOKEN": "fake_token",
        "BOT_RELEASE_RESTART_PENDING": str(marker),
        "BOT_RELEASE_VERIFIED_TOKEN_ROTATION": str(proof_file),
        "BOT_RELEASE_ENTRY_HOLD_REQUIRED": (
            "false" if overrides.get("TEST_HOLD") == "0" else "true"
        ),
        "BOT_RELEASE_EXPECTED_OWNER": "AB1234",
        "BOT_RELEASE_EXPECTED_SHA": "a" * 40,
        "BOT_RELEASE_MARGIN_SEGMENT": "equity",
        "TEST_AUTH": mode,
        "TEST_ARMED": "0",
        "TEST_HOLD": "1",
        "TEST_RECONCILED": "1",
        "TEST_KNOWN": "1",
        "TEST_POSITION": "0",
        "TEST_OPEN_ORDER": "0",
        "TEST_API_FAIL": "0",
        **overrides,
    }
    result = subprocess.run(
        [sys.executable, "-c", _STUB + "\n" + _guard_source()],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=15,
    )
    assert (result.returncode == 0) is expected_success, result.stderr


def test_rotation_exception_is_scoped_and_double_sampled() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text(encoding="utf-8")
    guard = _guard_source()
    assert 'BOT_RELEASE_RESTART_PENDING="$RESTART_PENDING_FILE"' in release
    assert "os.path.isfile(restart_marker)" in guard
    assert 'health.get("live_orders_armed") is not False' in guard
    assert 'ready.get("primary_blocker") == "startup_failed"' in guard
    assert (
        'ready.get("error") == "LIVE startup requires validated broker balance"'
        in guard
    )
    assert "token_rotation_verified and hold_required" in guard
    assert "profile = kite.profile()" in guard
    assert "margins = kite.margins(" in guard
    assert 'reconciliation.get("completed") is not True' in guard
    assert 'reconciliation.get("unprotected_positions")' in guard
    assert "token_rotation_verified and hold_required" in guard
    marker = 'BOT_RELEASE_VERIFIED_TOKEN_ROTATION="$VERIFIED_TOKEN_ROTATION_FILE"'
    assert marker in release
    assert 'rm -f "$RESTART_PENDING_FILE" "$VERIFIED_TOKEN_ROTATION_FILE"' in release
    assert "positions = kite.positions()" in guard
    assert "orders = kite.orders()" in guard
    assert guard.count("if not is_safe():") == 2
