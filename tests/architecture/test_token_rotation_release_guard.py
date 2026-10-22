"""Regression tests for fail-closed Zerodha token-rotation release recovery."""

from __future__ import annotations

import os
import subprocess
import sys
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
live = {"engine_http_responsive": True, "bot_loaded": True}
def response(url, timeout=3):
    value = live if url.endswith("/livez") else health
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

kiteconnect = types.ModuleType("kiteconnect")
kiteconnect.KiteConnect = FakeKite
sys.modules["kiteconnect"] = kiteconnect
"""


@pytest.mark.parametrize(
    ("mode", "marker_exists", "overrides", "expected_success"),
    [
        ("valid", False, {}, True),
        ("invalid", False, {}, False),
        ("invalid", True, {}, True),
        ("unknown", True, {}, False),
        ("invalid", True, {"TEST_POSITION": "1"}, False),
        ("invalid", True, {"TEST_OPEN_ORDER": "1"}, False),
        ("invalid", True, {"TEST_ARMED": "1"}, False),
        ("invalid", True, {"TEST_RECONCILED": "0"}, False),
        ("invalid", True, {"TEST_KNOWN": "0"}, False),
        ("invalid", True, {"TEST_API_FAIL": "1"}, False),
    ],
)
def test_queued_token_rotation_requires_flat_broker_and_disarmed_engine(
    tmp_path: Path,
    mode: str,
    marker_exists: bool,
    overrides: dict[str, str],
    expected_success: bool,
) -> None:
    marker = tmp_path / "restart_pending"
    if marker_exists:
        marker.touch()
    env = {
        **os.environ,
        "BOT_RELEASE_LOCAL_PORT": "8080",
        "BOT_RELEASE_KITE_API_KEY": "fake_key",
        "BOT_RELEASE_KITE_ACCESS_TOKEN": "fake_token",
        "BOT_RELEASE_RESTART_PENDING": str(marker),
        "TEST_AUTH": mode,
        "TEST_ARMED": "0",
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
    assert "health.get(\"live_orders_armed\") is not False" in guard
    assert "reconciliation.get(\"completed\") is not True" in guard
    assert "reconciliation.get(\"unprotected_positions\")" in guard
    assert "broker.get(\"auth_invalid\") is True" in guard
    assert "positions = kite.positions()" in guard
    assert "orders = kite.orders()" in guard
    assert guard.count("if not is_safe():") == 2
