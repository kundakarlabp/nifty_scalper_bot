"""Regressions for the single daily Zerodha request-token form."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot import superlite_admin_core as core
from nifty_scalper_bot.ops.service_control import ControlResult


def operator_client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard, "_read_env",
        lambda: {
            "ADMIN_ZERODHA_USER_ID": "AB1234",
            "KITE_API_KEY": "public-test-key",
            "KITE_API_SECRET": "private-test-secret",
            "ENABLE_LIVE": "false",
            "EXECUTION_MODE": "SHADOW",
        },
    )
    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(app, base_url="https://65.0.255.84")


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("opaque-request", "opaque-request"),
        (
            "https://callback.example/path?request_token=opaque-request&status=success",
            "opaque-request",
        ),
        ("request_token=opaque-request&status=success", "opaque-request"),
    ],
)
def test_paste_form_accepts_raw_token_or_full_url(
    monkeypatch, tmp_path, submission, expected
):
    client = operator_client(monkeypatch, tmp_path)
    page = client.get("/admin")
    assert page.status_code == 200
    assert '<details open><summary>Paste callback URL / request token' in page.text
    state = re.search(r'name="owner_state" value="([^"]+)"', page.text)
    assert state is not None
    exchanges = []
    monkeypatch.setattr(
        dashboard, "_exchange_request_token",
        lambda key, secret, token: exchanges.append((key, secret, token))
        or (True, "verified-access"),
    )
    monkeypatch.setattr(dashboard, "_verify_access_token", lambda *_: (True, "AB1234"))
    queued = []
    monkeypatch.setattr(
        dashboard, "_activate_access_token",
        lambda token: queued.append(token)
        or ControlResult(True, "restart_bot", "queued"),
    )
    result = client.post(
        "/admin/token",
        data={"owner_state": state.group(1), "request_token": submission},
        headers={"origin": "https://65.0.255.84"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert result.headers["location"] == "/admin?token=ok"
    assert exchanges == [("public-test-key", "private-test-secret", expected)]
    assert queued == ["verified-access"]
    assert result.cookies.get("nifty_owner_session")
    again = client.get("/admin")
    assert '<details open><summary>Paste callback URL / request token' in again.text


def test_existing_owner_session_cannot_rotate_to_wrong_kite_account(
    monkeypatch, tmp_path
):
    client = operator_client(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard, "valid_owner_session", lambda *_: True)
    monkeypatch.setattr(
        dashboard, "_exchange_request_token", lambda *_: (True, "foreign-access")
    )
    monkeypatch.setattr(dashboard, "_verify_access_token", lambda *_: (True, "ZZ9999"))
    monkeypatch.setattr(
        dashboard, "_activate_access_token",
        lambda *_: (_ for _ in ()).throw(AssertionError("wrong owner cannot rotate")),
    )
    result = client.post(
        "/admin/token",
        data={"request_token": "new-request"},
        headers={"origin": "https://65.0.255.84"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert "token=err" in result.headers["location"]
    assert "mismatch" in result.headers["location"]


def test_admin_uses_canonical_atomic_environment_writer(monkeypatch):
    assert dashboard.ENV_PATH == core.ENV_PATH
    written = []
    monkeypatch.setattr(
        dashboard, "write_canonical_env", lambda updates: written.append(updates)
    )
    dashboard._ENV_CACHE.update({"at": 100, "data": {"old": "stale"}})
    dashboard._write_env({"KITE_ACCESS_TOKEN": "sample"})
    assert written == [{"KITE_ACCESS_TOKEN": "sample"}]
    assert dashboard._ENV_CACHE == {"at": 0.0, "data": {}}


def test_review_links_target_secure_token_form():
    source = (
        Path(__file__).resolve().parents[2] / "dashboard/superlite_console.py"
    ).read_text()
    assert '_parsed_admin.scheme != "https"' in source
    assert 'ADMIN_PUBLIC = f"https://{_public_ip}/admin#session"' in source
    assert 'link_button("Paste request token", ADMIN_PUBLIC' in source
