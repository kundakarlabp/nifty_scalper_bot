"""The only passwordless operator path proves the allowlisted Zerodha owner."""

from __future__ import annotations

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot.ops.service_control import ControlResult


def _setup(monkeypatch, tmp_path, *, owner: str = "AB1234") -> TestClient:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: {
            "ADMIN_ZERODHA_USER_ID": owner,
            "KITE_API_KEY": "public-test-key",
            "KITE_API_SECRET": "private-test-secret",
            "ENABLE_LIVE": "false",
            "EXECUTION_MODE": "SHADOW",
        },
    )
    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(app, base_url="https://65.0.255.84")


def _start(client: TestClient) -> str:
    response = client.get("/admin/kite/start")
    assert response.status_code == 200
    assert response.cookies.get("nifty_owner_login")
    assert "secure" in response.headers["set-cookie"].lower()
    assert "httponly" in response.headers["set-cookie"].lower()
    challenge = re.search(r"name='owner_state' value='([^']+)'", response.text)
    assert challenge is not None
    return challenge.group(1)


def test_zerodha_owner_session_replaces_separate_admin_prompt(
    monkeypatch, tmp_path
) -> None:
    client = _setup(monkeypatch, tmp_path)
    login = client.get("/admin/manage", follow_redirects=False)
    assert login.status_code == 303
    assert login.headers["location"] == "/admin/kite/start"
    state = _start(client)
    monkeypatch.setattr(
        dashboard, "_exchange_request_token", lambda *_: (True, "access-secret")
    )
    monkeypatch.setattr(dashboard, "_verify_access_token", lambda *_: (True, "AB1234"))
    actions: list[str] = []
    monkeypatch.setattr(
        dashboard,
        "_activate_access_token",
        lambda token: (
            actions.append(token) or ControlResult(True, "restart_bot", "queued")
        ),
    )
    result = client.post(
        "/admin/kite/complete",
        data={"owner_state": state, "request_token": "one-time-request"},
        headers={"origin": "https://65.0.255.84"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert result.headers["location"].startswith("/admin/manage")
    assert actions == ["access-secret"]
    assert result.cookies.get("nifty_owner_session")
    owner = client.get("/admin/manage")
    assert owner.status_code == 200
    assert "Bot administration" in owner.text
    assert "www-authenticate" not in owner.headers


def test_different_zerodha_account_never_activates_token(monkeypatch, tmp_path) -> None:
    client = _setup(monkeypatch, tmp_path)
    state = _start(client)
    monkeypatch.setattr(
        dashboard, "_exchange_request_token", lambda *_: (True, "other-account")
    )
    monkeypatch.setattr(
        dashboard, "_verify_access_token", lambda *_: (True, "NOTOWNER")
    )
    monkeypatch.setattr(
        dashboard,
        "_activate_access_token",
        lambda *_: (_ for _ in ()).throw(AssertionError("not the owner")),
    )
    result = client.post(
        "/admin/kite/complete",
        data={"owner_state": state, "request_token": "request"},
        headers={"origin": "https://65.0.255.84"},
    )
    assert result.status_code == 403
    assert client.get("/admin/manage", follow_redirects=False).status_code == 303


def test_missing_challenge_or_cross_origin_cannot_rotate(monkeypatch, tmp_path) -> None:
    client = _setup(monkeypatch, tmp_path)
    _start(client)
    for state, origin in (
        ("forged", "https://65.0.255.84"),
        ("forged", "https://evil.invalid"),
    ):
        response = client.post(
            "/admin/kite/complete",
            data={"owner_state": state, "request_token": "request"},
            headers={"origin": origin},
        )
        assert response.status_code == 403


def test_no_owner_configuration_fails_closed(monkeypatch, tmp_path) -> None:
    client = _setup(monkeypatch, tmp_path, owner="")
    result = client.get("/admin/kite/start")
    assert result.status_code == 503
    assert "not configured" in result.text.lower()


def test_http_never_allows_owner_login_or_mutation(monkeypatch, tmp_path) -> None:
    _setup(monkeypatch, tmp_path)
    app = FastAPI()
    app.include_router(dashboard.router)
    client = TestClient(app, base_url="http://65.0.255.84")
    assert client.get("/admin/kite/start").status_code == 403
    assert (
        client.post(
            "/admin/kite/complete",
            data={"owner_state": "forged", "request_token": "request"},
        ).status_code
        == 403
    )
    assert client.get("/admin/manage").status_code == 403
