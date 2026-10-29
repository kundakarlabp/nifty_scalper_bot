"""The only passwordless operator path proves the allowlisted Zerodha owner."""

from __future__ import annotations

import re

import pytest
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
    response = client.get("/admin")
    assert response.status_code == 200
    assert response.cookies.get("nifty_owner_login")
    assert "secure" in response.headers["set-cookie"].lower()
    assert "httponly" in response.headers["set-cookie"].lower()
    challenge = re.search(r'name="owner_state" value="([^"]+)"', response.text)
    assert challenge is not None
    assert 'action="/admin/token"' in response.text
    return challenge.group(1)


def test_original_dashboard_token_rotates_and_unlocks_controls(
    monkeypatch, tmp_path
) -> None:
    client = _setup(monkeypatch, tmp_path)
    login = client.get("/admin/manage", follow_redirects=False)
    assert login.status_code == 200
    assert 'action="/admin/token"' in login.text
    assert 'action="/admin/mode" inert' in login.text
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
        "/admin/token",
        data={"owner_state": state, "request_token": "one-time-request"},
        headers={"origin": "https://65.0.255.84"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert result.headers["location"].startswith("/admin?token=ok")
    assert actions == ["access-secret"]
    assert result.cookies.get("nifty_owner_session")
    owner = client.get("/admin")
    assert owner.status_code == 200
    assert "Bot administration" in owner.text
    assert 'action="/admin/mode" inert' not in owner.text
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
        "/admin/token",
        data={"owner_state": state, "request_token": "request"},
        headers={"origin": "https://65.0.255.84"},
    )
    assert result.status_code == 403
    assert client.get("/admin/manage", follow_redirects=False).status_code == 200


def test_missing_challenge_or_cross_origin_cannot_rotate(monkeypatch, tmp_path) -> None:
    client = _setup(monkeypatch, tmp_path)
    _start(client)
    for state, origin in (
        ("forged", "https://65.0.255.84"),
        ("forged", "https://evil.invalid"),
    ):
        response = client.post(
            "/admin/token",
            data={"owner_state": state, "request_token": "request"},
            headers={"origin": origin},
        )
        assert response.status_code == 403


def test_no_owner_configuration_fails_closed(monkeypatch, tmp_path) -> None:
    client = _setup(monkeypatch, tmp_path, owner="")
    result = client.get("/admin")
    assert result.status_code == 503
    assert result.json()["detail"] == "admin_owner_not_configured"


def test_http_never_allows_owner_login_or_mutation(monkeypatch, tmp_path) -> None:
    _setup(monkeypatch, tmp_path)
    app = FastAPI()
    app.include_router(dashboard.router)
    client = TestClient(app, base_url="http://65.0.255.84")
    assert client.get("/admin/kite/start").status_code == 403
    assert (
        client.post(
            "/admin/token",
            data={"owner_state": "forged", "request_token": "request"},
        ).status_code
        == 403
    )
    assert client.get("/admin/manage").status_code == 403


@pytest.mark.parametrize(
    ("source_headers", "allowed"),
    [
        ({}, True),  # Mobile browser with no Origin and no Referer
        ({"origin": "null"}, True),  # Opaque origin with signed challenge
        ({"referer": "https://65.0.255.84/admin"}, True),
        ({"origin": "https://65.0.255.84:443"}, True),
        ({"origin": "https://evil.invalid"}, False),
        ({"referer": "https://evil.invalid/form"}, False),
        ({"origin": "http://65.0.255.84"}, False),
        ({"origin": "https://65.0.255.84:8443"}, False),
        (
            {
                "origin": "https://65.0.255.84",
                "referer": "https://evil.invalid/form",
            },
            False,
        ),
    ],
)
def test_mobile_owner_paste_origin_policy(
    monkeypatch, tmp_path, source_headers, allowed
) -> None:
    client = _setup(monkeypatch, tmp_path)
    state = _start(client)
    exchanged = []
    activated = []
    monkeypatch.setattr(
        dashboard,
        "_exchange_request_token",
        lambda *_: exchanged.append(True) or (True, "fresh-access"),
    )
    monkeypatch.setattr(dashboard, "_verify_access_token", lambda *_: (True, "AB1234"))
    monkeypatch.setattr(
        dashboard,
        "_activate_access_token",
        lambda token: activated.append(token)
        or ControlResult(True, "restart_bot", "queued"),
    )
    response = client.post(
        "/admin/token",
        data={"owner_state": state, "request_token": "fresh-request"},
        headers=source_headers,
        follow_redirects=False,
    )
    assert response.status_code == (303 if allowed else 403)
    assert len(exchanged) == len(activated) == (1 if allowed else 0)
    if allowed:
        assert response.headers["location"] == "/admin?token=ok"
    else:
        assert response.json()["detail"] == "admin_same_origin_required"


def test_missing_origin_still_requires_valid_cookie_bound_challenge(
    monkeypatch, tmp_path
) -> None:
    client = _setup(monkeypatch, tmp_path)
    _start(client)
    monkeypatch.setattr(
        dashboard,
        "_exchange_request_token",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not exchange")),
    )
    response = client.post(
        "/admin/token",
        data={"owner_state": "forged", "request_token": "fresh-request"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "owner_login_challenge_invalid"


def test_mobile_privacy_policy_and_secondary_login_submit(
    monkeypatch, tmp_path
) -> None:
    client = _setup(monkeypatch, tmp_path)
    page = client.get("/admin")
    assert page.headers["referrer-policy"] == "same-origin"
    state = re.search(r'name="owner_state" value="([^"]+)"', page.text)
    assert state is not None
    monkeypatch.setattr(
        dashboard, "_exchange_request_token", lambda *_: (True, "fresh-access")
    )
    monkeypatch.setattr(dashboard, "_verify_access_token", lambda *_: (True, "AB1234"))
    monkeypatch.setattr(
        dashboard,
        "_activate_access_token",
        lambda *_: ControlResult(True, "restart_bot", "queued"),
    )
    result = client.post(
        "/admin/kite/complete",
        data={"owner_state": state.group(1), "request_token": "fresh-request"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert result.headers["location"] == "/admin?token=ok"
