"""One admin URL: redirect cleartext to HTTPS, then use the Zerodha owner session.

Public status is still available through /livez and /releasez without
exposing real-money operator controls or a redundant HTTP Basic prompt.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard


def _client(*, https: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(
        app, base_url="https://65.0.255.84" if https else "http://65.0.255.84"
    )


def _owner(monkeypatch, *, authenticated: bool) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: {
            "ADMIN_ZERODHA_USER_ID": "AB1234",
            "ENABLE_LIVE": "false",
            "EXECUTION_MODE": "SHADOW",
        },
    )
    monkeypatch.setattr(dashboard, "valid_owner_session", lambda *_: authenticated)


def test_http_bookmark_redirects_to_secure_admin_without_basic(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    response = _client(https=False).get("/admin", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://65.0.255.84/admin"
    assert response.headers["cache-control"] == "no-store"
    assert "www-authenticate" not in response.headers
    assert _client(https=False).get("/admin/manage").status_code == 403


def test_https_admin_entry_routes_directly_to_owner_login(monkeypatch) -> None:
    _owner(monkeypatch, authenticated=False)
    client = _client()
    for path in ("/admin", "/admin/manage"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/kite/start"
        assert "www-authenticate" not in response.headers
    for path in ("/admin/api/logs", "/admin/trades.json"):
        response = client.get(path)
        assert response.status_code == 401
        assert "www-authenticate" not in response.headers
    response = client.post("/admin/mode", data={"mode": "live"})
    assert response.status_code == 401


def test_signed_owner_gets_existing_full_dashboard_without_admin_password(
    monkeypatch,
) -> None:
    _owner(monkeypatch, authenticated=True)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    response = _client().get("/admin")
    assert response.status_code == 200
    assert "Bot administration" in response.text
    assert 'action="/admin/token"' in response.text
    assert 'action="/admin/mode"' in response.text
    assert 'action="/admin/save"' in response.text
    assert "www-authenticate" not in response.headers


def test_existing_password_cannot_bypass_zerodha_ownership(monkeypatch) -> None:
    _owner(monkeypatch, authenticated=False)
    monkeypatch.setenv("ADMIN_PASSWORD", "obsolete")
    response = _client().get(
        "/admin/manage",
        headers={"authorization": "Basic YWRtaW46b2Jzb2xldGU="},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/kite/start"
    assert "www-authenticate" not in response.headers
