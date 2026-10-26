"""Production admin home is password-free but cannot operate live trading."""

from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard


def _client(*, https: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(app, base_url="https://65.0.255.84" if https else "http://65.0.255.84")


def _basic(password: str) -> dict[str, str]:
    value = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"authorization": "Basic " + value}


def test_public_admin_status_needs_no_basic_credentials(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "private-test-password")
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("anonymous status must never inspect broker secrets")
        ),
    )
    response = _client().get("/admin")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "www-authenticate" not in response.headers
    assert "No admin login required" in response.text
    assert 'href="/admin/manage"' in response.text
    assert "KITE_API_SECRET" not in response.text
    assert 'action="/admin/mode"' not in response.text
    assert 'action="/admin/token"' not in response.text
    assert "private-test-password" not in response.text


def test_public_admin_status_cannot_fall_back_to_cleartext_http(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    response = _client(https=False).get("/admin")
    assert response.status_code == 403
    assert response.json()["detail"] == "admin_https_required"


def test_operator_controls_still_require_authentication(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "private-test-password")
    client = _client()
    for path in ("/admin/manage", "/admin/api/logs", "/admin/trades.json"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Basic")
    response = client.post("/admin/mode", data={"mode": "live"})
    assert response.status_code == 401


def test_authenticated_controls_remain_available(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "private-test-password")
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: {"ENABLE_LIVE": "false", "EXECUTION_MODE": "SHADOW"},
    )
    response = _client().get("/admin/manage", headers=_basic("private-test-password"))
    assert response.status_code == 200
    assert "Bot administration" in response.text
    assert 'action="/admin/mode"' in response.text
    assert "private-test-password" not in response.text
