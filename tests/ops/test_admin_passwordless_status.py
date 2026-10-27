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


def _owner(monkeypatch, *, authenticated: bool, tmp_path=None) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    if tmp_path is not None:
        monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
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


def test_https_admin_opens_dashboard_without_extra_login_page(
    monkeypatch, tmp_path
) -> None:
    _owner(monkeypatch, authenticated=False, tmp_path=tmp_path)
    client = _client()
    for path in ("/admin", "/admin/manage"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 200
        assert "Bot administration" in response.text
        assert 'action="/admin/token"' in response.text
        assert 'name="owner_state"' in response.text
        assert "<details open><summary>Paste callback URL" in response.text
        assert 'action="/admin/mode" inert' in response.text
        assert "www-authenticate" not in response.headers
    for path in ("/admin/api/logs", "/admin/trades.json"):
        response = client.get(path)
        assert response.status_code == 401
        assert "www-authenticate" not in response.headers
    response = client.post("/admin/mode", data={"mode": "live"})
    assert response.status_code == 401


def test_unverified_browser_sees_real_engine_and_broker_status(
    monkeypatch, tmp_path
) -> None:
    _owner(monkeypatch, authenticated=False, tmp_path=tmp_path)
    probe = {
        "engine_ok": True,
        "bot_loaded": True,
        "quiet": False,
        "broker_authenticated": "authenticated",
        "mode": "LIVE",
        "primary_blocker": "",
        "runtime_sha": "ea18b5da485954e05f3a697bd58ce5dce6900ed3",
        "updater_state": "current",
        "checked_at": 1793000000,
    }
    monkeypatch.setattr(dashboard, "quick_overview", lambda: dict(probe))
    response = _client().get("/admin/api/overview")
    assert response.status_code == 200
    assert response.json() == probe
    assert response.headers["cache-control"] == "no-store"
    assert "www-authenticate" not in response.headers
    # The same unverified browser cannot fetch journals or change LIVE mode.
    assert _client().get("/admin/api/logs").status_code == 401
    assert _client().post("/admin/mode", data={"mode": "shadow"}).status_code == 401


def test_service_status_link_is_read_only_without_owner_cookie(
    monkeypatch, tmp_path
) -> None:
    _owner(monkeypatch, authenticated=False, tmp_path=tmp_path)
    monkeypatch.setattr(
        dashboard,
        "quick_overview",
        lambda: {"engine_ok": True, "quiet": False, "bot_loaded": True},
    )
    dashboard._STATUS_CACHE.update({"at": 0.0, "label": "", "color": ""})
    response = _client().get("/admin/status.json")
    assert response.status_code == 200
    assert response.json()["label"] == "operational"
    assert "www-authenticate" not in response.headers
    assert set(response.json()) == {"label", "color"}


def test_public_engine_port_cannot_fetch_secure_admin_overview(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    response = _client(https=False).get("/admin/api/overview")
    assert response.status_code == 403


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


def test_existing_password_cannot_bypass_zerodha_ownership(
    monkeypatch, tmp_path
) -> None:
    _owner(monkeypatch, authenticated=False, tmp_path=tmp_path)
    monkeypatch.setenv("ADMIN_PASSWORD", "obsolete")
    response = _client().get(
        "/admin/manage",
        headers={"authorization": "Basic YWRtaW46b2Jzb2xldGU="},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert "Bot administration" in response.text
    assert "www-authenticate" not in response.headers
