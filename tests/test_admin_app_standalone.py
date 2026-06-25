"""The standalone admin app must serve the dashboard routes end-to-end and stay
isolated from the trading engine (it must not import core.app / the engine)."""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from nifty_scalper_bot.admin_app import app

_client = TestClient(app)


async def test_healthz_independent_of_engine() -> None:
    r = _client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_root_redirects_to_admin() -> None:
    r = _client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/admin"


async def test_no_login_page_password_removed() -> None:
    # Password protection removed — there is no login route anymore.
    r = _client.get("/admin/login", follow_redirects=False)
    assert r.status_code == 404


async def test_admin_loads_without_auth() -> None:
    # /admin serves directly (no redirect to a login page).
    r = _client.get("/admin", follow_redirects=False)
    assert r.status_code == 200
    assert "Download Logs" in r.text
    assert "Sign in" not in r.text


async def test_log_viewer_removed_download_present() -> None:
    # The polling HTML log viewer and logs.json are gone (super-lite); only the
    # download-on-click endpoint remains.
    assert _client.get("/admin/logs", follow_redirects=False).status_code == 404
    assert _client.get("/admin/logs.json", follow_redirects=False).status_code == 404
    assert _client.get("/admin/logs/download", follow_redirects=False).status_code == 200


async def test_admin_app_does_not_import_engine_in_source() -> None:
    src = pathlib.Path("src/nifty_scalper_bot/admin_app.py").read_text()
    assert "core.app" not in src
    assert "run_bot_background" not in src
    assert "strategies" not in src
