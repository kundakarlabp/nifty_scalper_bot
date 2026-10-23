"""Regression tests for the mobile-first independent admin control plane."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot import superlite_admin_core as core


def client() -> TestClient:
    app = FastAPI()
    app.include_router(dashboard.router)
    return TestClient(app)


def test_mobile_dashboard_loads_without_expensive_live_probes(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: {
            "KITE_API_KEY": "my-key",
            "KITE_API_SECRET": "not-for-html",
            "KITE_ACCESS_TOKEN": "secret-token",
            "ENABLE_LIVE": "false",
            "EXECUTION_MODE": "SHADOW",
        },
    )
    monkeypatch.setattr(
        dashboard,
        "quick_overview",
        lambda: (_ for _ in ()).throw(AssertionError("blocked page render")),
    )
    body = client().get("/admin")
    assert body.status_code == 200
    assert body.headers["cache-control"] == "no-store"
    assert 'name="viewport"' in body.text
    assert 'id="overview"' in body.text
    assert 'id="session"' in body.text
    assert 'id="logs"' in body.text
    assert 'id="controls"' in body.text
    assert 'id="engine-value"' in body.text
    assert "Open official Zerodha login" in body.text
    assert "Download Logs" in body.text
    assert "<details>" in body.text
    assert "not-for-html" not in body.text
    assert "secret-token" not in body.text
    # Inline script runs after the DOM so it can initialize on mobile.
    assert body.text.index('id="engine-value"') < body.text.index("<script>")
    assert 'id="refresh-status"' in body.text
    assert "document.hidden" in body.text
    assert "textContent" in body.text


def test_overview_uses_live_engine_evidence_and_caches_it(monkeypatch) -> None:
    requests = []

    def engine(path):
        requests.append(path)
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {
                "broker_authentication": "authenticated",
                "primary_blocker": "market_closed",
                "build_sha": "abcdef123456",
            }
        if path == "/trading/status":
            return {"execution_mode": "SHADOW"}
        raise AssertionError(path)

    monkeypatch.setattr(core, "_http_json", engine)
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "deployed"})

    def no_git(*_args):
        raise AssertionError("no git in heartbeat")

    monkeypatch.setattr(core, "_git_ref", no_git)

    def no_journal(*_args, **_kwargs):
        raise AssertionError("no journal in heartbeat")

    monkeypatch.setattr(core, "bounded_logs", no_journal)
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})
    result = core.quick_overview()
    assert result["engine_ok"] is True
    assert result["bot_loaded"] is True
    assert result["broker_authenticated"] == "authenticated"
    assert result["primary_blocker"] == "market_closed"
    assert result["mode"] == "SHADOW"
    assert result["runtime_sha"] == "abcdef123456"
    assert result["updater_state"] == "deployed"
    assert isinstance(result["checked_at"], int)
    assert core.quick_overview() == result
    assert requests == ["/livez", "/health/trading", "/trading/status"]
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})


def test_timeout_never_looks_like_ready(monkeypatch) -> None:
    def engine(path):
        if path == "/livez":
            return {"_error": "ENGINE HTTP TIMEOUT"}
        return {"broker_authentication": "authenticated", "execution_mode": "LIVE"}

    monkeypatch.setattr(core, "_http_json", engine)
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})
    result = core.quick_overview()
    assert result["engine_ok"] is False
    assert result["bot_loaded"] is False
    assert result["broker_authenticated"] == "unknown"
    assert result["mode"] == "UNKNOWN"
    assert result["primary_blocker"] == "Engine unreachable"
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})


def test_overview_and_log_api_are_explicit_no_store(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "quick_overview", lambda: {"engine_ok": True})
    read = []
    monkeypatch.setattr(
        dashboard,
        "bounded_logs",
        lambda lines, needle: read.append((lines, needle)) or "redacted log",
    )
    app = client()
    overview = app.get("/admin/api/overview")
    assert overview.status_code == 200
    assert overview.json() == {"engine_ok": True}
    assert overview.headers["cache-control"] == "no-store"
    logs = app.get("/admin/api/logs?lines=9999&contains=" + "z" * 120)
    assert logs.status_code == 200
    assert logs.json() == {"text": "redacted log"}
    assert logs.headers["cache-control"] == "no-store"
    assert read == [(500, "z" * 80)]


def test_log_payload_is_bounded_before_browser_delivery(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "bounded_logs", lambda *_: "X" * 200_000)
    result = client().get("/admin/api/logs?lines=1")
    assert result.status_code == 200
    assert len(result.json()["text"]) == 100_000


def test_quiet_mode_reports_planned_idle_not_engine_failure(monkeypatch) -> None:
    responses = {
        "/livez": {
            "quiet": True,
            "bot_loaded": False,
            "operating_mode": "AUTO",
        },
        "/health/trading": {"_error": "HTTP 404"},
        "/trading/status": {"_error": "HTTP 404"},
    }
    monkeypatch.setattr(core, "_http_json", lambda path: responses[path])
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})
    view = core.quick_overview()
    assert view["engine_ok"] is True
    assert view["quiet"] is True
    assert view["bot_loaded"] is False
    assert view["mode"] == "AUTO"
    assert view["primary_blocker"] == "auto_quiet"
    assert view["broker_authenticated"] == "unknown"
    core.OVERVIEW_CACHE.update({"at": 0.0, "data": {}})


def test_admin_login_button_accepts_canonical_broker_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_read_env",
        lambda: {
            "BROKER_API_KEY": "broker-key",
            "ENABLE_LIVE": "true",
            "EXECUTION_MODE": "LIVE",
        },
    )
    response = client().get("/admin")
    assert response.status_code == 200
    assert "Open official Zerodha login" in response.text
    assert "api_key=broker-key" in response.text
