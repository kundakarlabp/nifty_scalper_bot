"""Independent admin mode controls and credential writes in QUIET state.

The production admin service is :8081. These tests must not import or start
the trading engine, and must not perform real broker operations.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as dashboard
from nifty_scalper_bot import operating_control as power
from nifty_scalper_bot.superlite_admin import app

IST = ZoneInfo("Asia/Kolkata")


def _owner(monkeypatch, *, verified: bool, mode: str = "AUTO"):
    env = {
        "ADMIN_ZERODHA_USER_ID": "AB1234",
        "BOT_OPERATING_MODE": mode,
        "BOT_AUTO_START_IST": "08:55",
        "BOT_AUTO_STOP_IST": "15:40",
        "KITE_API_KEY": "example-key",
        "KITE_API_SECRET": "example-secret",
        "EXECUTION_MODE": "SHADOW",
        "ENABLE_LIVE": "false",
    }
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setattr(dashboard, "_read_env", lambda: dict(env))
    monkeypatch.setattr(power, "_read_env", lambda: dict(env))
    monkeypatch.setattr(dashboard, "valid_owner_session", lambda *_: verified)
    return env


def _client():
    return TestClient(app, base_url="https://65.0.255.84")


def test_admin_mode_buttons_are_visible_without_engine_or_broker(monkeypatch):
    _owner(monkeypatch, verified=True, mode="AUTO")
    page = _client().get("/admin")
    assert page.status_code == 200
    assert "Operating mode" in page.text
    assert "Schedule target now" in page.text
    assert "AUTO" in page.text
    assert "ACTIVE" in page.text
    assert "QUIET" in page.text
    assert page.text.count('action="/admin/power/mode"') == 3
    assert 'id="settings-request-token"' in page.text
    assert 'action="/admin/save"' in page.text
    assert "API Secret" in page.text
    assert "API Key" in page.text


def test_guest_sees_mode_but_cannot_change_settings(monkeypatch, tmp_path):
    _owner(monkeypatch, verified=False)
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    with _client() as client:
        page = client.get("/admin")
        assert page.status_code == 200
        assert 'action="/admin/power/mode"' in page.text
        assert 'value="ACTIVE"' in page.text
        assert 'type="submit" disabled' in page.text
        result = client.post(
            "/admin/power/mode",
            data={"mode": "ACTIVE"},
            headers={"origin": "https://65.0.255.84"},
            follow_redirects=False,
        )
        assert result.status_code == 401


def test_owner_switches_modes_from_independent_admin(monkeypatch):
    env = _owner(monkeypatch, verified=True, mode="AUTO")
    saves = []
    queued = []

    def save(updates):
        saves.append(dict(updates))
        env.update(updates)

    monkeypatch.setattr(power, "_write_env", save)
    monkeypatch.setattr(
        power,
        "_restart_service",
        lambda: queued.append("guarded") or type("Result", (), {"ok": True})(),
    )
    with _client() as client:
        response = client.post(
            "/admin/power/mode",
            data={"mode": "QUIET"},
            headers={"origin": "https://65.0.255.84"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/admin?power=queued"
        assert saves[-1] == {
            "BOT_OPERATING_MODE": "QUIET",
            "BOT_AUTO_START_IST": "08:55",
            "BOT_AUTO_STOP_IST": "15:40",
        }
        assert queued == ["guarded"]
        assert "QUIET" in client.get("/admin").text
        response = client.post(
            "/admin/power/mode",
            data={"mode": "AUTO"},
            headers={"origin": "https://65.0.255.84"},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/admin?power=queued"
        assert saves[-1]["BOT_OPERATING_MODE"] == "AUTO"
        assert queued == ["guarded", "guarded"]


def test_credentials_are_editable_when_quiet(monkeypatch):
    _owner(monkeypatch, verified=True, mode="QUIET")
    saved = []
    monkeypatch.setattr(dashboard, "_write_env", lambda values: saved.append(values))
    with _client() as client:
        page = client.get("/admin")
        assert page.status_code == 200
        assert 'action="/admin/save"' in page.text
        assert 'name="KITE_API_KEY"' in page.text
        assert 'name="KITE_API_SECRET"' in page.text
        assert 'name="request_token"' in page.text
        result = client.post(
            "/admin/save",
            data={"KITE_API_KEY": "new-key", "KITE_API_SECRET": "new-secret"},
            headers={"origin": "https://65.0.255.84"},
            follow_redirects=False,
        )
        assert result.status_code == 303
        assert result.headers["location"] == "/admin?saved=1"
        assert saved[-1]["BROKER_API_KEY"] == "new-key"
        assert saved[-1]["BROKER_API_SECRET"] == "new-secret"


def test_auto_weekday_market_window_and_manual_overrides(monkeypatch):
    env = _owner(monkeypatch, verified=True, mode="AUTO")
    assert power.engine_should_run(datetime(2026, 10, 27, 8, 54, tzinfo=IST)) is False
    assert power.engine_should_run(datetime(2026, 10, 27, 8, 55, tzinfo=IST)) is True
    assert power.engine_should_run(datetime(2026, 10, 27, 15, 39, tzinfo=IST)) is True
    assert power.engine_should_run(datetime(2026, 10, 27, 15, 40, tzinfo=IST)) is False
    assert power.engine_should_run(datetime(2026, 10, 25, 11, 0, tzinfo=IST)) is False
    env["BOT_OPERATING_MODE"] = "ACTIVE"
    assert power.engine_should_run(datetime(2026, 10, 25, 11, 0, tzinfo=IST)) is True
    env["BOT_OPERATING_MODE"] = "QUIET"
    assert power.engine_should_run(datetime(2026, 10, 27, 11, 0, tzinfo=IST)) is False


def test_power_mode_does_not_change_live_order_mode(monkeypatch):
    env = _owner(monkeypatch, verified=True, mode="AUTO")
    saves = []

    def save(values):
        saves.append(values)
        env.update(values)

    monkeypatch.setattr(power, "_write_env", save)
    monkeypatch.setattr(
        power,
        "_restart_service",
        lambda: type("Result", (), {"ok": False})(),
    )
    result = _client().post(
        "/admin/power/mode",
        data={"mode": "ACTIVE"},
        headers={"origin": "https://65.0.255.84"},
        follow_redirects=False,
    )
    assert result.status_code == 303
    assert result.headers["location"] == "/admin?power=saved_not_applied"
    assert not any("ENABLE_LIVE" in item or "EXECUTION_MODE" in item for item in saves)
