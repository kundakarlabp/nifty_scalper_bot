"""HTTP control actions report queued vs failed rather than false completion."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nifty_scalper_bot import admin_dashboard as admin
from nifty_scalper_bot.ops.service_control import ControlResult


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def test_restart_returns_queue_evidence_not_immediate_restart(monkeypatch) -> None:
    actions = []
    monkeypatch.setattr(
        admin,
        "queue_validated_restart",
        lambda: actions.append("deployer") or ControlResult(
            True, "restart_bot", "safe restart queued"
        ),
    )
    monkeypatch.setattr(
        admin,
        "_restart_service",
        lambda: (_ for _ in ()).throw(
            AssertionError("unsafe engine restart called")
        ),
    )
    response = _client().post(
        "/admin/restart", headers={"accept": "application/json"}
    )

    assert response.status_code == 202
    assert response.json() == {
        "ok": True,
        "state": "queued",
        "message": "safe restart queued",
    }
    assert actions == ["deployer"]


def test_restart_reports_failed_dispatch_not_success(monkeypatch) -> None:
    monkeypatch.setattr(
        admin,
        "queue_validated_restart",
        lambda: ControlResult(False, "restart_bot", "systemd rejected"),
    )
    response = _client().post(
        "/admin/restart", headers={"accept": "application/json"}
    )
    assert response.status_code == 503
    assert response.json()["ok"] is False
    assert response.json()["state"] == "failed"


def test_browser_restart_response_is_queued_not_completed(monkeypatch) -> None:
    monkeypatch.setattr(
        admin,
        "queue_validated_restart",
        lambda: ControlResult(True, "restart_bot", "scheduled"),
    )
    response = _client().post("/admin/restart", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin?restart=queued"


def test_update_action_does_not_claim_code_was_deployed(monkeypatch) -> None:
    monkeypatch.setattr(
        admin, "_git_update", lambda: (True, "service restart accepted")
    )
    response = _client().post(
        "/admin/update", headers={"accept": "application/json"}
    )
    assert response.status_code == 202
    assert response.json()["state"] == "queued"
    assert "not yet verified" in response.json()["message"]


def test_update_failure_propagates_operator_error(monkeypatch) -> None:
    monkeypatch.setattr(
        admin, "_git_update", lambda: (False, "systemd rejected")
    )
    response = _client().post(
        "/admin/update", headers={"accept": "application/json"}
    )
    assert response.status_code == 503
    assert response.json() == {
        "ok": False,
        "state": "failed",
        "message": "systemd rejected",
    }
