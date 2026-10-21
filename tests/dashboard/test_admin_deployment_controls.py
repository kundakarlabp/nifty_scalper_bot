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
        lambda: actions.append("deployer")
        or ControlResult(True, "restart_bot", "safe restart queued"),
    )
    monkeypatch.setattr(
        admin,
        "_restart_service",
        lambda: (_ for _ in ()).throw(AssertionError("unsafe engine restart called")),
    )
    response = _client().post("/admin/restart", headers={"accept": "application/json"})

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
    response = _client().post("/admin/restart", headers={"accept": "application/json"})
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
    response = _client().post("/admin/update", headers={"accept": "application/json"})
    assert response.status_code == 202
    assert response.json()["state"] == "queued"
    assert "not yet verified" in response.json()["message"]


def test_update_failure_propagates_operator_error(monkeypatch) -> None:
    monkeypatch.setattr(admin, "_git_update", lambda: (False, "systemd rejected"))
    response = _client().post("/admin/update", headers={"accept": "application/json"})
    assert response.status_code == 503
    assert response.json() == {
        "ok": False,
        "state": "failed",
        "message": "systemd rejected",
    }


def test_token_activation_queues_safe_restart_not_direct_systemctl(monkeypatch) -> None:
    writes = []
    queued = []
    monkeypatch.setattr(admin, "_write_env", lambda values: writes.append(values))
    monkeypatch.setattr(
        admin,
        "queue_validated_restart",
        lambda: queued.append(True)
        or ControlResult(True, "restart_bot", "restart queued"),
    )
    result = admin._activate_access_token("verified-token")
    assert result.ok is True
    assert queued == [True]
    assert writes == [
        {
            "KITE_ACCESS_TOKEN": "verified-token",
            "ZERODHA_ACCESS_TOKEN": "verified-token",
            "BROKER_ACCESS_TOKEN": "verified-token",
        }
    ]


def test_verified_token_reports_restart_dispatch_failure(monkeypatch) -> None:
    monkeypatch.setattr(admin, "_read_env", lambda: {"KITE_API_KEY": "key"})
    monkeypatch.setattr(
        admin, "_verify_access_token", lambda *_args: (True, "verified-user")
    )
    monkeypatch.setattr(
        admin,
        "_activate_access_token",
        lambda _token: ControlResult(False, "restart_bot", "systemd rejected"),
    )
    response = _client().post(
        "/admin/token",
        data={"access_token": "new-token"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "token=saved_not_applied" in response.headers["location"]
    assert "systemd%20rejected" in response.headers["location"]


def test_verified_callback_reports_failed_restart(monkeypatch) -> None:
    monkeypatch.setattr(
        admin,
        "_read_env",
        lambda: {"KITE_API_KEY": "key", "KITE_API_SECRET": "secret"},
    )
    monkeypatch.setattr(
        admin,
        "_exchange_request_token",
        lambda *_args: (True, "new-token"),
    )
    monkeypatch.setattr(
        admin, "_verify_access_token", lambda *_args: (True, "verified-user")
    )
    monkeypatch.setattr(
        admin,
        "_activate_access_token",
        lambda _token: ControlResult(False, "restart_bot", "marker unavailable"),
    )
    response = _client().get(
        "/admin/kite/callback?request_token=valid&status=success",
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "token=saved_not_applied" in response.headers["location"]
    assert "marker%20unavailable" in response.headers["location"]


def test_mode_change_is_saved_but_not_claimed_running_on_dispatch_failure(
    monkeypatch,
) -> None:
    writes = []
    monkeypatch.setattr(admin, "_write_env", lambda values: writes.append(values))
    monkeypatch.setattr(
        admin,
        "_restart_service",
        lambda: ControlResult(False, "restart_bot", "systemd unavailable"),
    )
    response = _client().post(
        "/admin/mode", data={"mode": "shadow"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin?mode=saved_not_applied&msg=")
    assert writes == [{"ENABLE_LIVE": "false", "EXECUTION_MODE": "SHADOW"}]


def test_mode_change_success_means_queued_not_restarted(monkeypatch) -> None:
    monkeypatch.setattr(admin, "_write_env", lambda _values: None)
    monkeypatch.setattr(
        admin,
        "_restart_service",
        lambda: ControlResult(True, "restart_bot", "queued"),
    )
    response = _client().post(
        "/admin/mode", data={"mode": "shadow"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/admin?mode=queued"


def test_live_admin_removes_unverified_restart_and_market_time_claims() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    host = (root / "src/nifty_scalper_bot/superlite_admin.py").read_text()
    service = (root / "src/nifty_scalper_bot/ops/service_control.py").read_text()
    dashboard = (root / "src/nifty_scalper_bot/admin_dashboard.py").read_text()

    assert "dashboard._restart_service =" not in host
    assert "outside the protected market window" not in service
    assert '"sudo", "systemctl", "restart"' not in dashboard
    assert "os._exit(0)" not in dashboard
