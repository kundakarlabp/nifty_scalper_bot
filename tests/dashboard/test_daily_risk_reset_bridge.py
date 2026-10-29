"""Regression for authenticated admin → engine-owned intraday risk resets."""

from __future__ import annotations

import importlib
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nifty_scalper_bot import admin_dashboard, admin_owner_auth
from nifty_scalper_bot.risk.limits import RiskSwitches

engine = importlib.import_module("nifty_scalper_bot.main")


def _request(method: str, token: str = "", peer: str = "127.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "scheme": "http",
            "path": "/internal/admin/risk/reset-day",
            "headers": [
                (b"x-nifty-internal-command", token.encode("ascii")),
            ],
            "server": ("127.0.0.1", 8080),
            "client": (peer, 34000),
        }
    )


def _context(*, positions=None, orders=None, reconciled=True):
    switches = RiskSwitches(
        max_day_loss=700.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-640.0)
    manager = SimpleNamespace(
        get_all_positions=lambda: list(positions or []),
        get_pending_orders=lambda: list(orders or []),
        _trading_date_ist=lambda: "2026-10-29",
    )
    risk = SimpleNamespace(
        _switches=switches,
        _operator_reset_trading_date="",
        position_manager=manager,
        _breaker_tripped=False,
    )
    seen: list[bool] = []

    def reset():
        seen.append(True)
        risk._operator_reset_trading_date = "2026-10-29"
        switches.rebase_day_pnl()
        return {"prior_day_loss": 640.0, "new_day_loss": 0.0}

    risk.operator_reset_daily_loss_budget = reset
    ctx = SimpleNamespace(
        risk_manager=risk,
        position_reconciliation_completed=reconciled,
        position_reconciliation_failed=False,
        unprotected_broker_positions=set(),
    )
    return ctx, seen


def test_signed_local_admin_bridge_rebases_live_engine_once(monkeypatch):
    monkeypatch.setattr(admin_owner_auth, "_signing_key", lambda: b"K" * 32)
    ctx, calls = _context()
    monkeypatch.setattr(engine, "_latest_context", lambda: ctx)

    class Response:
        def __init__(self, value):
            self.body = BytesIO(json.dumps(value).encode("utf-8"))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            return self.body.read(limit)

    def local_urlopen(req, timeout):
        assert req.full_url == "http://127.0.0.1:8080/internal/admin/risk/reset-day"
        assert req.get_method() == "POST"
        assert timeout <= 3
        token = req.get_header("X-nifty-internal-command")
        return Response(engine.local_risk_reset(_request("POST", token)))

    monkeypatch.setattr(admin_dashboard.urllib.request, "urlopen", local_urlopen)
    result = admin_dashboard._engine_risk_request("risk_reset")
    assert result["state"] == "applied"
    assert result["risk"]["reset_applied_today"] is True
    assert result["risk"]["remaining_daily_risk_budget"] == 700.0
    assert len(calls) == 1


def test_remote_and_unsigned_requests_cannot_rebase(monkeypatch):
    monkeypatch.setattr(admin_owner_auth, "_signing_key", lambda: b"K" * 32)
    ctx, calls = _context()
    monkeypatch.setattr(engine, "_latest_context", lambda: ctx)
    token = admin_owner_auth.issue_local_engine_command("risk_reset")
    for request in (
        _request("POST", token, peer="8.8.8.8"),
        _request("POST"),
        _request("POST", admin_owner_auth.issue_local_engine_command("risk_status")),
        _request("POST", token + "tampered"),
    ):
        with pytest.raises(HTTPException) as error:
            engine.local_risk_reset(request)
        assert error.value.status_code == 403
    assert calls == []


@pytest.mark.parametrize(
    ("positions", "orders", "reconciled", "reason"),
    [
        (
            [SimpleNamespace(symbol="NFO:NIFTY")],
            None,
            True,
            "broker_positions_or_orders_open",
        ),
        (
            None,
            [SimpleNamespace(status="OPEN")],
            True,
            "broker_positions_or_orders_open",
        ),
        (None, None, False, "broker_reconciliation_unverified"),
    ],
)
def test_engine_never_resets_with_unknown_broker_exposure(
    monkeypatch, positions, orders, reconciled, reason
):
    monkeypatch.setattr(admin_owner_auth, "_signing_key", lambda: b"K" * 32)
    ctx, calls = _context(
        positions=positions, orders=orders, reconciled=reconciled
    )
    monkeypatch.setattr(engine, "_latest_context", lambda: ctx)
    token = admin_owner_auth.issue_local_engine_command("risk_reset")
    with pytest.raises(HTTPException) as error:
        engine.local_risk_reset(_request("POST", token))
    assert error.value.status_code == 409
    assert error.value.detail == reason
    assert not calls


def test_owner_dashboard_never_reports_success_for_engine_failure(monkeypatch):
    monkeypatch.setattr(admin_dashboard, "_check_auth", lambda _request: None)

    def fail(_action):
        raise RuntimeError("engine_risk_request_unavailable")

    monkeypatch.setattr(admin_dashboard, "_engine_risk_request", fail)
    response = admin_dashboard.reset_daily_risk(_request("POST"), confirm="RESET")
    assert response.status_code == 303
    assert "riskreset=err" in response.headers["location"]
    assert "riskreset=ok" not in response.headers["location"]


def test_owner_dashboard_shows_real_budget_only_after_auth(monkeypatch):
    monkeypatch.setattr(
        admin_dashboard,
        "_engine_risk_request",
        lambda action: {
            "daily_loss_cap": 700.0,
            "day_loss_since_reset": 640.0,
            "remaining_daily_risk_budget": 60.0,
            "reset_applied_today": False,
        },
    )
    assert "₹60.00" in admin_dashboard._daily_risk_panel(True)
    assert "₹700.00" in admin_dashboard._daily_risk_panel(True)
    assert "₹60.00" not in admin_dashboard._daily_risk_panel(False)
