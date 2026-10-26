"""Regression contracts for HTTPS owner controls and safe ingress synchronization."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nifty_scalper_bot import admin_dashboard

ROOT = Path(__file__).resolve().parents[2]


def _request(
    *,
    scheme: str = "https",
    method: str = "GET",
    peer: str = "203.0.113.5",
    authorization: str = "",
    origin: str = "",
    forwarded_for: str = "",
) -> Request:
    headers = [(b"host", b"65.0.255.84")]
    if authorization:
        headers.append((b"authorization", authorization.encode()))
    if origin:
        headers.append((b"origin", origin.encode()))
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "scheme": scheme,
            "method": method,
            "path": "/admin",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 54321),
            "server": ("65.0.255.84", 443),
        }
    )


def _owner(monkeypatch, verified: bool) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setattr(
        admin_dashboard, "_read_env", lambda: {"ADMIN_ZERODHA_USER_ID": "AB1234"}
    )
    monkeypatch.setattr(
        admin_dashboard, "valid_owner_session", lambda *_: verified
    )


@pytest.mark.parametrize(
    ("case_request", "status"),
    [
        (_request(scheme="http"), 403),
        (_request(), 401),
        (_request(authorization="Basic YWRtaW46b2Jzb2xldGU="), 401),
        (_request(method="POST", origin="https://evil.example"), 401),
    ],
)
def test_public_controls_require_verified_owner(monkeypatch, case_request, status):
    _owner(monkeypatch, verified=False)
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(case_request)
    assert failed.value.status_code == status
    assert "WWW-Authenticate" not in (failed.value.headers or {})


def test_verified_owner_controls_require_same_origin_for_writes(monkeypatch):
    _owner(monkeypatch, verified=True)
    admin_dashboard._check_auth(_request())
    admin_dashboard._check_auth(
        _request(method="POST", origin="https://65.0.255.84")
    )
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(
            _request(method="POST", origin="https://evil.example")
        )
    assert failed.value.status_code == 403


def test_missing_owner_fails_closed_without_basic_prompt(monkeypatch):
    _owner(monkeypatch, verified=False)
    monkeypatch.setattr(admin_dashboard, "_read_env", lambda: {})
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(_request())
    assert failed.value.status_code == 503
    assert "WWW-Authenticate" not in (failed.value.headers or {})


def test_local_diagnostics_need_no_browser_cookie(monkeypatch):
    _owner(monkeypatch, verified=False)
    admin_dashboard._check_auth(_request(scheme="http", peer="127.0.0.1"))


def test_forwarded_public_client_cannot_skip_owner_session(monkeypatch):
    _owner(monkeypatch, verified=False)
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(
            _request(
                scheme="https", peer="127.0.0.1",
                forwarded_for="203.0.113.5",
            )
        )
    assert failed.value.status_code == 401


def test_other_platforms_keep_existing_dev_route_behavior(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_PLATFORM", raising=False)
    admin_dashboard._check_auth(_request(scheme="http"))


def test_operator_proxy_is_secure_and_admin_is_loopback_only() -> None:
    caddy = (ROOT / "deploy/caddy/Caddyfile").read_text()
    service = (ROOT / "deploy/systemd/niftybot-admin.service").read_text()
    script = (ROOT / "deploy/scripts/apply_operator_ingress.sh").read_text()
    release = (ROOT / "deploy/lightsail_release.sh").read_text()
    assert "default_sni 65.0.255.84" in caddy
    assert "profile shortlived" in caddy
    assert "tls internal" not in caddy
    assert "reverse_proxy 127.0.0.1:8081" in caddy
    assert "--host 127.0.0.1 --port 8081" in service
    assert "--host 0.0.0.0 --port 8081" not in service
    assert 'caddy validate --config "$CADDY_SOURCE"' in script
    assert "systemctl reload caddy" in script
    assert 'systemctl restart "$SERVICE"' not in script
    assert "deploy/scripts/apply_operator_ingress.sh" in release
    assert "operator_ingress_degraded" in release
    assert (
        subprocess.run(
            ["bash", "-n", str(ROOT / "deploy/scripts/apply_operator_ingress.sh")],
            check=False,
        ).returncode
        == 0
    )


def test_timer_repairs_ingress_without_restarting_trading_engine() -> None:
    release = (ROOT / "deploy/lightsail_release.sh").read_text()
    start = release.index("repair_operator_ingress_if_needed() {")
    end = release.index("\n}\n", start) + 2
    repair = release[start:end]
    assert 'bash "$helper" && restart_admin' in repair
    assert 'systemctl restart "$SERVICE"' not in repair
    assert "intraday_flat_broker_confirmed" not in repair
    assert 'sudo -n cmp -s "$APP_DIR/deploy/caddy/Caddyfile"' in repair
    assert "OPERATOR_INGRESS_DEGRADED_MARKER" in repair
    # A same-revision timer tick runs this before the unchanged-SHA early exit.
    invocation = release.index("\n  repair_operator_ingress_if_needed\n")
    first_checkout = release.index('BEFORE="$(git rev-parse HEAD)"')
    assert end < invocation < first_checkout


def test_operator_repair_secures_admin_before_caddy_preflight() -> None:
    script = (ROOT / "deploy/scripts/apply_operator_ingress.sh").read_text()
    loopback_install = script.index('install -m 0644 "$ADMIN_SOURCE" "$ADMIN_TARGET"')
    admin_restart = script.index("systemctl restart niftybot-admin.service")
    caddy_version = script.index("if ! caddy_supported;")
    assert loopback_install < admin_restart < caddy_version
    assert "systemctl restart niftybot" not in script.replace(
        "systemctl restart niftybot-admin.service", ""
    )


def test_free_upgrade_is_quiet_only_and_bounded() -> None:
    script = (ROOT / "deploy/scripts/apply_operator_ingress.sh").read_text()
    assert "caddy_upgrade_attempted" in script
    assert 'find "$marker" -mmin +360' in script
    assert "/livez" in script and "quiet" in script
    assert "--only-upgrade -y -qq caddy" in script
    assert "DEBIAN_FRONTEND=noninteractive" in script


def test_ingress_success_requires_verified_public_certificate() -> None:
    script = (ROOT / "deploy/scripts/apply_operator_ingress.sh").read_text()
    assert "--resolve 65.0.255.84:443:127.0.0.1" in script
    assert "https://65.0.255.84/livez" in script
    assert "--noproxy '*'" in script
    assert " -k " not in script
    release = (ROOT / "deploy/lightsail_release.sh").read_text()
    assert release.count("current_operator_ingress_degraded") >= 2
