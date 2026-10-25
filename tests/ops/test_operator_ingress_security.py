"""Regression contracts for production-only HTTPS admin access and ingress sync."""

from __future__ import annotations

import base64
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


def _basic(password: str) -> str:
    return "Basic " + base64.b64encode(f"admin:{password}".encode()).decode()


@pytest.mark.parametrize(
    ("case_request", "status"),
    [
        (_request(scheme="http", authorization=_basic("long-test-pass")), 403),
        (_request(), 401),
        (_request(authorization=_basic("incorrect")), 401),
        (_request(authorization="Basic not-base64"), 401),
        (
            _request(
                method="POST",
                authorization=_basic("long-test-pass"),
                origin="https://evil.example",
            ),
            403,
        ),
    ],
)
def test_public_production_controls_fail_closed(
    monkeypatch, case_request, status
) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "long-test-pass")
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(case_request)
    assert failed.value.status_code == status


def test_public_https_requires_correct_admin_password_and_origin(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "long-test-pass")
    admin_dashboard._check_auth(_request(authorization=_basic("long-test-pass")))
    admin_dashboard._check_auth(
        _request(
            method="POST",
            authorization=_basic("long-test-pass"),
            origin="https://65.0.255.84",
        )
    )


def test_public_missing_password_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(admin_dashboard, "_read_env", lambda: {})
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(_request())
    assert failed.value.status_code == 503


def test_host_local_process_probes_need_no_browser_credentials(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    admin_dashboard._check_auth(_request(scheme="http", peer="127.0.0.1"))


def test_forwarded_public_client_cannot_skip_auth_as_loopback(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ADMIN_PASSWORD", "long-test-pass")
    with pytest.raises(HTTPException) as failed:
        admin_dashboard._check_auth(
            _request(
                scheme="https",
                peer="127.0.0.1",
                forwarded_for="203.0.113.5",
            )
        )
    assert failed.value.status_code == 401


def test_other_platforms_keep_existing_dev_route_behavior(monkeypatch) -> None:
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
