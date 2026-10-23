"""No broker or AppDeploy dependencies in private Supabase heartbeat."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.infra.runtime_heartbeat import (
    RuntimeHeartbeat,
    build_runtime_heartbeat,
    heartbeat_interval_seconds,
)


def _make_sender(status=200, health=None):
    sent = []

    def transport(url, payload, timeout):
        sent.append((url, dict(payload), timeout))
        return {"ok": True}

    sender = RuntimeHeartbeat(
        endpoint_url="https://supabase.example/internal",
        transport=transport,
        health_reader=lambda: (status, health or {}),
    )
    return sender, sent


def test_disabled_without_explicit_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_RUNTIME_HEARTBEAT_ENABLED", raising=False)
    assert build_runtime_heartbeat() is None


def test_interval_is_clamped_off_trading_hot_path(monkeypatch):
    monkeypatch.setenv("SUPABASE_RUNTIME_HEARTBEAT_INTERVAL_SECONDS", "2")
    assert heartbeat_interval_seconds() == 60.0


def test_sanitized_healthy_payload_never_contains_account_secrets(monkeypatch):
    monkeypatch.setattr(
        "nifty_scalper_bot.infra.runtime_heartbeat.resolve_build_sha",
        lambda: "a" * 40,
    )
    sender, calls = _make_sender(
        health={
            "status": "armed",
            "blockers": [],
            "broker_access_token": "MUST_NOT_LEAK",
            "available_balance": 15206.4,
        }
    )
    result = sender.report_once(datetime(2026, 10, 22, tzinfo=timezone.utc))
    assert result["published"] is True
    assert calls[0][1] == {
        "observed_at": "2026-10-22T00:00:00+00:00",
        "build_sha": "a" * 40,
        "engine_healthy": True,
        "http_status": 200,
        "trading_status": "armed",
        "blockers": [],
        "failure_reason": None,
        "release_state": None,
        "release_updated_at": None,
    }


def test_market_closed_is_not_a_process_outage():
    sender, calls = _make_sender(
        health={"status": "blocked", "blockers": ["market_closed"]}
    )
    result = sender.report_once()
    assert result["engine_healthy"] is True
    assert calls[0][1]["blockers"] == ["market_closed"]


def test_http_error_is_reported_without_fabricating_liveness():
    sender, calls = _make_sender(status=503)
    assert sender.report_once()["engine_healthy"] is False
    assert calls[0][1]["failure_reason"] == "local_health_http_error"


def test_local_connection_error_is_reported(monkeypatch):
    import requests

    sender, calls = _make_sender()

    def fail():
        raise requests.ConnectionError("secret data")

    sender.health_reader = fail
    assert sender.report_once()["engine_healthy"] is False
    assert calls[0][1]["failure_reason"] == "local_health_unavailable"
    assert "secret" not in str(calls[0][1])


def test_disallowed_blocker_values_removed():
    sender, calls = _make_sender(
        health={
            "status": "blocked",
            "blockers": ["market_closed", "private/token:value"],
        }
    )
    sender.report_once()
    assert calls[0][1]["blockers"] == ["market_closed"]


def test_failed_ingest_does_not_claim_success():
    sender, _ = _make_sender()
    sender.transport = lambda *_: {"ok": False}
    with pytest.raises(RuntimeError, match="runtime_heartbeat_ingest_failed"):
        sender.report_once()


def test_heartbeat_sends_only_allowlisted_release_metadata(tmp_path, monkeypatch):
    import json

    from nifty_scalper_bot.infra import runtime_heartbeat as mod

    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "state": "deferred_broker_exposure",
                "updated_at": "2026-10-23T09:10:00+05:30",
                "message": "sensitive token=DO_NOT_LEAK",
                "command": "never forward",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_UPDATE_STATUS_FILE", str(path))
    sender, calls = _make_sender(health={"status": "armed", "blockers": []})
    assert sender.report_once()["published"] is True
    payload = calls[0][1]
    assert payload["release_state"] == "deferred_broker_exposure"
    assert payload["release_updated_at"] == "2026-10-23T03:40:00+00:00"
    assert "DO_NOT_LEAK" not in str(payload)
    assert "command" not in str(payload)
    assert mod._local_release_state() == (
        "deferred_broker_exposure",
        "2026-10-23T03:40:00+00:00",
    )


def test_invalid_release_metadata_is_not_published(tmp_path, monkeypatch):
    path = tmp_path / "release.json"
    path.write_text(
        '{"state":"secret=not_valid","updated_at":"not-a-date"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("BOT_UPDATE_STATUS_FILE", str(path))
    sender, calls = _make_sender()
    sender.report_once()
    assert calls[0][1]["release_state"] is None
    assert calls[0][1]["release_updated_at"] is None
