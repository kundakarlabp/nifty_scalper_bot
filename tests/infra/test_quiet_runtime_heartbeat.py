"""The quiet control plane must remain observable without a broker."""

from __future__ import annotations

from nifty_scalper_bot.infra.runtime_heartbeat import RuntimeHeartbeat


def test_quiet_mode_is_distinct_from_runtime_failure() -> None:
    messages = []

    def send(url, payload, timeout):
        messages.append(payload)
        return {"ok": True}

    heartbeat = RuntimeHeartbeat(
        endpoint_url="https://test.invalid",
        health_reader=lambda: (200, {"status": "quiet", "blockers": ["auto_quiet"]}),
        service_health_reader=lambda: {
            "admin_http_responsive": True,
            "streamlit_http_responsive": True,
        },
        transport=send,
    )
    result = heartbeat.report_once()
    assert result["engine_healthy"] is False
    assert messages[0]["trading_status"] == "quiet"
    assert messages[0]["blockers"] == ["auto_quiet"]
    assert messages[0]["failure_reason"] is None
    assert messages[0]["admin_http_responsive"] is True
    assert messages[0]["streamlit_http_responsive"] is True
