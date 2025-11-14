from __future__ import annotations

from fastapi.testclient import TestClient

from nifty_scalper_bot.infra.health import HealthState, create_health_app


class DummyStreamer:
    def __init__(self) -> None:
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def backlog_size(self) -> int:
        return 3

    def simulate_disconnect(self) -> None:
        self._connected = False


class DummyOrderManager:
    def throttled_count(self) -> int:
        return 1

    def rejection_count(self) -> int:
        return 2


class DummyRiskManager:
    def cooldown_remaining(self) -> float:
        return 0.0


def test_health_endpoint_returns_status() -> None:
    state = HealthState(
        streamer=DummyStreamer(),
        order_manager=DummyOrderManager(),
        risk_manager=DummyRiskManager(),
        live_enabled=lambda: False,
    )
    app = create_health_app(state)
    client = TestClient(app)
    data = client.get("/health").json()
    assert data["live_enabled"] is False
    assert data["orders"]["rejections"] == 2
    assert "metrics" in data
    assert "alerts" in data
    metrics = client.get("/metrics")
    assert metrics.status_code == 200


def test_health_endpoint_reports_websocket_disabled() -> None:
    state = HealthState(
        streamer=None,
        order_manager=DummyOrderManager(),
        risk_manager=DummyRiskManager(),
        live_enabled=lambda: True,
        websocket_enabled=False,
    )
    app = create_health_app(state)
    client = TestClient(app)
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["websocket"]["enabled"] is False
    assert payload["websocket"]["mode"] == "polling"
