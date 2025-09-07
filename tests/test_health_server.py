from __future__ import annotations

from typing import Any

from src.server import health


def test_ready_includes_uptime() -> None:
    """The /ready endpoint exposes a monotonic uptime counter."""
    app = health.app
    client = app.test_client()
    resp = client.get("/ready")
    data: dict[str, Any] = resp.get_json()  # type: ignore[assignment]
    assert resp.status_code == 200
    assert data["status"] == "ready"
    assert data["uptime_sec"] >= 0

