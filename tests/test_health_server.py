from src.server import health


def test_live_endpoint_returns_uptime():
    client = health.app.test_client()
    resp = client.get("/live")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "live"
    assert data["uptime_sec"] >= 0


def test_ready_respects_status_callback():
    try:
        health._status_callback = lambda: {"ok": False}
        client = health.app.test_client()
        resp = client.get("/ready")
        assert resp.status_code == 503
    finally:
        health._status_callback = None

