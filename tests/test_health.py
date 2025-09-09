import types
import logging
from src.server import health


def test_health_live():
    body, status = health.live()
    assert status == 200
    assert body["status"] == "live"


def test_health_head():
    resp, status = health.health_head()
    assert status == 200


def test_health_get_paths(caplog):
    # default callback None -> ok path
    health._status_callback = None
    body, status = health.health_get()
    assert status == 200
    assert body["ok"]

    # error path via callback raising
    def bad_cb():
        raise RuntimeError("boom")

    health._status_callback = bad_cb
    with caplog.at_level("ERROR", logger="src.server.health"):
        body, status = health.health_get()
    assert status == 500
    assert not body["ok"]


def test_status_get_paths(caplog):
    def ok_cb():
        return {"ok": True}

    health._status_callback = ok_cb
    body, status = health.status_get()
    assert status == 200
    assert body["ok"]

    def bad_cb():
        raise RuntimeError("boom")

    health._status_callback = bad_cb
    with caplog.at_level("ERROR", logger="src.server.health"):
        body, status = health.status_get()
    assert status == 500
    assert not body["ok"]
