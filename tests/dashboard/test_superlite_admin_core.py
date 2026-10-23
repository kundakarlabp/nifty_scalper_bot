from __future__ import annotations

import json
import os

from starlette.requests import Request


def test_atomic_update_retains_unspecified_settings(tmp_path, monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    path = tmp_path / "settings.env"
    path.write_text("FIRST=one\nSECOND=two\nDAILY=old\n", encoding="utf-8")
    monkeypatch.setattr(core, "ENV_PATH", path)
    core.ENV_CACHE.update({"at": 0.0, "data": {}})

    core.write_env({"DAILY": "new"})

    values = core.read_env()
    assert values == {"FIRST": "one", "SECOND": "two", "DAILY": "new"}
    if os.name == "nt":
        assert path.exists()
    else:
        assert path.stat().st_mode & 0o777 == 0o600


def test_same_origin_post_is_allowed() -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/control",
            "raw_path": b"/control",
            "query_string": b"",
            "headers": [(b"host", b"bot.local"), (b"origin", b"http://bot.local")],
            "client": ("127.0.0.1", 1),
            "server": ("bot.local", 80),
        }
    )
    core.same_origin(request)


def test_shadow_closed_market_is_still_operational(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def fake_json(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/trading/status":
            return {"execution_mode": "SHADOW"}
        return {
            "live_orders_armed": False,
            "blockers": ["not_live_mode", "market_closed"],
            "broker": {"ready": True},
            "reconciliation": {"completed": True},
        }

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(core, "bounded_logs", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["operational_ready"] is True
    assert status["live_orders_armed"] is False
    assert status["operational_blockers"] == []


def test_engine_timeout_status_wins(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def fake_json(path: str):
        if path == "/livez":
            return {"_error": "ENGINE HTTP TIMEOUT"}
        if path == "/health/trading":
            return {
                "selected": {"ce": "NFO:CE", "pe": "NFO:PE", "atm": 24000},
                "blockers": [""],
            }
        return {"execution_mode": "LIVE"}

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(
        core,
        "bounded_logs",
        lambda *_args, **_kwargs: (
            "CONTRACT_SSOT_ATM_PAIR_SELECTED "
            "selected_ce=OLD selected_pe=OLD atm_strike=1"
        ),
    )
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["engine_http_responsive"] is False
    assert status["engine_http_status"] == "ENGINE HTTP TIMEOUT"
    assert status["mode"] == "LIVE"
    assert status["selected"] == {"ce": "NFO:CE", "pe": "NFO:PE", "atm": 24000}
    assert status["selected_source"] == "health_trading.selected"


def test_blank_blockers_and_stale_updater(monkeypatch, tmp_path) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"state":"deploying","updated_ts":1,"message":"deploying old"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "STATUS_PATH", status_file)
    monkeypatch.setenv("BOT_UPDATER_STALE_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(core, "_service_process_known", lambda: None)

    def fake_json(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {"blockers": ["", "selected_option_history_cold", ""]}
        return {"execution_mode": "PAPER"}

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(core, "bounded_logs", lambda *_args, **_kwargs: "")
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["blockers"] == ["selected_option_history_cold"]
    assert status["service_process_known"] is None
    assert status["updater"]["state"] == "stale_interrupted"
    assert status["updater"]["previous_state"] == "deploying"


def test_updater_current_iso_timestamp(monkeypatch, tmp_path) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"state":"deploying","updated_at":"2999-01-01T00:00:00+00:00","message":"fresh"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "STATUS_PATH", status_file)
    monkeypatch.setenv("BOT_UPDATER_STALE_TIMEOUT_SECONDS", "1")
    value = core._update_state()
    assert value["state"] == "deploying"
    assert value.get("stale") is not True


def test_updater_malformed_timestamp_is_stale(monkeypatch, tmp_path) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"state":"validating","updated_at":"not-a-date","message":"bad clock"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "STATUS_PATH", status_file)
    monkeypatch.setenv("BOT_UPDATER_STALE_TIMEOUT_SECONDS", "1")
    value = core._update_state()
    assert value["state"] == "stale_interrupted"
    assert value["previous_state"] == "validating"
    assert value["stale_reason"] == "malformed_timestamp"


def test_interrupted_runtime_recovery_is_reported_stale(monkeypatch, tmp_path) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    status_file = tmp_path / "status.json"
    status_file.write_text(
        '{"state":"recovering_runtime","updated_ts":1,"message":"restarting"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "STATUS_PATH", status_file)
    monkeypatch.setenv("BOT_UPDATER_STALE_TIMEOUT_SECONDS", "1")
    value = core._update_state()
    assert value["state"] == "stale_interrupted"
    assert value["previous_state"] == "recovering_runtime"


def test_broker_auth_unknown_is_not_reported_authenticated(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def fake_json(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {
                "broker_authentication": "unknown",
                "broker": {"ready": True},
                "blockers": [],
            }
        return {"execution_mode": "SHADOW"}

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(core, "bounded_logs", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["broker_authenticated"] == "unknown"


def test_status_selected_prefers_structured_top_level_dynamic_fields(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def fake_json(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {
                "selected_ce": "NFO:NIFTY26JUL24000CE",
                "selected_pe": "NFO:NIFTY26JUL24000PE",
                "atm_strike": 24000,
                "blockers": [],
            }
        return {
            "execution_mode": "LIVE",
            "selected": {"ce": "OLDCE", "pe": "OLDPE", "atm": 1},
        }

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(
        core,
        "bounded_logs",
        lambda *_args, **_kwargs: "ACTIVE_DYNAMIC_BASKET_COMMITTED selected_ce=LOGCE selected_pe=LOGPE futures_symbol=NFO:NIFTY26JULFUT atm_strike=999",
    )
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    monkeypatch.setattr(core, "_service_process_known", lambda: True)
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["selected"] == {
        "ce": "NFO:NIFTY26JUL24000CE",
        "pe": "NFO:NIFTY26JUL24000PE",
        "atm": 24000,
    }
    assert status["selected_source"] == "health_trading.top_level"


def test_status_selected_falls_back_to_dynamic_basket_log(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def fake_json(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {"blockers": []}
        return {"execution_mode": "LIVE"}

    monkeypatch.setattr(core, "_http_json", fake_json)
    monkeypatch.setattr(core, "_git_ref", lambda _ref: "abc1234")
    monkeypatch.setattr(
        core,
        "bounded_logs",
        lambda *_args, **_kwargs: (
            "CONTRACT_SSOT_ATM_PAIR_SELECTED selected_ce=OLDCE selected_pe=OLDPE atm_strike=1\n"
            "ACTIVE_DYNAMIC_BASKET_COMMITTED selected_ce=NFO:NEWCE selected_pe=NFO:NEWPE futures_symbol=NFO:NIFTY26JULFUT atm_strike=24050"
        ),
    )
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "current"})
    monkeypatch.setattr(core, "_service_process_known", lambda: True)
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()
    assert status["selected"] == {"ce": "NFO:NEWCE", "pe": "NFO:NEWPE", "atm": "24050"}
    assert status["selected_source"] == "journal:active_dynamic_basket"


def test_dashboard_runtime_sha_differs_from_checkout(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    def payload(path: str):
        if path == "/livez":
            return {"bot_loaded": True}
        if path == "/health/trading":
            return {
                "build_sha": "eee8cf5246034259a41e27d25571796d09ad2648",
                "blockers": [],
            }
        return {"execution_mode": "LIVE"}

    monkeypatch.setattr(core, "_http_json", payload)
    monkeypatch.setattr(
        core,
        "_git_ref",
        lambda ref: "1234abc" if ref == "HEAD" else "a4d9a58",
    )
    monkeypatch.setattr(
        core, "_update_state", lambda: {"state": "deferred_market_window"}
    )
    monkeypatch.setattr(core, "bounded_logs", lambda *_args, **_kwargs: "")
    core.STATUS_CACHE.update({"at": 0.0, "data": {}})

    status = core.status_snapshot()

    assert status["running"] == "eee8cf5"
    assert status["runtime_sha"].startswith("eee8cf52")
    assert status["checkout_sha"] == "1234abc"
    assert status["runtime_checkout_mismatch"] is True
    assert status["stale"] is True
    assert status["remote_source"] == "cached_local_origin/main"
    assert status["updater"]["state"] == "deferred_market_window"


def test_host_local_service_health_separates_quiet_from_liveness(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    monkeypatch.setattr(
        core, "_http_json", lambda _path: {"quiet": True, "bot_loaded": False}
    )
    monkeypatch.setattr(core, "_streamlit_reachable", lambda: True)
    monkeypatch.setattr(
        core, "_update_state", lambda: {"state": "current", "api_key": "SECRET"}
    )
    core.SERVICE_HEALTH_CACHE.update({"at": 0.0, "data": {}})
    probe = core.service_health_snapshot()
    assert probe["scope"] == "host_local_only"
    assert probe["engine_http_responsive"] is True
    assert probe["engine_quiet"] is True
    assert probe["admin_http_responsive"] is True
    assert probe["streamlit_http_responsive"] is True
    assert probe["release_state"] == "current"
    assert "SECRET" not in str(probe)
    assert "api_key" not in probe


def test_host_local_service_health_unknown_fail_closed_and_cached(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    observed = []
    monkeypatch.setattr(
        core, "_http_json", lambda path: observed.append(path) or {"_error": "timeout"}
    )
    monkeypatch.setattr(core, "_streamlit_reachable", lambda: False)
    monkeypatch.setattr(core, "_update_state", lambda: {"state": "api_key=DO_NOT_LEAK"})
    core.SERVICE_HEALTH_CACHE.update({"at": 0.0, "data": {}})
    probe = core.service_health_snapshot()
    assert probe["engine_http_responsive"] is False
    assert probe["engine_quiet"] is None
    assert probe["streamlit_http_responsive"] is False
    assert probe["release_state"] == "unknown"
    assert core.service_health_snapshot() == probe
    assert observed == ["/livez"]


def test_streamlit_probe_requires_exact_health_response(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin_core as core

    responses = []

    class FakeResponse:
        status = 302

        def read(self, _max_bytes: int) -> bytes:
            return b"redirect"

    class FakeHTTPConnection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ("127.0.0.1", 8501, 1.0)
            responses.append(self)

        def request(self, method, path):
            assert (method, path) == ("GET", "/_stcore/health")

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr(core.http.client, "HTTPConnection", FakeHTTPConnection)
    assert core._streamlit_reachable() is False
    assert len(responses) == 1


def test_service_health_endpoint_has_no_control_actions(monkeypatch) -> None:
    import nifty_scalper_bot.superlite_admin as admin

    monkeypatch.setattr(
        admin,
        "service_health_snapshot",
        lambda: {
            "scope": "host_local_only",
            "engine_http_responsive": True,
            "admin_http_responsive": True,
            "streamlit_http_responsive": False,
        },
    )
    routes = [
        route
        for route in admin.app.routes
        if getattr(route, "path", None) == "/admin/api/service-health"
    ]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}
    response = routes[0].endpoint()
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = json.loads(response.body)
    assert payload["streamlit_http_responsive"] is False
    assert "api_key" not in payload
