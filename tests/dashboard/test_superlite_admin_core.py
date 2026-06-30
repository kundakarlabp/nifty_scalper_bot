from __future__ import annotations

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


def test_engine_http_timeout_is_explicit_and_structured_status_wins(monkeypatch) -> None:
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
