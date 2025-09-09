from __future__ import annotations

import builtins
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from src.server import health as health_module
from src.strategies import runner as runner_module


def _runner_with_tick(ts: datetime) -> SimpleNamespace:
    ds = SimpleNamespace(
        last_tick_ts=lambda: ts,
        last_bar_open_ts=lambda: None,
        timeframe_seconds=60,
    )
    cfg = SimpleNamespace(max_tick_lag_s=10, max_bar_lag_s=75)
    kite = SimpleNamespace(is_connected=lambda: True)
    return SimpleNamespace(kite=kite, data_source=ds, strategy_cfg=cfg)


def test_ready_endpoint_ok(monkeypatch) -> None:
    fake_runner = _runner_with_tick(datetime.utcnow())
    monkeypatch.setattr(
        runner_module.StrategyRunner, "get_singleton", classmethod(lambda cls: fake_runner)
    )
    client = health_module.app.test_client()
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_endpoint_stale(monkeypatch) -> None:
    stale = datetime.utcnow() - timedelta(seconds=999)
    fake_runner = _runner_with_tick(stale)
    monkeypatch.setattr(
        runner_module.StrategyRunner, "get_singleton", classmethod(lambda cls: fake_runner)
    )
    client = health_module.app.test_client()
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_live_endpoint() -> None:
    client = health_module.app.test_client()
    resp = client.get("/live")
    assert resp.status_code == 200


def test_run_uses_waitress_when_available(monkeypatch) -> None:
    calls: dict[str, tuple[str, int]] = {}

    def fake_serve(app, host: str, port: int) -> None:  # noqa: ANN001
        calls["args"] = (host, port)

    monkeypatch.setitem(sys.modules, "waitress", SimpleNamespace(serve=fake_serve))
    monkeypatch.setattr(
        health_module.app,
        "run",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not call app.run")),
    )

    health_module.run(host="1.2.3.4", port=5555)
    assert calls["args"] == ("1.2.3.4", 5555)


def test_run_falls_back_when_waitress_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "waitress", raising=False)
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "waitress":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    calls: dict[str, Any] = {}

    def fake_run(*, host: str, port: int, debug: bool, use_reloader: bool, threaded: bool) -> None:
        calls.update(
            {
                "host": host,
                "port": port,
                "debug": debug,
                "use_reloader": use_reloader,
                "threaded": threaded,
            }
        )

    monkeypatch.setattr(health_module.app, "run", fake_run)

    health_module.run(host="0.0.0.0", port=8080)
    assert calls == {
        "host": "0.0.0.0",
        "port": 8080,
        "debug": False,
        "use_reloader": False,
        "threaded": True,
    }
