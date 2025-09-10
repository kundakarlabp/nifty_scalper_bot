from types import SimpleNamespace

import src.server.health as health
from src.strategies.runner import StrategyRunner


def test_ready_handles_runner_states(monkeypatch) -> None:
    monkeypatch.setattr(StrategyRunner, "get_singleton", lambda: None)
    resp, status = health.ready()
    assert status == 503
    assert resp["status"] == "starting"

    runner = SimpleNamespace(kite=None)
    monkeypatch.setattr(StrategyRunner, "get_singleton", lambda: runner)
    resp, status = health.ready()
    assert status == 503
    assert resp["reason"] == "broker"

    runner = SimpleNamespace(
        kite=SimpleNamespace(is_connected=lambda: True),
        data_source=None,
    )
    monkeypatch.setattr(StrategyRunner, "get_singleton", lambda: runner)
    resp, status = health.ready()
    assert status == 503
    assert resp["reason"] == "data"
