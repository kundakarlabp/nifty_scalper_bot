from __future__ import annotations

from types import SimpleNamespace

import src.strategies.runner as runner_module
from src.utils.health import STATE


def test_enqueue_tick_updates_health_state(monkeypatch) -> None:
    orchestrator = runner_module.Orchestrator(
        data_source=SimpleNamespace(),
        executor=SimpleNamespace(),
        on_tick=lambda _tick: None,
    )
    fake_time = 1234.0
    monkeypatch.setattr(runner_module.time, "time", lambda: fake_time)
    monkeypatch.setattr(STATE, "last_tick_ts", 0.0)
    orchestrator._enqueue_tick(SimpleNamespace())
    assert orchestrator._last_tick == fake_time
    assert STATE.last_tick_ts == fake_time
