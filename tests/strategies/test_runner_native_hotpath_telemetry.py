from __future__ import annotations

import time
import types
from unittest.mock import MagicMock

import nifty_scalper_bot.strategies.runner as runner_module
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_datahub_latency_telemetry_is_native(monkeypatch) -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._entry_evaluation_route = lambda _symbol: types.SimpleNamespace(
        value="context_only"
    )
    runner.on_tick_event = lambda _tick: time.sleep(0.051)
    captured: list[dict[str, object]] = []

    def _capture(_logger, _key, _message, *args, **kwargs) -> None:
        _ = args
        captured.append(dict(kwargs.get("extra") or {}))

    monkeypatch.setattr(runner_module, "log_throttled", _capture)

    StrategyRunner.on_datahub_tick(runner, {"symbol": "NSE:NIFTY"})

    assert captured
    assert captured[-1]["event"] == "RUNNER_DATAHUB_TICK_SLOW"
    assert captured[-1]["route"] == "context_only"
    assert float(captured[-1]["duration_ms"]) >= 50.0
