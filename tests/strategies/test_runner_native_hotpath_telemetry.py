from __future__ import annotations

import time
import types
from unittest.mock import MagicMock

import nifty_scalper_bot.strategies.runner as runner_module
from nifty_scalper_bot.core import runtime_reliability_hardening
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


def test_runner_cpu_count_uses_active_options_when_whitelist_empty() -> None:
    assert StrategyRunner._bump_cpu_metric.__module__ == StrategyRunner.__module__
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._cpu_opt_metrics = {}
    runner._eval_option_whitelist = set()
    runner._active_symbols = {"NFO:NIFTY26SEP25000CE", "NSE:NIFTY"}
    runner._should_log_throttled = lambda *_args: True

    runner._bump_cpu_metric("evaluated_symbols")

    logged = runner._logger.info.call_args.kwargs["extra"]
    assert logged["active_option_symbols_count"] == 1
    assert logged["active_option_count_source"] == "dynamic_active_symbols"


def test_runtime_reliability_install_does_not_replace_runner_methods() -> None:
    cpu_method = StrategyRunner._bump_cpu_metric
    tick_method = StrategyRunner.on_datahub_tick

    runtime_reliability_hardening.apply_patches()

    assert StrategyRunner._bump_cpu_metric is cpu_method
    assert StrategyRunner.on_datahub_tick is tick_method
