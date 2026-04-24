"""Tests for structured signal execution outcomes in StrategyRunner."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from nifty_scalper_bot.execution.order_state_machine import ExecutionState
from nifty_scalper_bot.strategies.runner import SignalExecutionResult, StrategyRunner


class _StubLogger:
    def info(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def error(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None


def test_handle_signal_returns_outside_market_hours_result(monkeypatch) -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = _StubLogger()
    runner._normalize_symbol = lambda symbol: symbol
    runner._transition_execution_state = lambda *_args, **_kwargs: ExecutionState.IDLE

    monkeypatch.setattr(
        'nifty_scalper_bot.strategies.runner.is_market_hours_cached',
        lambda: False,
    )

    signal = SimpleNamespace(action='BUY', symbol='NFO:NIFTY26APR23800CE')
    result = runner._handle_signal(
        signal,
        120.0,
        datetime.now(timezone.utc),
        trace_id='trace-1',
    )

    assert isinstance(result, SignalExecutionResult)
    assert result.accepted is False
    assert result.reason == 'outside_market_hours'
