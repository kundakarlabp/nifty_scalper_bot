"""Tests for structured signal execution outcomes in StrategyRunner."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

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


def _state_runner() -> StrategyRunner:
    import threading

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._execution_state_lock = threading.RLock()
    runner._execution_state_by_symbol = {}
    return runner


def test_prepare_order_state_from_idle_reaches_order_pending() -> None:
    runner = _state_runner()

    ok, reason, details = runner._prepare_order_state_for_submission(
        'NFO:NIFTY2661623900PE', trace_id='idle-final'
    )

    assert ok is True
    assert reason == 'ok'
    assert details['after']['state'] == ExecutionState.ORDER_PENDING.value


def test_prepare_order_state_from_ready_reaches_order_pending() -> None:
    runner = _state_runner()
    machine = runner._get_execution_state_machine('NFO:NIFTY2661623900PE')
    assert machine.transition(ExecutionState.READY) is True

    ok, reason, details = runner._prepare_order_state_for_submission(
        'NFO:NIFTY2661623900PE', trace_id='ready-final'
    )

    assert ok is True
    assert reason == 'ok'
    assert details['after']['state'] == ExecutionState.ORDER_PENDING.value


@pytest.mark.parametrize('busy_state', [ExecutionState.ORDER_PENDING, ExecutionState.POSITION_OPEN])
def test_prepare_order_state_rejects_busy_final_symbol(busy_state) -> None:
    runner = _state_runner()
    machine = runner._get_execution_state_machine('NFO:NIFTY2661623900PE')
    assert machine.transition(ExecutionState.SIGNAL_RECEIVED) is True
    assert machine.transition(ExecutionState.ORDER_PENDING) is True
    if busy_state == ExecutionState.POSITION_OPEN:
        assert machine.transition(ExecutionState.POSITION_OPEN) is True

    ok, reason, details = runner._prepare_order_state_for_submission(
        'NFO:NIFTY2661623900PE', trace_id='busy-final'
    )

    assert ok is False
    assert reason == 'signal_state_rejected'
    assert details['after']['state'] == busy_state.value
