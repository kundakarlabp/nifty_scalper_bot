from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from nifty_scalper_bot.strategies.runner import (
    SignalExecutionResult,
    StrategyRunner,
    SymbolRuntimeState,
    TradeRecord,
)
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.log_throttle import event_is_never_throttled


def _bare_runner() -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    runner._logger = Mock()
    runner._terminal_signal_trace_ids = {}
    runner._record_trade_decision_snapshot = Mock()
    return runner


def _signal(action: str = "BUY") -> Signal:
    return Signal(
        action,
        "NFO:NIFTY26SEP25000CE",
        75,
        0.9,
        "test_signal",
        90.0,
        120.0,
        metadata={},
    )


def test_terminal_result_is_unthrottled_exactly_once_per_candidate_trace() -> None:
    runner = _bare_runner()

    first = runner._emit_signal_execution_result(
        symbol="NFO:NIFTY26SEP25000CE",
        trace_id="candidate-1",
        result=SignalExecutionResult(False, "low_volatility"),
        broker_attempted=False,
    )
    duplicate = runner._emit_signal_execution_result(
        symbol="NFO:NIFTY26SEP25000CE",
        trace_id="candidate-1",
        result=SignalExecutionResult(False, "duplicate"),
        broker_attempted=False,
    )

    assert first is True
    assert duplicate is False
    terminal_calls = [
        call
        for call in runner._logger.info.call_args_list
        if call.kwargs.get("extra", {}).get("event") == "SIGNAL_EXECUTION_RESULT"
    ]
    assert len(terminal_calls) == 1
    assert terminal_calls[0].kwargs["extra"]["broker_attempted"] is False
    assert terminal_calls[0].kwargs["extra"]["bypass_filters"] is True
    assert event_is_never_throttled("SIGNAL_EXECUTION_RESULT") is True


@pytest.mark.asyncio
async def test_preparation_task_exception_has_terminal_result() -> None:
    runner = _bare_runner()
    runner._prepare_signal_for_handling = AsyncMock(
        side_effect=RuntimeError("prepare exploded")
    )

    scheduled, _ = runner._schedule_signal_preparation(
        _signal(), 100.0, datetime.now(timezone.utc), "prepare-error"
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert scheduled is True
    terminal_calls = [
        call
        for call in runner._logger.info.call_args_list
        if call.kwargs.get("extra", {}).get("event") == "SIGNAL_EXECUTION_RESULT"
    ]
    assert len(terminal_calls) == 1
    assert (
        terminal_calls[0].kwargs["extra"]["reason"] == "signal_preparation_task_failed"
    )
    assert terminal_calls[0].kwargs["extra"]["broker_attempted"] is False


@pytest.mark.asyncio
async def test_preparation_task_cancellation_has_terminal_result() -> None:
    runner = _bare_runner()
    started = asyncio.Event()

    async def _wait_forever(*_args):
        started.set()
        await asyncio.Event().wait()

    runner._prepare_signal_for_handling = _wait_forever
    scheduled, _ = runner._schedule_signal_preparation(
        _signal(), 100.0, datetime.now(timezone.utc), "prepare-cancel"
    )
    assert scheduled is True
    await started.wait()
    task = next(
        task
        for task in asyncio.all_tasks()
        if task.get_name().endswith(":prepare-cancel")
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    terminal_calls = [
        call
        for call in runner._logger.info.call_args_list
        if call.kwargs.get("extra", {}).get("event") == "SIGNAL_EXECUTION_RESULT"
    ]
    assert len(terminal_calls) == 1
    assert terminal_calls[0].kwargs["extra"]["reason"] == "signal_preparation_cancelled"


def test_preparation_scheduling_failure_returns_traceable_reason(monkeypatch) -> None:
    runner = _bare_runner()
    runner._prepare_signal_for_handling = AsyncMock()
    loop = Mock()
    loop.create_task.side_effect = RuntimeError("scheduler stopped")
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: loop)

    scheduled, reason = runner._schedule_signal_preparation(
        _signal(), 100.0, datetime.now(timezone.utc), "schedule-error"
    )

    assert scheduled is False
    assert reason == "signal_preparation_scheduling_failed"


def test_trade_timestamp_advances_only_for_authoritative_accepted_entry() -> None:
    runner = _bare_runner()
    runner._lock = threading.RLock()
    runner._persistent_state = None
    symbol = "NFO:NIFTY26SEP25000CE"
    runner._symbol_state = {symbol: SymbolRuntimeState(symbol, 10)}
    now = datetime.now(timezone.utc)

    runner._record_trade(symbol, TradeRecord(now, "BUY", 0, 100.0, "error"))
    assert runner._symbol_state[symbol].last_trade_at is None

    runner._record_trade(
        symbol, TradeRecord(now, "BUY", 75, 100.0, "submitted", order_id="OID-1")
    )
    assert runner._symbol_state[symbol].last_trade_at == now.timestamp()

    later = datetime.fromtimestamp(now.timestamp() + 5.0, tz=timezone.utc)
    runner._record_trade(
        symbol, TradeRecord(later, "BUY", 0, 100.0, "error", order_id=None)
    )
    assert runner._symbol_state[symbol].last_trade_at == now.timestamp()


def test_deterministic_risk_cooldown_result_is_explicitly_prebroker() -> None:
    runner = _bare_runner()
    runner._execution_reject_cooldown_ts = {
        "NFO:NIFTY26SEP25000CE:test:risk_capacity_unavailable": 100.0
    }
    runner._exec_reject_runtime_not_ready_seconds = 10.0
    runner._exec_reject_invalid_lot_seconds = 300.0
    runner._exec_reject_rr_seconds = 5.0
    runner._exec_reject_margin_seconds = 30.0
    runner._exec_reject_position_seconds = 15.0

    result = runner._execution_reject_cooldown_result(
        "NFO:NIFTY26SEP25000CE", "test", 101.0, "risk-repeat"
    )

    assert result is not None
    assert result.reason == "risk_capacity_unavailable_reject_cooldown"
    assert result.details["broker_attempted"] is False
