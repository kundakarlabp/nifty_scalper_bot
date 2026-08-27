"""Selected live ticks must keep causing entry evaluation after startup.

Production: one evaluation after MDM_CACHED_TICKS_REPLAYED path=direct_datahub,
then silence while quotes stayed fresh. This is a functional pipeline test,
not another watchdog detector test.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from nifty_scalper_bot.strategies.runner import EntryEvaluationRoute
from tests.strategies.test_entry_eval_coalescing import (
    _run_loop_in_thread,
    _stop_loop,
    _wait_until,
)
from tests.strategies.test_runner_symbol_role_gate import _build_phase9_runner


def _selected_option_runner(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    from nifty_scalper_bot.strategies.runner import SymbolRuntimeState

    runner_obj._symbol_state[selected_ce] = SymbolRuntimeState(selected_ce, 100)
    runner_obj._symbol_state[selected_ce].active = True
    runner_obj._data_phase[selected_ce] = "LIVE"
    runner_obj._history_ready_by_symbol[selected_ce] = True
    assert (
        runner_obj._entry_evaluation_route(selected_ce)
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )
    return runner_obj, strategy_manager, risk_manager, order_manager, selected_ce


def test_selected_option_ticks_keep_completing_entry_eval_after_first(monkeypatch):
    """Repeated selected-option ticks must increase selected eval completions."""
    runner_obj, _sm, _risk, _order, selected_ce = _selected_option_runner(monkeypatch)
    completed: list[str] = []
    original = runner_obj._evaluate_entry_from_latest_state

    def _capture(symbol, *, trace_id=None):
        original(symbol, trace_id=trace_id)
        completed.append(symbol)

    runner_obj._evaluate_entry_from_latest_state = _capture
    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        for i in range(5):
            runner_obj._on_tick_safe(
                {
                    "symbol": selected_ce,
                    "last_price": 110.0 + i,
                    "timestamp": time.time(),
                    "source": "ws",
                }
            )
            time.sleep(0.05)
        assert _wait_until(
            lambda: int(runner_obj._selected_candidate_eval_completed_count) >= 2,
            timeout=3.0,
        )
        assert runner_obj._entry_eval_completed_count >= 2
        assert runner_obj._entry_eligible_tick_count >= 2
        assert completed.count(selected_ce) >= 2
    finally:
        _stop_loop(loop, thread)


def test_context_only_eval_does_not_count_as_selected_candidate_eval(monkeypatch):
    runner_obj, _sm, _risk, _order, selected_ce = _selected_option_runner(monkeypatch)
    spot = "NSE:NIFTY"
    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": spot,
                "last_price": 24000.0,
                "timestamp": time.time(),
                "source": "ws",
            }
        )
        assert _wait_until(
            lambda: int(runner_obj._entry_eval_completed_count) >= 1, timeout=3.0
        )
        assert int(runner_obj._selected_candidate_eval_completed_count) == 0
        assert selected_ce not in (
            getattr(runner_obj, "_entry_eval_active_symbol", None) or ""
        )
    finally:
        _stop_loop(loop, thread)


def test_valid_selected_signal_reaches_order_manager_once(monkeypatch):
    """Fresh tick → eval → StrategyManager → risk → one OrderManager request.

    Uses the underlying trigger route that already constructs an option
    TradePlan through StrategyManager; the broker boundary is the mocked
    OrderManager.submit (never a live broker).
    """
    from tests.strategies.test_entry_eval_coalescing import (
        UNDERLYING_SYMBOL,
        _underlying_runner,
    )

    runner_obj, strategy_manager, risk_manager, order_manager, selected_ce = (
        _underlying_runner(monkeypatch)
    )
    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
                "source": "ws",
                "trace_id": "lifecycle-e2e",
            }
        )
        assert _wait_until(lambda: order_manager.submit.called, timeout=3.0)
        assert order_manager.submit.call_count == 1
        assert order_manager.submit.call_args.args[0].symbol == selected_ce
        strategy_manager.generate_signal.assert_called()
        risk_manager.validate.assert_called()
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24001.0,
                "timestamp": time.time(),
                "source": "ws",
            }
        )
        time.sleep(0.2)
        assert order_manager.submit.call_count == 1
    finally:
        _stop_loop(loop, thread)


def test_no_signal_produces_no_order(monkeypatch):
    runner_obj, strategy_manager, _risk, order_manager, selected_ce = (
        _selected_option_runner(monkeypatch)
    )
    strategy_manager.generate_signal = Mock(return_value=None)
    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": selected_ce,
                "last_price": 112.0,
                "timestamp": time.time(),
                "source": "ws",
            }
        )
        assert _wait_until(
            lambda: int(runner_obj._selected_candidate_eval_completed_count) >= 1,
            timeout=3.0,
        )
        assert order_manager.submit.call_count == 0
    finally:
        _stop_loop(loop, thread)


def test_disarmed_stall_does_not_place_an_order(monkeypatch):
    runner_obj, _sm, _risk, order_manager, selected_ce = _selected_option_runner(
        monkeypatch
    )
    runner_obj._entry_eval_stall_disarmed = True
    runner_obj._runtime_live_orders_armed = False
    runner_obj._runtime_readiness_reason = "strategy_evaluation_stalled"
    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": selected_ce,
                "last_price": 112.0,
                "timestamp": time.time(),
                "source": "ws",
            }
        )
        assert _wait_until(
            lambda: int(runner_obj._selected_candidate_eval_completed_count) >= 1,
            timeout=3.0,
        )
        assert order_manager.submit.call_count == 0
    finally:
        _stop_loop(loop, thread)


def test_schedule_exception_does_not_leave_drain_falsely_scheduled(monkeypatch):
    runner_obj, _sm, _risk, _order, selected_ce = _selected_option_runner(monkeypatch)
    runner_obj._main_loop = None
    runner_obj._runtime_loop_attached = False
    runner_obj._on_tick_safe(
        {
            "symbol": selected_ce,
            "last_price": 112.0,
            "timestamp": time.time(),
            "source": "ws",
        }
    )
    assert selected_ce in runner_obj._pending_entry_eval_symbols
    assert runner_obj._entry_eval_drain_scheduled is False
