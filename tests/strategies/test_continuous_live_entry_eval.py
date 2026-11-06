"""Selected live ticks must keep causing entry evaluation after startup.

Production: one evaluation after MDM_CACHED_TICKS_REPLAYED path=direct_datahub,
then silence while quotes stayed fresh. This is a functional pipeline test,
not another watchdog detector test.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import Mock

from nifty_scalper_bot.strategies.runner import EntryEvaluationRoute, StrategyRunner
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


def test_hydrated_selected_option_with_fresh_ws_quote_evaluates_without_live_bar(
    monkeypatch,
):
    """Fresh, activated selected options must not starve before their first live bar."""
    runner_obj, strategy_manager, _risk, order_manager, selected_ce = (
        _selected_option_runner(monkeypatch)
    )
    strategy_manager.generate_signal = Mock(return_value=None)
    runner_obj._data_phase[selected_ce] = "HYDRATION"
    runner_obj._symbol_has_completed_strategy_eval.add(selected_ce)
    runner_obj._last_bar_ts[selected_ce] = datetime.now(timezone.utc)
    runner_obj._last_strategy_versions[selected_ce] = 0
    runner_obj._candle_versions[selected_ce] = 0
    runner_obj._symbol_history.pop(selected_ce, None)
    runner_obj._same_bar_eval_reason = StrategyRunner._same_bar_eval_reason.__get__(
        runner_obj, StrategyRunner
    )
    runner_obj._is_option_symbol_tick_fresh = lambda symbol: symbol == selected_ce
    runner_obj._fallback_enabled = False
    runner_obj._allow_polling_fallback = False
    monkeypatch.setenv("REQUIRE_BID_ASK_FOR_OPTION_EVAL", "false")

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    runner_obj._runtime_loop_attached = True
    try:
        for i in range(2):
            # Advance the two existing rate-limit clocks, not the candle itself.
            runner_obj._last_same_bar_eval_ts_by_symbol[selected_ce] = 0.0
            runner_obj._last_periodic_eval_at_by_symbol[selected_ce] = 0.0
            runner_obj._on_tick_safe(
                {
                    "symbol": selected_ce,
                    "last_price": 112.0 + i,
                    "timestamp": time.time(),
                    "source": "ws",
                }
            )
            assert _wait_until(
                lambda: runner_obj._selected_candidate_eval_completed_count >= i + 1,
                timeout=3.0,
            )
        assert strategy_manager.generate_signal.call_count == 2
        assert (
            runner_obj.get_entry_eval_lifecycle_snapshot()[
                "selected_strategy_manager_eval_count"
            ]
            == 2
        )
        assert runner_obj._data_phase[selected_ce] == "HYDRATION"
        assert selected_ce not in runner_obj._live_bar_seen
        assert order_manager.submit.call_count == 0
    finally:
        _stop_loop(loop, thread)


def test_hydration_intrabar_never_promotes_stale_or_nonselected_options(
    monkeypatch,
):
    """Only fresh selected contracts are eligible for the hydration exception."""
    runner_obj, _strategy, _risk, _order, selected_ce = _selected_option_runner(
        monkeypatch
    )
    nonselected_ce = "NFO:NIFTY26JUN24100CE"
    for symbol, is_fresh in ((selected_ce, False), (nonselected_ce, True)):
        runner_obj._data_phase[symbol] = "HYDRATION"
        runner_obj._is_option_symbol_tick_fresh = lambda _symbol, fresh=is_fresh: fresh
        reason = StrategyRunner._same_bar_eval_reason(
            runner_obj,
            symbol=symbol,
            price=112.0,
            tick={"bid": 111.0, "ask": 113.0},
            candle_count=100,
            now_ts=time.time(),
        )
        assert reason is None
        assert (
            runner_obj._last_same_bar_eval_block_reason_by_symbol[symbol]
            == "intrabar_eval_non_live_phase"
        )


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
    # A valid completed-bar anchor is required to reach StrategyManager.
    # The synthetic MDM has no book, so bypass that pre-gate only in this test.
    runner_obj._last_bar_ts[selected_ce] = datetime.now(timezone.utc)
    monkeypatch.setenv("REQUIRE_BID_ASK_FOR_OPTION_EVAL", "false")
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
        # A completed worker callback alone does not prove strategy execution.
        assert strategy_manager.generate_signal.call_count >= 1
        snapshot = runner_obj.get_entry_eval_lifecycle_snapshot()
        assert (
            snapshot["selected_strategy_manager_eval_count"]
            == strategy_manager.generate_signal.call_count
        )
        assert snapshot["selected_strategy_manager_eval_age_s"] is not None
    finally:
        _stop_loop(loop, thread)


def test_pre_gate_worker_completion_is_not_counted_as_strategy_evaluation(
    monkeypatch,
):
    runner_obj, strategy_manager, _risk, _order, selected_ce = _selected_option_runner(
        monkeypatch
    )
    # A worker can return without reaching the strategy manager (e.g. a
    # same-bar/readiness gate). This must never masquerade as a strategy call.
    runner_obj._on_tick = Mock(return_value=None)
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
        snapshot = runner_obj.get_entry_eval_lifecycle_snapshot()
        assert snapshot["selected_strategy_manager_eval_count"] == 0
        assert snapshot["selected_strategy_manager_eval_age_s"] is None
        strategy_manager.generate_signal.assert_not_called()
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
