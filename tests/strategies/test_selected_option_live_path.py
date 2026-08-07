"""Regression coverage for the live selected-option evaluation path."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import (
    apply_patches,
)
from nifty_scalper_bot.strategies.runner import (
    EntryEvaluationRoute,
    SymbolRuntimeState,
    SymbolState,
)
from tests.strategies.test_runner_symbol_role_gate import _build_phase9_runner


def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def _wait_until(predicate, *, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_live_selected_option_tick_reaches_strategy_manager_after_atm_switch(monkeypatch):
    """A dynamically selected option must not be blocked by the startup snapshot."""
    apply_patches()
    runner, strategy_manager, risk_manager, order_manager, old_ce = (
        _build_phase9_runner(monkeypatch)
    )
    old_pe = "NFO:NIFTY26JUN24000PE"
    selected_ce = "NFO:NIFTY26JUN24100CE"
    selected_pe = "NFO:NIFTY26JUN24100PE"

    # Production starts with a frozen startup snapshot. Dynamic ATM selection
    # later makes a previously contextual contract active; the startup snapshot
    # must not veto that authoritative dynamic universe update.
    runner._universe_dynamic_mode = True
    runner._frozen_universe = {"NSE:NIFTY", old_ce, old_pe}

    for symbol in (old_pe, selected_ce, selected_pe):
        runner._active_symbols.add(symbol)
        runner._tracked_symbols.add(symbol)
        runner._history_ready_by_symbol[symbol] = True
        runner._data_phase[symbol] = "LIVE"
        runner._symbol_history[symbol] = [{"timestamp": time.time()}]
        runner._last_bar_ts[symbol] = datetime.now(timezone.utc)
        runner._symbol_state[symbol] = SymbolRuntimeState(symbol, 100)
        runner._symbol_state[symbol].active = True
        runner._symbol_states[symbol] = SymbolState.READY
        runner._active_basket_token_by_symbol[symbol] = 1

    runner.set_active_option_context(
        selected_ce=selected_ce,
        selected_pe=selected_pe,
        atm_strike=24100,
        option_symbols=[old_ce, old_pe, selected_ce, selected_pe],
    )
    runner._refresh_underlying_context_snapshots = lambda **_kwargs: None
    runner._get_cached_quote_for_live_entry = lambda _symbol: {
        "symbol": _symbol,
        "ltp": 100.0,
        "last_price": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "timestamp": time.time(),
    }

    assert runner._entry_evaluation_route(selected_ce) == EntryEvaluationRoute.OPTION_CANDIDATE

    loop, thread = _run_loop_in_thread()
    runner.attach_runtime_loop(loop)
    try:
        runner._on_tick_safe(
            {
                "symbol": selected_ce,
                "last_price": 100.0,
                "ltp": 100.0,
                "bid": 99.5,
                "ask": 100.5,
                "timestamp": time.time(),
                "source": "ws",
                "trace_id": "selected-option-live-path",
            }
        )

        assert _wait_until(
            lambda: any(
                call.args and call.args[0] == selected_ce
                for call in strategy_manager.generate_signal.call_args_list
            )
        )
        risk_manager.validate.assert_called()
        order_manager.submit.assert_called()
    finally:
        _stop_loop(loop, thread)
