"""Regression coverage for the live selected-option evaluation path."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

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


def test_basket_sync_defers_cold_selected_pair_until_indicator_history_is_ready(monkeypatch):
    """Basket SSOT must not replace the executable pair before both new legs are warm."""
    apply_patches()
    runner, _strategy_manager, _risk_manager, _order_manager, old_ce = (
        _build_phase9_runner(monkeypatch)
    )
    old_pe = runner._active_selected_pe
    new_ce = "NFO:NIFTY26JUN24100CE"
    new_pe = "NFO:NIFTY26JUN24100PE"
    histories = {
        old_ce: [{}] * 100,
        old_pe: [{}] * 100,
        new_ce: [],
        new_pe: [],
    }
    runner._indicator_engine.get_history = lambda symbol: histories.get(symbol, [])
    runner._option_required_bars = 20
    runner._prewarm_active_option_history = lambda **_kwargs: None
    selection = SimpleNamespace(
        selected_ce=new_ce,
        selected_pe=new_pe,
        atm_strike=24100,
        option_symbols=(new_ce, new_pe),
        basket_version="test-cold-switch",
        selected_at=time.time(),
        source="test",
    )

    runner._sync_active_selection_from_basket(selection)

    assert runner._active_selected_ce == old_ce
    assert runner._active_selected_pe == old_pe
    assert runner._pending_selected_ce == new_ce
    assert runner._pending_selected_pe == new_pe

    histories[new_ce] = [{}] * 20
    histories[new_pe] = [{}] * 20
    assert runner._maybe_promote_pending_active_basket(source="test") is True
    assert runner._active_selected_ce == new_ce
    assert runner._active_selected_pe == new_pe
    assert runner._pending_selected_ce is None
    assert runner._pending_selected_pe is None


def test_quote_versions_do_not_advance_candle_version_or_starve_same_bar_evaluation(monkeypatch):
    """Quote/data sequence counters are not candle identity and must not suppress a leg."""
    apply_patches()
    runner, strategy_manager, _risk_manager, _order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner._active_symbols.add(selected_ce)
    runner._tracked_symbols.add(selected_ce)
    runner._history_ready_by_symbol[selected_ce] = True
    runner._data_phase[selected_ce] = "LIVE"
    runner._symbol_history[selected_ce] = [{"timestamp": time.time()}]
    fixed_bar_ts = datetime.now(timezone.utc)
    runner._last_bar_ts[selected_ce] = fixed_bar_ts
    runner._symbol_state[selected_ce] = SymbolRuntimeState(selected_ce, 100)
    runner._symbol_state[selected_ce].active = True
    runner._symbol_states[selected_ce] = SymbolState.READY
    runner._active_basket_token_by_symbol[selected_ce] = 1
    runner._refresh_underlying_context_snapshots = lambda **_kwargs: None
    runner._get_cached_quote_for_live_entry = lambda _symbol: {
        "symbol": _symbol,
        "ltp": 100.0,
        "last_price": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "timestamp": time.time(),
    }

    first_tick = {
        "symbol": selected_ce,
        "last_price": 100.0,
        "ltp": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "timestamp": time.time(),
        "source": "ws",
        "trace_id": "quote-version-1",
        "version": 1,
    }
    second_tick = {
        **first_tick,
        "timestamp": time.time() + 0.1,
        "trace_id": "quote-version-2",
        "version": 2,
        "data_version": 22,
    }

    runner._on_tick(selected_ce, first_tick)
    runner._on_tick(selected_ce, second_tick)

    calls = [
        call
        for call in strategy_manager.generate_signal.call_args_list
        if call.args and call.args[0] == selected_ce
    ]
    assert len(calls) == 2
    assert runner._candle_versions.get(selected_ce, 0) == 0
    assert first_tick["version"] == 1
    assert second_tick["data_version"] == 22
