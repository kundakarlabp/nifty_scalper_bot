from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.strategies.runner import EntryEvaluationRoute

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.simulation_component]


def _mark_activation(
    system, symbol: str, *, state: str = "ACTIVE", token: int | None = None
) -> None:
    token = int(token or system.exchange.instruments[symbol].token)
    mdm = system.market_data
    mdm.register_symbol(symbol, token)
    mdm.request_token_subscription(token, symbol=symbol)
    generation = getattr(mdm, "_symbol_subscription_generation", {}).get(symbol, 1)
    with mdm._lock:  # noqa: SLF001 - post-885 lifecycle proof against production state
        mdm._tracked_symbols.add(symbol)  # noqa: SLF001
        mdm._symbol_to_token[symbol] = token  # noqa: SLF001
        mdm._token_by_symbol[symbol] = token  # noqa: SLF001
        mdm._token_to_symbol[token] = symbol  # noqa: SLF001
        mdm._desired_tokens.add(token)  # noqa: SLF001
        mdm._symbol_subscription_generation[symbol] = generation  # noqa: SLF001
        mdm._symbol_token_generation[symbol] = (token, generation)  # noqa: SLF001
        mdm._dispatched_subscriptions.discard(token)  # noqa: SLF001
        mdm._confirmed_subscriptions.discard(token)  # noqa: SLF001
        mdm._symbol_first_tick_generation.pop(symbol, None)  # noqa: SLF001
        mdm._last_valid_live_tick_mono.pop(symbol, None)  # noqa: SLF001
        if state in {"DISPATCHED", "CONFIRMED", "OLD_GENERATION", "ACTIVE"}:
            mdm._dispatched_subscriptions.add(token)  # noqa: SLF001
        if state in {"CONFIRMED", "OLD_GENERATION", "ACTIVE"}:
            mdm._confirmed_subscriptions.add(token)  # noqa: SLF001
        if state == "OLD_GENERATION":
            mdm._symbol_first_tick_generation[symbol] = generation - 1  # noqa: SLF001
            mdm._last_valid_live_tick_mono[symbol] = time.monotonic()  # noqa: SLF001
        if state == "ACTIVE":
            mdm._symbol_first_tick_generation[symbol] = generation  # noqa: SLF001
            mdm._last_valid_live_tick_mono[symbol] = time.monotonic()  # noqa: SLF001
    system.runner._active_symbols.add(symbol)  # noqa: SLF001
    system.runner._tracked_symbols.add(symbol)  # noqa: SLF001
    system.runner._mdm_callback_registered = True  # noqa: SLF001
    system.runner._active_basket_token_by_symbol[symbol] = token  # noqa: SLF001
    system.runner._history_ready_by_symbol[symbol] = True  # noqa: SLF001
    system.runner._symbol_history[symbol] = system.market_data.get_ohlc_bars(
        symbol, limit=100
    )  # noqa: SLF001


def _eligibility(system, symbol: str, bias: str):
    return system.runner._live_entry_candidate_eligibility(  # noqa: SLF001
        symbol, direction_bias=bias
    )


def _bootstrap(system):
    system.start()
    system.hydrate_via_production_path()
    _mark_activation(system, system.scenario.ce_symbol)
    _mark_activation(system, system.scenario.pe_symbol)


def test_underlying_evaluation_not_blocked_by_option_readiness(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    assert (
        system.runner._entry_evaluation_route(
            system.scenario.spot_symbol
        )  # noqa: SLF001
        == EntryEvaluationRoute.UNDERLYING
    )
    # Underlying stage must not require the option activation proof.
    system.runner._live_symbol_activation = pytest.fail  # type: ignore[method-assign]  # noqa: SLF001
    assert (
        system.runner._entry_evaluation_route(
            system.scenario.spot_symbol
        )  # noqa: SLF001
        == EntryEvaluationRoute.UNDERLYING
    )


def test_bullish_spot_future_selects_ce(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    allowed, reason, details = _eligibility(system, system.scenario.ce_symbol, "CE")
    assert allowed is True
    assert reason == "candidate_selected_option"
    assert details["contract_side"] == "CE"


def test_bearish_spot_future_selects_pe(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    allowed, reason, details = _eligibility(system, system.scenario.pe_symbol, "PE")
    assert allowed is True
    assert reason == "candidate_selected_option"
    assert details["contract_side"] == "PE"


def test_spot_future_disagreement_blocks_or_follows_configured_policy(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    category, reason = system.runner._classify_no_trade_decision(  # noqa: SLF001
        symbol=system.scenario.spot_symbol,
        signal=None,
        indicators_ctx={
            "direction_bias": "CE",
            "spot_direction_bias": "CE",
            "futures_direction_bias": "PE",
            "underlying_direction_conflict": True,
        },
        option_count=100,
        option_required=50,
    )
    assert category == "context_direction_conflict"
    assert reason == "underlying_direction_conflict"


def test_option_premium_cannot_reverse_underlying_direction(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    allowed, reason, details = _eligibility(system, system.scenario.ce_symbol, "PE")
    assert allowed is False
    assert reason == "context_direction_conflict"
    assert details["direction_bias"] == "PE"


def test_requested_subscription_is_not_active(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    _mark_activation(system, system.scenario.ce_symbol, state="REQUESTED")
    activation = system.runner._live_symbol_activation(
        system.scenario.ce_symbol
    )  # noqa: SLF001
    assert activation.executable is False
    assert "subscription_pending" in activation.blockers


def test_confirmed_without_current_generation_tick_is_not_active(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    _mark_activation(system, system.scenario.ce_symbol, state="CONFIRMED")
    activation = system.runner._live_symbol_activation(
        system.scenario.ce_symbol
    )  # noqa: SLF001
    assert activation.executable is False
    assert "current_generation_tick_pending" in activation.blockers


def test_context_only_symbol_does_not_enter_phase9_trigger_evaluation(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    assert (
        system.runner._entry_evaluation_route(
            system.scenario.future_symbol
        )  # noqa: SLF001
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_selected_option_enters_phase9_only_when_fully_active(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    _mark_activation(system, system.scenario.ce_symbol, state="CONFIRMED")
    pending = system.runner._live_symbol_activation(
        system.scenario.ce_symbol
    )  # noqa: SLF001
    _mark_activation(system, system.scenario.ce_symbol, state="ACTIVE")
    active = system.runner._live_symbol_activation(
        system.scenario.ce_symbol
    )  # noqa: SLF001
    assert pending.executable is False
    assert active.executable is True
    assert (
        system.runner._entry_evaluation_route(system.scenario.ce_symbol)  # noqa: SLF001
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )


def test_unselected_option_is_role_gated(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    assert (
        system.runner._entry_evaluation_route("NFO:NIFTY26JUL25100CE")  # noqa: SLF001
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_atm_rotation_rehydrates_and_reactivates_new_contracts(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    old_ce = system.scenario.ce_symbol
    new_ce = "NFO:NIFTY26JUL25050CE"
    system.runner.set_active_option_context(  # production basket commit surface
        selected_ce=new_ce,
        selected_pe=system.scenario.pe_symbol,
        atm_strike=25050,
        option_symbols=[new_ce, system.scenario.pe_symbol],
    )
    system.exchange.add_instrument(
        type(system.exchange.instruments[old_ce])(
            new_ce, 500004, "NFO", "CE", 25050, "2026-07-30", 75, 0.05
        )
    )
    system.history.set_history(new_ce, system.history.fetch_history(old_ce, limit=100))
    frame = system.history.fetch_history(new_ce, limit=100)
    system.market_data.ingest_historical_ohlc(new_ce, frame.to_dict(orient="records"))
    system.runner.sync_history_from_mdm(
        new_ce, required_bars=50, reason="basket_rotation", role="selected_option"
    )
    _mark_activation(system, new_ce, state="ACTIVE", token=500004)
    assert system.runner._active_selected_ce == new_ce  # noqa: SLF001
    assert (
        system.runner._live_symbol_activation(new_ce).executable is True
    )  # noqa: SLF001


def test_old_candidate_rejected_after_basket_rotation(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    old_ce = system.scenario.ce_symbol
    system.runner.set_active_option_context(
        selected_ce="NFO:NIFTY26JUL25050CE",
        selected_pe=system.scenario.pe_symbol,
        atm_strike=25150,
        option_symbols=["NFO:NIFTY26JUL25050CE", system.scenario.pe_symbol],
    )
    system.runner._active_option_symbols = {  # noqa: SLF001
        "NFO:NIFTY26JUL25050CE",
        system.scenario.pe_symbol,
    }
    system.runner._active_basket_all_symbols = set(
        system.runner._active_option_symbols
    )  # noqa: SLF001
    system.runner._active_selected_ce = "NFO:NIFTY26JUL25050CE"  # noqa: SLF001
    system.runner._active_atm_strike = 25150  # noqa: SLF001
    system.runner._active_basket_token_by_symbol.pop(old_ce, None)  # noqa: SLF001
    allowed, reason, _details = _eligibility(system, old_ce, "CE")
    assert allowed is False
    assert reason in {
        "candidate_not_in_active_basket",
        "candidate_not_selected_or_near_atm",
    }


def test_open_position_exit_continues_after_direction_context_degrades(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    symbol = system.scenario.ce_symbol
    system.position_manager.has_open_position = lambda s: s == symbol  # type: ignore[method-assign]
    assert (
        system.runner._entry_evaluation_route(symbol)  # noqa: SLF001
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )
    # Bracket ticks are routed before entry-readiness gates, so
    # protective handling remains callable.
    system.bracket_manager.on_tick(symbol, 95.0)


def test_mdm_history_is_authoritative_over_indicator_diagnostics(live_sim_system):
    system = live_sim_system
    _bootstrap(system)
    symbol = system.scenario.ce_symbol
    assert system.market_data.get_latest_closed_bar(symbol) is not None
    # Diagnostic divergence: indicator history can be present while
    # canonical MDM is intentionally removed.
    indicator_count = system.indicator_engine.history_count(symbol)
    with system.market_data._lock:  # noqa: SLF001
        system.market_data._ohlc.pop(symbol, None)  # noqa: SLF001
    assert indicator_count > 0
    assert system.market_data.get_latest_closed_bar(symbol) is None
