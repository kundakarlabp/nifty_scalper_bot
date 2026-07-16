from __future__ import annotations

from types import SimpleNamespace

from unittest.mock import Mock

from nifty_scalper_bot.strategies.runner import EntryEvaluationRoute, StrategyRunner


def runner():
    r = StrategyRunner.__new__(StrategyRunner)
    r._active_selected_ce = "NFO:NIFTY26JUN24000CE"
    r._active_selected_pe = "NFO:NIFTY26JUN24000PE"
    r._selected_ce_symbol = None
    r._selected_pe_symbol = None
    r._pending_selected_ce = None
    r._pending_selected_pe = None
    r._active_contract_basket = None
    r._data_hub = None
    r._market_data = None
    r._position_manager = SimpleNamespace(has_open_position=lambda _s: False)
    return r


def test_futures_context_routes_as_underlying_context():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUNFUT")
        == EntryEvaluationRoute.UNDERLYING
    )


def test_spot_context_routes_as_underlying_context():
    assert (
        runner()._entry_evaluation_route("NSE:NIFTY") == EntryEvaluationRoute.UNDERLYING
    )


def test_non_selected_option_context_cannot_trigger_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_selected_option_can_trigger_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24000CE")
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )


def test_open_position_role_can_trigger_management_path():
    r = runner()
    r._position_manager = SimpleNamespace(
        has_open_position=lambda s: s.endswith("24100CE")
    )
    assert (
        r._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )


def test_nifty_underlying_reaches_strategy_manager():
    r = runner()
    strategy_manager = Mock()
    order_manager = Mock()
    selected_ce = "NFO:NIFTY26JUN24000CE"
    strategy_manager.generate_signal.return_value = SimpleNamespace(symbol=selected_ce)

    route = r._entry_evaluation_route("NSE:NIFTY")
    if route == EntryEvaluationRoute.UNDERLYING:
        signal = strategy_manager.generate_signal("NSE:NIFTY", 24000.0, trace_id="t")
        order_manager.submit(signal)

    assert route == EntryEvaluationRoute.UNDERLYING
    strategy_manager.generate_signal.assert_called_once()
    order_manager.submit.assert_called_once()


def test_underlying_does_not_require_option_subscription_activation():
    r = runner()
    r._live_symbol_activation = Mock(
        side_effect=AssertionError("underlying must not be activation gated")
    )

    assert r._entry_evaluation_route("NSE:NIFTY") == EntryEvaluationRoute.UNDERLYING
    r._live_symbol_activation.assert_not_called()


def test_non_selected_option_context_does_not_enter_phase9_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_selected_option_enters_phase9_when_fully_active():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24000CE")
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )


def test_open_position_routes_to_management_not_new_entry():
    r = runner()
    r._position_manager = SimpleNamespace(
        has_open_position=lambda s: s.endswith("24100CE")
    )
    strategy_manager = Mock()
    order_manager = Mock()

    assert (
        r._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )
    strategy_manager.generate_signal.assert_not_called()
    order_manager.submit.assert_not_called()
