from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner


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


def test_futures_context_cannot_trigger_entry():
    assert runner()._symbol_may_trigger_entry("NFO:NIFTY26JUNFUT") is False


def test_spot_context_cannot_trigger_entry():
    assert runner()._symbol_may_trigger_entry("NSE:NIFTY") is False


def test_non_selected_option_context_cannot_trigger_entry():
    assert runner()._symbol_may_trigger_entry("NFO:NIFTY26JUN24100CE") is False


def test_selected_option_can_trigger_entry():
    assert runner()._symbol_may_trigger_entry("NFO:NIFTY26JUN24000CE") is True


def test_open_position_role_can_trigger_management_path():
    r = runner()
    r._position_manager = SimpleNamespace(
        has_open_position=lambda s: s.endswith("24100CE")
    )
    assert r._symbol_may_trigger_entry("NFO:NIFTY26JUN24100CE") is True
