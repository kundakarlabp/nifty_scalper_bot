from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from nifty_scalper_bot.core import app

# The previous selected pair remains authoritative until both new delivery edges are proven.


def _ctx(old_ce: str, old_pe: str) -> SimpleNamespace:
    return SimpleNamespace(
        selected_ce=old_ce,
        selected_pe=old_pe,
        atm_ce_symbol=old_ce,
        atm_pe_symbol=old_pe,
        active_symbol_tokens={old_ce: 1, old_pe: 2},
        active_trading_universe={},
        active_contract_basket={},
        market_data_manager=SimpleNamespace(),
        strategy_runner=SimpleNamespace(),
        data_hub=SimpleNamespace(),
        instrument_manager=None,
        broker_client=None,
        option_universe=None,
        strategy_manager=None,
    )


def test_dynamic_basket_does_not_publish_selected_pair_before_delivery_proof(
    monkeypatch,
) -> None:
    old_ce = "NFO:NIFTY26AUG24550CE"
    old_pe = "NFO:NIFTY26AUG24550PE"
    new_ce = "NFO:NIFTY26AUG24600CE"
    new_pe = "NFO:NIFTY26AUG24600PE"
    ctx = _ctx(old_ce, old_pe)
    observed: dict[str, object] = {}

    def _delivery_probe(_ctx, *, selected_ce, selected_pe, reason):
        observed["selected_during_probe"] = (_ctx.selected_ce, _ctx.selected_pe)
        observed["requested"] = (selected_ce, selected_pe)
        observed["reason"] = reason
        return {new_ce: False, new_pe: True}

    monkeypatch.setattr(app, "_ensure_selected_option_runtime_delivery", _delivery_probe)

    basket = {
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26AUGFUT",
        "selected_ce": new_ce,
        "selected_pe": new_pe,
        "atm_ce": new_ce,
        "atm_pe": new_pe,
        "atm_strike": 24600,
        "option_symbols": [new_ce, new_pe],
        "symbols": ["NSE:NIFTY", "NFO:NIFTY26AUGFUT", new_ce, new_pe],
        "token_by_symbol": {new_ce: 101, new_pe: 102},
    }

    selected = app._commit_active_dynamic_basket(
        cast(app.BotContext, ctx),
        basket=basket,
        option_symbols=[new_ce, new_pe],
        symbols=basket["symbols"],
        atm_strike=24600,
    )

    assert observed["selected_during_probe"] == (old_ce, old_pe)
    assert observed["requested"] == (new_ce, new_pe)
    assert observed["reason"] == "dynamic_basket_precommit"
    assert selected == (old_ce, old_pe)
    assert (ctx.selected_ce, ctx.selected_pe) == (old_ce, old_pe)
    assert ctx.active_symbol_tokens == {old_ce: 1, old_pe: 2}


def test_dynamic_basket_publishes_pair_after_delivery_proof(monkeypatch) -> None:
    old_ce = "NFO:NIFTY26AUG24550CE"
    old_pe = "NFO:NIFTY26AUG24550PE"
    new_ce = "NFO:NIFTY26AUG24600CE"
    new_pe = "NFO:NIFTY26AUG24600PE"
    ctx = _ctx(old_ce, old_pe)
    observed: dict[str, object] = {}

    def _delivery_probe(_ctx, *, selected_ce, selected_pe, reason):
        observed["selected_during_probe"] = (_ctx.selected_ce, _ctx.selected_pe)
        return {new_ce: True, new_pe: True}

    monkeypatch.setattr(app, "_ensure_selected_option_runtime_delivery", _delivery_probe)

    basket = {
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26AUGFUT",
        "selected_ce": new_ce,
        "selected_pe": new_pe,
        "atm_ce": new_ce,
        "atm_pe": new_pe,
        "atm_strike": 24600,
        "option_symbols": [new_ce, new_pe],
        "symbols": ["NSE:NIFTY", "NFO:NIFTY26AUGFUT", new_ce, new_pe],
        "token_by_symbol": {new_ce: 101, new_pe: 102},
    }

    selected = app._commit_active_dynamic_basket(
        cast(app.BotContext, ctx),
        basket=basket,
        option_symbols=[new_ce, new_pe],
        symbols=basket["symbols"],
        atm_strike=24600,
    )

    assert observed["selected_during_probe"] == (old_ce, old_pe)
    assert selected == (new_ce, new_pe)
    assert (ctx.selected_ce, ctx.selected_pe) == (new_ce, new_pe)
