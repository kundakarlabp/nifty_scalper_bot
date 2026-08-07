"""Regression tests for off-market dynamic basket mutation safety."""

from __future__ import annotations

from types import SimpleNamespace

import nifty_scalper_bot.core.off_market_basket_safety as safety
from nifty_scalper_bot.core.universe_controller import UniverseController
from nifty_scalper_bot.utils.market_hours import MarketState


def test_existing_universe_does_not_mutate_while_market_is_closed(monkeypatch) -> None:
    safety.apply_patches()
    monkeypatch.setattr(safety, "get_market_state", lambda: MarketState.CLOSED, raising=False)
    controller = UniverseController()

    controller.update(["OLD_CE", "OLD_PE"])
    added, removed = controller.update(["NEW_CE", "NEW_PE"])

    assert added == set()
    assert removed == set()
    assert controller.current_universe == {"OLD_CE", "OLD_PE"}
    assert controller.previous_universe == set()


def test_existing_universe_can_change_when_market_opens(monkeypatch) -> None:
    safety.apply_patches()
    states = iter([MarketState.CLOSED, MarketState.OPEN])
    monkeypatch.setattr(safety, "get_market_state", lambda: next(states), raising=False)
    controller = UniverseController()

    controller.update(["OLD_CE", "OLD_PE"])
    added, removed = controller.update(["NEW_CE", "NEW_PE"])

    assert added == {"NEW_CE", "NEW_PE"}
    assert removed == {"OLD_CE", "OLD_PE"}
    assert controller.current_universe == {"NEW_CE", "NEW_PE"}


def test_existing_basket_selection_commit_is_preserved_off_market(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def original_commit(ctx, **kwargs):
        calls.append(dict(kwargs))
        return kwargs["basket"]["selected_ce"], kwargs["basket"]["selected_pe"]

    app_module = SimpleNamespace(_commit_active_dynamic_basket=original_commit)
    safety.apply_app_patch(app_module)
    monkeypatch.setattr(safety, "get_market_state", lambda: MarketState.PREOPEN, raising=False)
    ctx = SimpleNamespace(
        active_contract_basket={
            "selected_ce": "OLD_CE",
            "selected_pe": "OLD_PE",
        },
        active_trading_universe={},
        selected_ce="OLD_CE",
        selected_pe="OLD_PE",
    )

    selected = app_module._commit_active_dynamic_basket(
        ctx,
        basket={"selected_ce": "NEW_CE", "selected_pe": "NEW_PE"},
        option_symbols=["NEW_CE", "NEW_PE"],
        symbols=["NEW_CE", "NEW_PE"],
        atm_strike=25000,
    )

    assert selected == ("OLD_CE", "OLD_PE")
    assert calls == []
    assert ctx.selected_ce == "OLD_CE"
    assert ctx.selected_pe == "OLD_PE"


def test_initial_basket_commit_is_allowed_off_market(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def original_commit(ctx, **kwargs):
        calls.append(dict(kwargs))
        return kwargs["basket"]["selected_ce"], kwargs["basket"]["selected_pe"]

    app_module = SimpleNamespace(_commit_active_dynamic_basket=original_commit)
    safety.apply_app_patch(app_module)
    monkeypatch.setattr(safety, "get_market_state", lambda: MarketState.CLOSED, raising=False)
    ctx = SimpleNamespace(
        active_contract_basket=None,
        active_trading_universe={},
        selected_ce=None,
        selected_pe=None,
    )

    selected = app_module._commit_active_dynamic_basket(
        ctx,
        basket={"selected_ce": "NEW_CE", "selected_pe": "NEW_PE"},
        option_symbols=["NEW_CE", "NEW_PE"],
        symbols=["NEW_CE", "NEW_PE"],
        atm_strike=25000,
    )

    assert selected == ("NEW_CE", "NEW_PE")
    assert len(calls) == 1
