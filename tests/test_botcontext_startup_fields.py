from __future__ import annotations

from dataclasses import fields
from nifty_scalper_bot.core.app import BotContext


def test_botcontext_has_startup_spot_fields() -> None:
    names = {field.name for field in fields(BotContext)}
    assert 'startup_spot_refresh_done' in names
    assert 'startup_spot_listener_registered' in names
    assert 'data_ready' in names
    assert 'strategy_evaluation_ready' in names
    assert 'trading_signal_ready' in names
    assert 'execution_armed' in names
    assert 'execution_block_reason' in names
    assert 'market_open' in names
    assert 'execution_ready_by_symbol' in names
    assert 'selected_ce_exec_ready' in names
    assert 'selected_pe_exec_ready' in names
    assert 'context_exec_ready' in names
    assert 'broker_ready' in names
    assert 'active_contract_basket' in names
    assert 'active_basket_hydration' in names
    # Basket token map and degraded-mode flag must be declared fields because
    # BotContext uses slots=True — undeclared dynamic assignments raise AttributeError.
    assert 'active_symbol_tokens' in names, (
        "active_symbol_tokens must be a declared BotContext field (slots=True forbids dynamic attrs)"
    )
    assert 'degraded_mode' in names, (
        "degraded_mode must be a declared BotContext field (slots=True forbids dynamic attrs)"
    )


def test_assigning_startup_fields_does_not_raise() -> None:
    ctx = object.__new__(BotContext)
    ctx.data_ready = False
    ctx.strategy_evaluation_ready = False
    ctx.trading_signal_ready = False
    ctx.execution_armed = False
    ctx.execution_block_reason = 'data_not_ready'
    ctx.market_open = False
    ctx.execution_ready_by_symbol = {}
    ctx.selected_ce_exec_ready = True
    ctx.selected_pe_exec_ready = False
    ctx.active_basket_hydration = {'hard_ready': False, 'missing': ['hydrator_missing']}
    assert ctx.data_ready is False
    assert ctx.execution_block_reason == 'data_not_ready'
    assert isinstance(ctx.execution_ready_by_symbol, dict)
    assert ctx.selected_ce_exec_ready is True
    assert ctx.active_basket_hydration['missing'] == ['hydrator_missing']


def test_active_symbol_tokens_and_degraded_mode_assignable() -> None:
    """Regression: slots=True raised AttributeError on dynamic ctx.active_symbol_tokens = {...}."""
    ctx = object.__new__(BotContext)
    # Must not raise AttributeError — these are now declared slots fields.
    ctx.active_symbol_tokens = {'NSE:NIFTY': 256265, 'NFO:NIFTY2660923500CE': 12345678}
    ctx.degraded_mode = True
    assert isinstance(ctx.active_symbol_tokens, dict)
    assert len(ctx.active_symbol_tokens) == 2
    assert ctx.degraded_mode is True

