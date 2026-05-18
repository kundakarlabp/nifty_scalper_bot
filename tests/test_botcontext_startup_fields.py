from __future__ import annotations

from dataclasses import fields
from nifty_scalper_bot.core.app import BotContext


def test_botcontext_has_startup_spot_fields() -> None:
    names = {field.name for field in fields(BotContext)}
    assert 'startup_spot_refresh_done' in names
    assert 'startup_spot_listener_registered' in names
    assert 'execution_ready_by_symbol' in names
    assert 'selected_ce_exec_ready' in names
    assert 'selected_pe_exec_ready' in names
    assert 'context_exec_ready' in names
    assert 'broker_ready' in names


def test_assigning_startup_fields_does_not_raise() -> None:
    ctx = object.__new__(BotContext)
    ctx.execution_ready_by_symbol = {}
    ctx.selected_ce_exec_ready = True
    ctx.selected_pe_exec_ready = False
    assert isinstance(ctx.execution_ready_by_symbol, dict)
    assert ctx.selected_ce_exec_ready is True
