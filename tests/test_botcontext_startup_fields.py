from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from nifty_scalper_bot.core.app import BotContext


def test_botcontext_has_startup_spot_fields() -> None:
    names = {field.name for field in fields(BotContext)}
    assert 'startup_spot_refresh_done' in names
    assert 'startup_spot_listener_registered' in names


def test_assigning_startup_fields_does_not_raise() -> None:
    ctx = SimpleNamespace(startup_spot_refresh_done=False, startup_spot_listener_registered=False)
    ctx.startup_spot_refresh_done = True
    ctx.startup_spot_listener_registered = True
    assert ctx.startup_spot_refresh_done is True
    assert ctx.startup_spot_listener_registered is True
