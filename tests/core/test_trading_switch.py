"""Tests covering the trading switch operator controls."""

from __future__ import annotations

import types

import pytest

from nifty_scalper_bot.core import trading_switch as switch_module
from nifty_scalper_bot.core.trading_switch import trading_switch


def test_trading_switch_pause_blocks_until_resume() -> None:
    switch = trading_switch()
    switch.resume()
    switch.pause()
    snapshot = switch.snapshot()
    assert not snapshot.can_trade
    switch.resume()
    assert switch.can_trade()


def test_trading_switch_cooldown_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    switch = trading_switch()
    switch.resume()
    fake_time = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(switch_module.time, "time", lambda: fake_time.value)
    switch.cooldown(5.0)
    assert not switch.can_trade()
    fake_time.value += 6.0
    assert switch.can_trade()
