"""Tests covering the trading switch operator controls."""

from __future__ import annotations

import types

import pytest

from nifty_scalper_bot.core import boot_readiness_safety
from nifty_scalper_bot.core import trading_switch as switch_module
from nifty_scalper_bot.core.trading_switch import TradingSwitch, trading_switch


def test_new_trading_switch_starts_disabled() -> None:
    switch = TradingSwitch()

    assert switch.can_trade() is False
    assert switch.snapshot().enabled is False


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


def test_runtime_arm_enables_only_pristine_switch() -> None:
    switch = TradingSwitch()

    assert switch.arm_for_runtime() is True
    assert switch.can_trade() is True

    switch.pause()
    assert switch.arm_for_runtime() is False
    assert switch.can_trade() is False

    switch.resume()
    assert switch.can_trade() is True


def test_runtime_arm_does_not_clear_active_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switch = TradingSwitch()
    fake_time = types.SimpleNamespace(value=100.0)
    monkeypatch.setattr(switch_module.time, "time", lambda: fake_time.value)
    switch.cooldown(5.0)

    assert switch.arm_for_runtime() is False
    assert switch.can_trade() is False
    assert switch.remaining() == pytest.approx(5.0)


def test_live_readiness_bootstraps_pristine_trading_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switch = TradingSwitch()
    monkeypatch.setattr(
        boot_readiness_safety,
        "trading_switch",
        lambda: switch,
        raising=False,
    )

    wrapped = boot_readiness_safety.adapt_compute_live_readiness(
        lambda **_kwargs: (True, [])
    )
    armed, reasons = wrapped(live_mode=True, market_open=True)

    assert armed is True
    assert reasons == []
    assert switch.can_trade() is True


def test_live_readiness_reports_operator_paused_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switch = TradingSwitch()
    switch.resume()
    switch.pause()
    monkeypatch.setattr(
        boot_readiness_safety,
        "trading_switch",
        lambda: switch,
        raising=False,
    )

    wrapped = boot_readiness_safety.adapt_compute_live_readiness(
        lambda **_kwargs: (True, [])
    )
    armed, reasons = wrapped(live_mode=True, market_open=True)

    assert armed is False
    assert "trading_switch_off" in reasons
    assert switch.can_trade() is False
