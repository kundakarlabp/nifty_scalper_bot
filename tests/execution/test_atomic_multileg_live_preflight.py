"""Regression checks for the canonical sequential multi-leg LIVE fail-close."""

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.order_manager_core import OrderPlacementError


def test_live_two_leg_atomic_helper_rejects_before_normalization_or_broker():
    calls = []
    owner = SimpleNamespace(
        is_live_mode=lambda: True,
        _normalize_leg=lambda leg: calls.append(("normalized", leg)),
        _ensure_trading_allowed=lambda **kwargs: calls.append(("gate", kwargs)),
        _place_single_order=lambda **kwargs: calls.append(("broker", kwargs)),
    )
    legs = (
        {"symbol": "NFO:NIFTY26NOV24000CE", "side": "BUY", "quantity": 65},
        {"symbol": "NFO:NIFTY26NOV24000PE", "side": "BUY", "quantity": 65},
    )
    with pytest.raises(
        OrderPlacementError, match="linked_leg_live_execution_not_supported"
    ):
        OrderManager.place_atomic_entry(owner, legs)
    assert calls == []


def test_nonlive_two_leg_helper_preserves_existing_leg_count_gate():
    owner = SimpleNamespace(is_live_mode=lambda: False)
    with pytest.raises(ValueError, match="exactly two legs"):
        OrderManager.place_atomic_entry(owner, [])


def test_live_four_leg_condor_cannot_bypass_two_leg_guard():
    owner = SimpleNamespace(is_live_mode=lambda: True)
    legs = [
        {"symbol": f"NFO:NIFTY26NOV{strike}{side}", "side": action, "quantity": 65}
        for strike, side, action in (
            (23800, "PE", "BUY"),
            (23900, "PE", "SELL"),
            (24100, "CE", "SELL"),
            (24200, "CE", "BUY"),
        )
    ]
    with pytest.raises(
        OrderPlacementError, match="linked_leg_live_execution_not_supported"
    ):
        OrderManager.place_atomic_entry(owner, legs)
