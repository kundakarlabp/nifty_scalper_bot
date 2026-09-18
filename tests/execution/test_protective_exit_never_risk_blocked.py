# fmt: off
# ruff: noqa: E501,I001,F841
"""Protective exits must survive a tripped risk breaker (P1)."""

from __future__ import annotations

import inspect
import re

from nifty_scalper_bot.execution import bracket_core, order_manager_core
from nifty_scalper_bot.risk.risk_manager import RiskManager


def test_tripped_breaker_blocks_every_order_at_the_risk_manager() -> None:
    """Documents why exits must not reach check_order at all."""
    source = inspect.getsource(RiskManager._check_order_core)
    assert "if self._breaker_tripped:" in source
    assert "Stop loss required" in source


def test_bracket_exit_paths_bypass_the_risk_manager() -> None:
    for func in (
        bracket_core.BracketManager.submit_exit_order,
        bracket_core.BracketManager._market_fallback_exit,
    ):
        source = inspect.getsource(func)
        assert "place_order" in source
        assert re.search(r"check_risk[\"']?\s*[:=]\s*False", source), func.__name__


def test_reducing_intent_disables_the_risk_check_structurally() -> None:
    source = inspect.getsource(order_manager_core.OrderManager.place_order)
    guard = source.index("_REDUCING_ORDER_INTENTS")
    risk_call = source.index("self._risk_manager.check_order")
    # The bypass must be evaluated before the risk manager is consulted.
    assert guard < risk_call
    assert "EXIT" in order_manager_core._REDUCING_ORDER_INTENTS
    assert "SQUARE_OFF" in order_manager_core._REDUCING_ORDER_INTENTS


def test_validate_close_position_remains_available_after_breaker(monkeypatch) -> None:
    manager = RiskManager.__new__(RiskManager)
    manager._breaker_tripped = True
    manager._breaker_reason = "daily loss"
    monkeypatch.setattr(RiskManager, "_reset_daily_if_needed", lambda self: None)
    monkeypatch.setattr(RiskManager, "_refresh_realized_pnl", lambda self: None)

    allowed, reason = manager.validate_close_position(
        symbol="NFO:NIFTY24JUL24000CE",
        exit_price=95.0,
    )

    assert allowed is True
    assert reason == ""


def test_legacy_safety_bracket_uses_explicit_reducing_intent() -> None:
    source = inspect.getsource(order_manager_core.OrderManager._ensure_safety_bracket)
    assert source.count('intent="EXIT"') >= 2
    assert source.count("check_risk=False") >= 2
    assert source.count('strategy_name="protective_exit"') >= 2

# fmt: on
