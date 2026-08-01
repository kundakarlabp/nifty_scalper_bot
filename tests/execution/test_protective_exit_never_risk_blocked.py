"""Protective exits must survive a tripped risk breaker (P1)."""

from __future__ import annotations

import inspect
import re

from nifty_scalper_bot.execution import bracket_core, order_manager_core
from nifty_scalper_bot.risk import entry_guard_patch


def test_tripped_breaker_blocks_every_order_at_the_risk_manager() -> None:
    """Documents why exits must not reach check_order at all."""
    source = inspect.getsource(entry_guard_patch._ORIGINAL_CHECK_ORDER)
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
