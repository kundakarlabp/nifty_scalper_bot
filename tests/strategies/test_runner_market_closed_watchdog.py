"""Tests for the market-session-aware watchdog patches inside StrategyRunner.

Rather than instantiating the full runner (heavy dependency surface),
these tests exercise the helper used by the watchdog and assert that the
stale-threshold path produces sensible values. Behavioral checks for the
runner's stall-suppression and zombie-WS-restart logic are kept structural
to avoid pulling in the full broker stack.
"""

from __future__ import annotations

import logging
import re

import pytest

from nifty_scalper_bot.utils import logging as nsb_logging
from nifty_scalper_bot.utils import market_hours


@pytest.fixture(autouse=True)
def _reset_throttle_state():
    with nsb_logging._THROTTLE_LOCK:
        nsb_logging._THROTTLE_STATE.clear()
    yield
    with nsb_logging._THROTTLE_LOCK:
        nsb_logging._THROTTLE_STATE.clear()


def _read_runner_source() -> str:
    import nifty_scalper_bot.strategies.runner as runner

    return open(runner.__file__, "r", encoding="utf-8").read()


def test_runner_uses_central_stale_threshold():
    """Args: none. Returns: None. Raises: AssertionError."""
    source = _read_runner_source()
    assert "stale_threshold_for_symbol(symbol, market_open_now)" in source
    assert "is_market_open_now()" in source
    # Ensure the legacy hardcoded 10.0 NIFTY threshold is gone.
    assert "else (30.0 if source in (\"rest\", \"polling\") else 10.0)" not in source


def test_runner_strategy_stall_check_skipped_when_market_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    source = _read_runner_source()
    # Both stall paths must guard with market-open check and emit the
    # STALL_CHECK_SKIPPED throttle key.
    assert "STALL_CHECK_SKIPPED reason=market_closed" in source
    assert "strategy_stall_check_skipped_market_closed" in source


def test_runner_zombie_ws_restart_skipped_when_market_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    source = _read_runner_source()
    assert "WS_RESTART_SKIPPED" in source
    assert "zombie_ws_restart_skipped_market_closed" in source
    # The actual restart trigger must be guarded by `if not market_open: ... return`
    block = re.search(
        r"if stale_count > 0:\s*\n\s+if not market_open:",
        source,
    )
    assert block is not None, "zombie WS restart must be gated by market_open"


def test_runner_stale_tick_offmarket_uses_central_threshold():
    """Args: none. Returns: None. Raises: AssertionError."""
    source = _read_runner_source()
    assert "stale_tick_offmarket:" in source
    assert "OFFMARKET_STALE_TICK" in source
    # Open-market path must still warn.
    assert "STALE TICK:" in source


def test_runner_offmarket_nifty_stale_threshold_gte_3600():
    """Args: none. Returns: None. Raises: AssertionError."""
    threshold = market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=False)
    assert threshold >= 3600.0
    threshold_open = market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=True)
    assert threshold_open == 120.0
