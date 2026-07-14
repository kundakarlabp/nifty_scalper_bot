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
import time

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
    assert 'else (30.0 if source in ("rest", "polling") else 10.0)' not in source


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


def test_runner_skips_inactive_option_stale_restart_count():
    """Args: none. Returns: None. Raises: AssertionError."""
    source = _read_runner_source()
    assert "outside_active_option_basket" in source
    assert "is_nifty_option_symbol(symbol)" in source
    assert "self._is_selected_option_symbol(symbol)" in source
    skip_index = source.index("outside_active_option_basket")
    stale_count_index = source.index("stale_count += 1", skip_index)
    assert skip_index < stale_count_index


def test_runner_watchdog_does_not_restart_for_inactive_option(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    import nifty_scalper_bot.strategies.runner as runner_module
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class _MarketData:
        def __init__(self) -> None:
            self.restart_count = 0

        def _trigger_zombie_ws_restart(self) -> None:
            self.restart_count += 1

    old_symbol = "NFO:NIFTY2671424100PE"
    selected_ce = "NFO:NIFTY2671423950CE"
    selected_pe = "NFO:NIFTY2671423950PE"
    market_data = _MarketData()
    now_mono = time.monotonic()
    now_wall = time.time()

    runner = StrategyRunner.__new__(StrategyRunner)
    runner.ready = False
    runner._last_tick_seen_ts = now_mono
    runner._last_global_eval_ts = now_mono
    runner._eval_stall_recovery_attempted = False
    runner._candle_engines = {old_symbol: object()}
    runner._active_symbols = {old_symbol}
    runner._tracked_symbols = {old_symbol}
    runner._last_tick_time_by_symbol = {old_symbol: now_wall - 4000.0}
    runner._active_option_symbols = {selected_ce, selected_pe}
    runner._active_selected_ce = selected_ce
    runner._active_selected_pe = selected_pe
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._active_contract_basket = None
    runner._data_hub = None
    runner._last_ws_stale_log_ts_by_symbol = {}
    runner._last_ws_reconnect_attempt_ts = 0.0
    runner._market_data = market_data
    runner._log_throttle_state = {}
    runner._logger = logging.getLogger("test.runner.watchdog")

    monkeypatch.setattr(runner_module, "is_market_open_now", lambda: True)

    runner._health_watchdog()

    assert market_data.restart_count == 0



def test_runner_watchdog_delegates_restart_decision_to_mdm_transport_gate(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    import nifty_scalper_bot.strategies.runner as runner_module
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class _MarketData:
        def __init__(self) -> None:
            self.restart_count = 0
            self.zombie_checks = 0

        def _check_zombie_ticks(self) -> None:
            self.zombie_checks += 1

        def _trigger_zombie_ws_restart(self) -> None:
            self.restart_count += 1

    symbol = "NFO:NIFTY2671423950CE"
    market_data = _MarketData()
    now_mono = time.monotonic()
    now_wall = time.time()

    runner = StrategyRunner.__new__(StrategyRunner)
    runner.ready = False
    runner._last_tick_seen_ts = now_mono
    runner._last_global_eval_ts = now_mono
    runner._eval_stall_recovery_attempted = False
    runner._candle_engines = {symbol: object()}
    runner._active_symbols = {symbol}
    runner._tracked_symbols = {symbol}
    runner._last_tick_time_by_symbol = {symbol: now_wall - 4000.0}
    runner._active_option_symbols = {symbol}
    runner._active_selected_ce = symbol
    runner._active_selected_pe = None
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._active_contract_basket = None
    runner._data_hub = None
    runner._last_ws_stale_log_ts_by_symbol = {}
    runner._last_ws_reconnect_attempt_ts = 0.0
    runner._market_data = market_data
    runner._log_throttle_state = {}
    runner._logger = logging.getLogger("test.runner.watchdog")

    monkeypatch.setattr(runner_module, "is_market_open_now", lambda: True)

    runner._health_watchdog()

    assert market_data.zombie_checks == 1
    assert market_data.restart_count == 0


def test_runner_watchdog_delegates_never_ticked_required_symbol(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    import nifty_scalper_bot.strategies.runner as runner_module
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class _MarketData:
        def __init__(self) -> None:
            self.zombie_checks = 0
            self.restart_count = 0

        def _required_live_symbols(self) -> set[str]:
            return {"NFO:NIFTY2671423950CE"}

        def time_since_last_live_ws_tick(self, symbol: str):
            assert symbol == "NFO:NIFTY2671423950CE"
            return None

        def _check_zombie_ticks(self) -> None:
            self.zombie_checks += 1

        def _trigger_zombie_ws_restart(self) -> None:
            self.restart_count += 1

    symbol = "NFO:NIFTY2671423950CE"
    market_data = _MarketData()
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.ready = False
    runner._last_tick_seen_ts = time.monotonic()
    runner._last_global_eval_ts = time.monotonic()
    runner._eval_stall_recovery_attempted = False
    runner._candle_engines = {symbol: object()}
    runner._active_symbols = {symbol}
    runner._tracked_symbols = {symbol}
    runner._last_tick_time_by_symbol = {}
    runner._active_option_symbols = {symbol}
    runner._active_selected_ce = symbol
    runner._active_selected_pe = None
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._active_contract_basket = None
    runner._data_hub = None
    runner._last_ws_stale_log_ts_by_symbol = {}
    runner._last_ws_reconnect_attempt_ts = 0.0
    runner._market_data = market_data
    runner._log_throttle_state = {}
    runner._logger = logging.getLogger("test.runner.watchdog")

    monkeypatch.setattr(runner_module, "is_market_open_now", lambda: True)

    runner._health_watchdog()

    assert market_data.zombie_checks == 1
    assert market_data.restart_count == 0


def test_runner_watchdog_excludes_superseded_non_required_option(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    import nifty_scalper_bot.strategies.runner as runner_module
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class _MarketData:
        def __init__(self) -> None:
            self.zombie_checks = 0
            self.restart_count = 0

        def _required_live_symbols(self) -> set[str]:
            return {"NFO:NIFTY2671423950CE"}

        def time_since_last_live_ws_tick(self, symbol: str):
            return 1.0

        def _check_zombie_ticks(self) -> None:
            self.zombie_checks += 1

        def _trigger_zombie_ws_restart(self) -> None:
            self.restart_count += 1

    old_symbol = "NFO:NIFTY2671424100PE"
    selected_symbol = "NFO:NIFTY2671423950CE"
    market_data = _MarketData()
    now_mono = time.monotonic()
    now_wall = time.time()
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.ready = False
    runner._last_tick_seen_ts = now_mono
    runner._last_global_eval_ts = now_mono
    runner._eval_stall_recovery_attempted = False
    runner._candle_engines = {old_symbol: object(), selected_symbol: object()}
    runner._active_symbols = {old_symbol, selected_symbol}
    runner._tracked_symbols = {old_symbol, selected_symbol}
    runner._last_tick_time_by_symbol = {
        old_symbol: now_wall - 4000.0,
        selected_symbol: now_wall,
    }
    runner._active_option_symbols = {selected_symbol}
    runner._active_selected_ce = selected_symbol
    runner._active_selected_pe = None
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._active_contract_basket = None
    runner._data_hub = None
    runner._last_ws_stale_log_ts_by_symbol = {}
    runner._last_ws_reconnect_attempt_ts = 0.0
    runner._market_data = market_data
    runner._log_throttle_state = {}
    runner._logger = logging.getLogger("test.runner.watchdog")

    monkeypatch.setattr(runner_module, "is_market_open_now", lambda: True)

    runner._health_watchdog()

    assert market_data.zombie_checks == 0
    assert market_data.restart_count == 0


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
    threshold_open = market_hours.stale_threshold_for_symbol(
        "NSE:NIFTY", market_open=True
    )
    assert threshold_open == 120.0
