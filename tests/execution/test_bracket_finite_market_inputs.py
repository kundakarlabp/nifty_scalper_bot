"""Boundary validation for canonical bracket market and ATR inputs."""

from __future__ import annotations

import math
import unittest.mock

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager


SYMBOL = "NFO:NIFTY26OCT22700CE"


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "fill-ledger.db"))
    owner = BracketManager(order_manager=unittest.mock.Mock())
    owner._running = False
    owner._watchdog_thread.join(timeout=1.0)
    owner.register_virtual_bracket(
        "entry-finite",
        SYMBOL,
        "BUY",
        65,
        100.0,
        95.0,
        120.0,
        activate_immediately=True,
    )
    yield owner
    owner._running = False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_tick_never_contaminates_bracket_or_exits(manager, bad):
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    original = (
        bracket.last_ltp,
        bracket.highest_ltp,
        bracket.sl_trigger_price,
        bracket.trail_revision,
    )
    manager.on_tick(SYMBOL, bad)
    manager.on_tick_event({"symbol": SYMBOL, "ltp": bad, "bid": 1.0, "ask": 2.0})
    assert (
        bracket.last_ltp,
        bracket.highest_ltp,
        bracket.sl_trigger_price,
        bracket.trail_revision,
    ) == original
    assert SYMBOL not in manager._exit_quotes
    assert not bracket.exit_pending


@pytest.mark.parametrize(
    ("bid", "ask"),
    [
        (math.nan, 102.0),
        (99.0, math.nan),
        (math.inf, 102.0),
        (99.0, math.inf),
    ],
)
def test_nonfinite_quote_is_not_considered_fresh_executable_depth(
    manager, bid, ask
):
    assert not manager._capture_exit_quote(
        SYMBOL, {"bid": bid, "ask": ask, "ltp": 100.0}
    )
    assert SYMBOL not in manager._exit_quotes


def test_corrupt_cached_depth_falls_back_to_ltp(manager):
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    manager._exit_quotes[SYMBOL] = (math.nan, math.inf, 1.0)
    assert manager._executable_exit_price(bracket, 99.0) == (
        99.0,
        "ltp_stale_quote",
    )


@pytest.mark.parametrize("bad", [math.inf, math.nan, -math.inf])
def test_nonfinite_atr_provider_and_external_updates_fail_closed(manager, bad):
    manager._atr_provider = Mock()
    manager._atr_provider.get_current_atr.return_value = bad
    manager._current_atr[SYMBOL] = 2.5
    assert manager._get_current_atr(SYMBOL) == 2.5
    manager.update_market_stats(SYMBOL, atr=bad)
    assert manager._current_atr[SYMBOL] == 2.5
    manager._atr_provider.feed_manual.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_activation_r_uses_safe_default(manager, value):
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = value
    assert manager._trail_activation_r(bracket) == pytest.approx(0.60)
