"""Boundary validation for canonical bracket market and ATR inputs."""

from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager

SYMBOL = "NFO:NIFTY26OCT22700CE"


def _manager(monkeypatch, tmp_path) -> BracketManager:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "fills.db"))
    manager = BracketManager(order_manager=Mock())
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        "entry-finite",
        SYMBOL,
        "BUY",
        65,
        100.0,
        95.0,
        120.0,
        activate_immediately=True,
    )
    return manager


def test_nonfinite_tick_never_contaminates_bracket_or_exits(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    original = (
        bracket.last_ltp,
        bracket.highest_ltp,
        bracket.sl_trigger_price,
        bracket.trail_revision,
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
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


def test_nonfinite_quote_cannot_be_cached_as_executable_depth(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    for bid, ask in (
        (float("nan"), 102.0),
        (99.0, float("nan")),
        (float("inf"), 102.0),
        (99.0, float("inf")),
    ):
        assert not manager._capture_exit_quote(
            SYMBOL, {"bid": bid, "ask": ask, "ltp": 100.0}
        )
        assert SYMBOL not in manager._exit_quotes


def test_corrupt_cached_depth_falls_back_to_ltp(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    manager._exit_quotes[SYMBOL] = (float("nan"), float("inf"), 1.0)
    assert manager._executable_exit_price(bracket, 99.0) == (
        99.0,
        "ltp_stale_quote",
    )


def test_nonfinite_atr_is_rejected_from_provider_and_manual_feed(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    manager._atr_provider = Mock()
    manager._current_atr[SYMBOL] = 2.5
    for bad in (float("nan"), float("inf"), float("-inf")):
        manager._atr_provider.get_current_atr.return_value = bad
        assert manager._get_current_atr(SYMBOL) == 2.5
        manager.update_market_stats(SYMBOL, atr=bad)
        assert manager._current_atr[SYMBOL] == 2.5
    manager._atr_provider.feed_manual.assert_not_called()


def test_invalid_activation_r_uses_existing_default(monkeypatch, tmp_path):
    manager = _manager(monkeypatch, tmp_path)
    bracket = manager.get_bracket("entry-finite")
    assert bracket is not None
    for value in ("nan", "inf", "-inf", "0", "-1"):
        bracket.trailing_config["breakeven_activation_r"] = value
        assert manager._trail_activation_r(bracket) == 0.60
