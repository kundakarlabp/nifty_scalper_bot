from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager


BUY_SYMBOL = "NFO:NIFTYEXECBUYCE"
SELL_SYMBOL = "NFO:NIFTYEXECSELLPE"


def _manager() -> BracketManager:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    return manager


def _register(
    manager: BracketManager,
    *,
    order_id: str,
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    tp1: float | None = None,
):
    manager.register_virtual_bracket(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=130 if tp1 is not None else 65,
        price=entry,
        sl=sl,
        tp=tp,
        tp1_price=tp1,
        tp1_qty=65 if tp1 is not None else None,
        resolved_lot_size=65,
        activate_immediately=True,
    )
    bracket = manager.get_bracket(order_id)
    assert bracket is not None
    return bracket


def _fresh_quote(manager: BracketManager, symbol: str, *, bid: float, ask: float) -> None:
    manager._exit_quotes[symbol] = (bid, ask, time.time())


def test_long_final_tp_does_not_trigger_when_ltp_crosses_but_bid_does_not() -> None:
    manager = _manager()
    bracket = _register(
        manager,
        order_id="buy-final-blocked",
        symbol=BUY_SYMBOL,
        side="BUY",
        entry=100.0,
        sl=90.0,
        tp=120.0,
    )
    _fresh_quote(manager, BUY_SYMBOL, bid=119.0, ask=121.5)

    action = manager._evaluate_exit_fast(bracket, 121.0, committed_sl=90.0)

    assert action is None


def test_long_final_tp_triggers_when_executable_bid_reaches_target() -> None:
    manager = _manager()
    bracket = _register(
        manager,
        order_id="buy-final-hit",
        symbol=BUY_SYMBOL,
        side="BUY",
        entry=100.0,
        sl=90.0,
        tp=120.0,
    )
    _fresh_quote(manager, BUY_SYMBOL, bid=120.05, ask=121.0)

    action = manager._evaluate_exit_fast(bracket, 120.5, committed_sl=90.0)

    assert action is not None
    assert action["type"] == "FINAL_TP"
    assert action["trigger_price_source"] == "bid"


def test_long_tp1_uses_executable_bid_not_ltp() -> None:
    manager = _manager()
    bracket = _register(
        manager,
        order_id="buy-tp1",
        symbol=BUY_SYMBOL,
        side="BUY",
        entry=100.0,
        sl=90.0,
        tp=130.0,
        tp1=110.0,
    )
    _fresh_quote(manager, BUY_SYMBOL, bid=109.5, ask=111.5)
    assert manager._evaluate_exit_fast(bracket, 111.0, committed_sl=90.0) is None

    _fresh_quote(manager, BUY_SYMBOL, bid=110.05, ask=111.5)
    action = manager._evaluate_exit_fast(bracket, 111.0, committed_sl=90.0)

    assert action is not None
    assert action["type"] == "PARTIAL_TP"
    assert action["trigger_price_source"] == "bid"


@pytest.mark.parametrize(
    ("ask", "expected"),
    [
        (80.5, None),
        (79.95, "FINAL_TP"),
    ],
)
def test_short_final_tp_uses_executable_ask(ask: float, expected: str | None) -> None:
    manager = _manager()
    bracket = _register(
        manager,
        order_id=f"sell-final-{ask}",
        symbol=SELL_SYMBOL,
        side="SELL",
        entry=100.0,
        sl=110.0,
        tp=80.0,
    )
    _fresh_quote(manager, SELL_SYMBOL, bid=79.0, ask=ask)

    action = manager._evaluate_exit_fast(bracket, 79.5, committed_sl=110.0)

    if expected is None:
        assert action is None
    else:
        assert action is not None
        assert action["type"] == expected
        assert action["trigger_price_source"] == "ask"


def test_stale_quote_preserves_ltp_fallback_for_tp() -> None:
    manager = _manager()
    bracket = _register(
        manager,
        order_id="buy-stale-fallback",
        symbol=BUY_SYMBOL,
        side="BUY",
        entry=100.0,
        sl=90.0,
        tp=120.0,
    )
    manager._exit_quotes[BUY_SYMBOL] = (
        115.0,
        116.0,
        time.time() - manager._exit_quote_max_age - 1.0,
    )

    action = manager._evaluate_exit_fast(bracket, 121.0, committed_sl=90.0)

    assert action is not None
    assert action["type"] == "FINAL_TP"
    assert action["trigger_price_source"] == "ltp_stale_quote"
