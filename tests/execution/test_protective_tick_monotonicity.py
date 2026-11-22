"""Out-of-order option events must not reverse the protection-time watermark."""

from __future__ import annotations

import time
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager

SYMBOL = "NFO:NIFTY26NOV25700CE"


class _MarketData:
    def get_latest_tick(self, _symbol):
        return {}


def _manager():
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    manager = BracketManager(order_manager=order_manager, market_data=_MarketData())
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=110.0,
        sl=100.0,
        tp=130.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    bracket.entry_fill_ts = time.time() - 100.0
    bracket.last_ltp = 110.0
    manager._fire_exits_batch = Mock()
    return manager, bracket


def _full_tick(ltp, bid, at):
    return {
        "symbol": SYMBOL,
        "ltp": ltp,
        "bid": bid,
        "ask": bid + 0.2,
        "exchange_timestamp": at,
    }


def test_older_protection_event_does_not_trigger_spurious_stop() -> None:
    manager, bracket = _manager()
    now = time.time() - 1.0

    manager.on_tick(
        SYMBOL,
        110.0,
        exchange_ts=now,
        tick=_full_tick(110.0, 109.8, now),
        defer_submission=True,
    )
    manager.on_tick(
        SYMBOL,
        90.0,
        exchange_ts=now - 1.0,
        tick=_full_tick(90.0, 89.8, now - 1.0),
        defer_submission=True,
    )

    assert bracket.last_ltp == 110.0
    assert manager._executable_exit_price(bracket, 110.0) == (109.8, "bid")
    manager._fire_exits_batch.assert_not_called()


def test_same_second_changed_option_price_still_triggers_protection() -> None:
    manager, bracket = _manager()
    now = time.time() - 1.0

    manager.on_tick(
        SYMBOL,
        110.0,
        exchange_ts=now,
        tick=_full_tick(110.0, 109.8, now),
        defer_submission=True,
    )
    manager.on_tick(
        SYMBOL,
        90.0,
        exchange_ts=now,
        tick=_full_tick(90.0, 89.8, now),
        defer_submission=True,
    )

    assert bracket.last_ltp == 90.0
    manager._fire_exits_batch.assert_called_once()
    action = manager._fire_exits_batch.call_args.args[0][0][1]
    assert action["type"] == "SL"


def test_depthless_untimestamped_protection_event_clears_prior_bid() -> None:
    manager, bracket = _manager()
    now = time.time() - 1.0
    manager.on_tick(
        SYMBOL,
        110.0,
        exchange_ts=now,
        tick=_full_tick(110.0, 109.8, now),
        defer_submission=True,
    )
    manager.on_tick(
        SYMBOL,
        110.0,
        exchange_ts=None,
        defer_submission=True,
        tick={"symbol": SYMBOL, "ltp": 110.0},
    )

    assert manager._executable_exit_price(bracket, 110.0)[1] != "bid"
