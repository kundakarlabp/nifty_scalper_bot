from __future__ import annotations

import time
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager


SYMBOL = "NFO:NIFTYTRAILCE"


def test_ltp_only_spike_does_not_ratchet_buy_stop_without_executable_profit() -> None:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.place_order.return_value = "unused"
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="trail-entry",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=70.80,
        sl=66.60,
        tp=78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("trail-entry")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = 0.20

    # Last trade spikes enough to activate the old LTP-based trail, but a long
    # can only be liquidated at the bid, which has barely moved from entry.
    manager._exit_quotes[SYMBOL] = (70.90, 72.70, time.time())
    manager.on_tick(SYMBOL, 72.60, exchange_ts=1_000.0)

    assert bracket.highest_ltp == 72.60  # informational LTP analytics stay intact
    assert bracket.sl_trigger_price == 66.60
    assert bracket.trail_revision == 0

    # Once executable bid itself reaches the same profitable region, trailing
    # is allowed to activate and ratchet protection normally.
    manager._exit_quotes[SYMBOL] = (72.60, 72.70, time.time())
    manager.on_tick(SYMBOL, 72.60, exchange_ts=1_001.0)

    assert bracket.sl_trigger_price > 66.60
    assert bracket.trail_revision == 1
