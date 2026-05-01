from __future__ import annotations

import logging
import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_no_direct_subscriber_warning_after_grace(caplog) -> None:
    mdm = MarketDataManager(rest_client=None)
    symbol = 'NFO:TEST'
    mdm._active_subscribed_symbols.add(symbol)

    caplog.set_level(logging.WARNING)
    mdm._emit_tick(symbol, {'symbol': symbol, 'ltp': 100, 'instrument_token': 1}, source='ws')
    assert not any('MDM_TICK_NO_DIRECT_SUBSCRIBER' in r.message for r in caplog.records)

    mdm._first_seen_at_by_symbol[symbol] = time.monotonic() - 31
    mdm._emit_tick(symbol, {'symbol': symbol, 'ltp': 101, 'instrument_token': 1}, source='ws')
    assert any('MDM_TICK_NO_DIRECT_SUBSCRIBER' in r.message for r in caplog.records)
