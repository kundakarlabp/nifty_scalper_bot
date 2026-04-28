from __future__ import annotations

import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_proof_rejects_quote_source() -> None:
    mdm = MarketDataManager(broker=None, websocket=None)
    symbol = 'NSE:NIFTY'
    tick = {
        'last_price': 24011.6,
        'source': 'quote',
        'tradable_quote': True,
        'bid': 24011.0,
        'ask': 24012.0,
    }
    mdm._latest_ticks[symbol] = tick  # noqa: SLF001
    mdm._last_tick_source[symbol] = 'quote'  # noqa: SLF001
    now = time.time()
    mdm._last_tick_time[symbol] = now  # noqa: SLF001
    assert mdm.get_fresh_spot_tick(symbol, require_ws=True) is None
    assert mdm.has_ws_tradable_quote([symbol]) is False

    tick['source'] = 'ws_full'
    mdm._last_tick_source[symbol] = 'ws_full'  # noqa: SLF001
    assert mdm.get_fresh_spot_tick(symbol, require_ws=True) is not None
    assert mdm.has_ws_tradable_quote([symbol]) is True
