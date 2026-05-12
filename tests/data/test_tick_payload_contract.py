from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_full_tick_preserves_depth_bid_ask_tradable_quote() -> None:
    mdm = MarketDataManager()
    mdm._symbol_by_token[123] = 'NFO:NIFTYTESTCE'
    raw = {
        'instrument_token': 123,
        'last_price': 100.0,
        'depth': {'buy': [{'price': 99.95, 'quantity': 100}], 'sell': [{'price': 100.05, 'quantity': 150}]},
        'volume_traded_today': 1000,
        'oi': 5000,
        'exchange_timestamp': datetime.now(timezone.utc),
    }
    tick = mdm._normalize_ws_tick(raw)
    assert tick is not None
    assert tick['bid'] == pytest.approx(99.95)
    assert tick['ask'] == pytest.approx(100.05)
    assert tick['best_bid'] == pytest.approx(99.95)
    assert tick['best_ask'] == pytest.approx(100.05)
    assert tick['spread'] == pytest.approx(0.10)
    assert tick['mid'] == pytest.approx(100.00)
    assert tick['depth_available'] is True
    assert tick['tradable_quote'] is True
    assert tick['depth']['buy'][0]['price'] == pytest.approx(99.95)
    assert tick['source'] == 'ws_full' or tick['bid_ask_source'] in {'ws_full', 'market_depth'}
