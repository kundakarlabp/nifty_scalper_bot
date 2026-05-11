from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_normalize_live_tick_uses_market_depth_bid_ask() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick(
        {
            'symbol': 'NFO:NIFTY26MAY24000CE',
            'ltp': 120.0,
            'depth': {
                'buy': [{'price': 119.5, 'quantity': 100}],
                'sell': [{'price': 120.5, 'quantity': 120}],
            },
        },
        source='ws',
    )
    assert tick is not None
    assert tick['bid'] == 119.5
    assert tick['ask'] == 120.5
    assert tick['depth_available'] is True
    assert tick['bid_ask_source'] == 'market_depth'
    assert tick['tradable_quote'] is True


def test_normalize_live_tick_ltp_only_not_tradable() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick({'symbol': 'NFO:NIFTY26MAY24000CE', 'ltp': 120.0}, source='ws')
    assert tick is not None
    assert tick['bid'] is None
    assert tick['ask'] is None
    assert tick['tradable_quote'] is False
