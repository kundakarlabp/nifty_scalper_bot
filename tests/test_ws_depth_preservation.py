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


def test_one_sided_depth_does_not_claim_full_book() -> None:
    mdm = MarketDataManager()
    raw = {
        "symbol": "NFO:NIFTY26MAY24000CE",
        "ltp": 120.0,
        "bid": 119.5,
        "ask": 120.5,
        "depth": {
            "buy": [{"price": 119.5, "quantity": 65}],
            "sell": [],
        },
    }
    normalized = mdm.normalize_live_tick(raw, source="ws")
    extracted = mdm._extract_depth_quote_fields(raw)

    assert normalized is not None
    assert normalized["depth_available"] is False
    assert normalized["bid_ask_source"] == "quote"
    assert extracted["depth_available"] is False
    assert extracted["bid_ask_source"] == "quote"
    assert mdm._is_full_websocket_quote(normalized) is False


def test_two_sided_depth_retains_full_websocket_source() -> None:
    mdm = MarketDataManager()
    raw = {
        "symbol": "NFO:NIFTY26MAY24000CE",
        "ltp": 120.0,
        "depth": {
            "buy": [{"price": 119.5, "quantity": 65}],
            "sell": [{"price": 120.5, "quantity": 65}],
        },
    }
    normalized = mdm.normalize_live_tick(raw, source="ws")
    assert normalized is not None
    assert normalized["depth_available"] is True
    assert mdm._is_full_websocket_quote(normalized) is True
