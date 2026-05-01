from nifty_scalper_bot.strategies.directional_bias import DirectionalBiasEngine


def test_up_bias():
    d = DirectionalBiasEngine().evaluate({'price': 102, 'spot_vwap': 100, 'ema9': 101, 'ema21': 100, 'rsi': 60, 'macd_hist': 1.0, 'adx': 20})
    assert d.side == 'CE'


def test_down_bias():
    d = DirectionalBiasEngine().evaluate({'price': 98, 'spot_vwap': 100, 'ema9': 99, 'ema21': 100, 'rsi': 40, 'macd_hist': -1.0, 'adx': 20})
    assert d.side == 'PE'


def test_mixed_none():
    d = DirectionalBiasEngine().evaluate({'price': 100, 'spot_vwap': 100, 'ema9': 100, 'ema21': 100, 'rsi': 50, 'macd_hist': 0.0, 'adx': 10})
    assert d.side == 'NONE'
