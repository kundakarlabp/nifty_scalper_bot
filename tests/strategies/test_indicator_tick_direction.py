from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def _seed(engine, closes):
    for idx, close in enumerate(closes):
        engine.update_price('NFO:NIFTY26MAY23750CE', close, volume=100, timestamp=idx + 1, open_price=close, high=close, low=close)


def test_tick_direction_derived_from_history():
    engine = IndicatorEngine()
    _seed(engine, [100, 101])
    vals = engine.get_indicators('NFO:NIFTY26MAY23750CE', names={'tick_direction'})
    assert vals['tick_direction'] == 'UP'

    engine = IndicatorEngine(); _seed(engine, [101, 100])
    vals = engine.get_indicators('NFO:NIFTY26MAY23750CE', names={'tick_direction'})
    assert vals['tick_direction'] == 'DOWN'

    engine = IndicatorEngine(); _seed(engine, [100, 100])
    vals = engine.get_indicators('NFO:NIFTY26MAY23750CE', names={'tick_direction'})
    assert vals['tick_direction'] == 'FLAT'


def test_runtime_tick_direction_preserved():
    engine = IndicatorEngine()
    _seed(engine, [100, 101])
    engine.set_runtime_context('NFO:NIFTY26MAY23750CE', {'tick_direction': 'BUY'})
    vals = engine.get_indicators('NFO:NIFTY26MAY23750CE', names={'tick_direction'})
    assert vals['tick_direction'] == 'BUY'
    assert vals['tick_direction_source'] == 'runtime_context'
