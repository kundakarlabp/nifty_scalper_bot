from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_regime_inputs_storage() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._last_regime_inputs_by_symbol = {}
    runner._last_regime_inputs_by_symbol['NFO:NIFTY26MAY23500CE'] = {'atr': 1.0, 'avg_volume': 10}
    assert 'atr' in runner._last_regime_inputs_by_symbol['NFO:NIFTY26MAY23500CE']
