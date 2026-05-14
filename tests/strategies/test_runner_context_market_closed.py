from nifty_scalper_bot.strategies.runner import StrategyRunner

def test_runner_phase9_market_closed_context_skip():
    r = StrategyRunner.__new__(StrategyRunner)
    assert r._symbol_role_for_runner('NSE:NIFTY') == 'spot_context'
    assert r._symbol_role_for_runner('NFO:NIFTY26MAYFUT') == 'futures_context'
