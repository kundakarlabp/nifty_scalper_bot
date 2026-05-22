from nifty_scalper_bot.strategies.runner import StrategyRunner

def test_runner_phase9_market_closed_context_skip():
    r = StrategyRunner.__new__(StrategyRunner)
    assert r._symbol_role_for_runner('NSE:NIFTY') == 'spot_context'
    assert r._symbol_role_for_runner('NFO:NIFTY26MAYFUT') == 'futures_context'


def test_runner_market_closed_compact_log_present() -> None:
    source = open('src/nifty_scalper_bot/strategies/runner.py', encoding='utf-8').read()
    assert 'MARKET_SESSION_CLOSED orders_disabled=True reason=normal_close' in source
