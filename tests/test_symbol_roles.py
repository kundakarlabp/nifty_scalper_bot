from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_symbol_roles() -> None:
    assert StrategyRunner._is_context_symbol(None, 'NSE:NIFTY')
    assert not StrategyRunner._is_tradable_symbol(None, 'NSE:NIFTY')

    assert StrategyRunner._is_context_symbol(None, 'NFO:NIFTY26MAYFUT')
    assert not StrategyRunner._is_tradable_symbol(None, 'NFO:NIFTY26MAYFUT')

    assert StrategyRunner._is_tradable_symbol(None, 'NFO:NIFTY26MAY24000CE')
    assert StrategyRunner._is_tradable_symbol(None, 'NFO:NIFTY26MAY24000PE')
