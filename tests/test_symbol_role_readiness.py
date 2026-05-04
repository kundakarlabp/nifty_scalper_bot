from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_symbol_role_helpers() -> None:
    runner = StrategyRunner()
    assert runner._is_context_symbol('NSE:NIFTY')
    assert runner._is_context_symbol('NFO:NIFTY26MAYFUT')
    assert runner._is_tradable_symbol('NFO:NIFTY26MAY24000CE')
    assert runner._is_tradable_symbol('NFO:NIFTY26MAY24000PE')
    assert not runner._is_tradable_symbol('NSE:NIFTY')
