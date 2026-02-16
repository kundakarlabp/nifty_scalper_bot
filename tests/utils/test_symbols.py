from nifty_scalper_bot.utils.symbols import is_strategy_instrument


def test_is_strategy_instrument_matches_nifty_derivatives() -> None:
    assert is_strategy_instrument('NFO:NIFTY24APR22500CE')
    assert is_strategy_instrument('NFO:NIFTY FUT')
    assert not is_strategy_instrument('NSE:RELIANCE')
