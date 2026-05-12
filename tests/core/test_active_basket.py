from nifty_scalper_bot.core.active_basket import pick_atm_option_symbols_from_basket


def test_selected_symbols_preferred_over_atm_symbols() -> None:
    ce, pe = pick_atm_option_symbols_from_basket({'selected_ce': 'NFO:SELECTEDCE', 'atm_ce': 'NFO:ATMCE', 'selected_pe': 'NFO:SELECTEDPE', 'atm_pe': 'NFO:ATMPE'})
    assert ce == 'NFO:SELECTEDCE'
    assert pe == 'NFO:SELECTEDPE'
