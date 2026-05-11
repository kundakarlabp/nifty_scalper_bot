from nifty_scalper_bot.core.app import build_active_trading_basket_symbols


def test_active_basket_includes_selected_options_first() -> None:
    basket = {
        'spot_symbol': 'NSE:NIFTY',
        'futures_symbol': 'NFO:NIFTY26MAYFUT',
        'selected_ce': 'NFO:NIFTY26MAY23900CE',
        'selected_pe': 'NFO:NIFTY26MAY23900PE',
        'option_symbols': [
            'NFO:NIFTY26MAY23800CE',
            'NFO:NIFTY26MAY23800PE',
            'NFO:NIFTY26MAY23900CE',
            'NFO:NIFTY26MAY23900PE',
        ],
    }
    symbols = build_active_trading_basket_symbols(None, basket)
    assert symbols[:4] == [
        'NSE:NIFTY',
        'NFO:NIFTY26MAYFUT',
        'NFO:NIFTY26MAY23900CE',
        'NFO:NIFTY26MAY23900PE',
    ]
    assert 'NFO:NIFTY26MAY23800CE' in symbols
