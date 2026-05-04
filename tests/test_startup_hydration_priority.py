from nifty_scalper_bot.core.app import prioritize_startup_hydration_symbols


def test_prioritize_startup_hydration_symbols() -> None:
    symbols = [
        'NSE:NIFTY',
        'NFO:NIFTY26MAYFUT',
        'NFO:NIFTY26MAY24000CE',
        'NFO:NIFTY26MAY24000PE',
        'NFO:NIFTY26MAY24100CE',
    ]
    out = prioritize_startup_hydration_symbols(
        symbols,
        atm_ce='NFO:NIFTY26MAY24000CE',
        atm_pe='NFO:NIFTY26MAY24000PE',
        futures_symbol='NFO:NIFTY26MAYFUT',
    )
    assert out == [
        'NSE:NIFTY',
        'NFO:NIFTY26MAYFUT',
        'NFO:NIFTY26MAY24000CE',
        'NFO:NIFTY26MAY24000PE',
    ]
