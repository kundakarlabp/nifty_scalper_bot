from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_spot_ready_uses_fresh_tick_not_bars() -> None:
    mdm = MarketDataManager()
    mdm.register_symbol('NSE:NIFTY', 256265)
    mdm._latest_ticks['NSE:NIFTY'] = {'ltp': 22500.0, 'source': 'ws'}
    mdm._last_tick_monotonic['NSE:NIFTY'] = mdm._clock()
    mdm.set_readiness_requirements(
        spot_symbol='NSE:NIFTY',
        futures_symbol='NFO:NIFTY26MAYFUT',
        atm_ce_symbol='NFO:NIFTY26MAY22500CE',
        atm_pe_symbol='NFO:NIFTY26MAY22500PE',
        option_symbols=[],
    )
    state = mdm.readiness_state_snapshot()
    assert state['spot_ready'] is True
