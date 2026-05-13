from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_normalise_tick_volume_delta_from_cumulative_sequence() -> None:
    manager = MarketDataManager(broker=None, websocket=None)
    symbol = 'NFO:NIFTY26MAYFUT'

    tick1 = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 1000})
    tick2 = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 1040})
    tick3 = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 1090})

    assert tick1['volume'] == 0.0
    assert tick2['volume'] == 40.0
    assert tick3['volume'] == 50.0
    assert tick3['volume_cumulative'] == 1090.0


def test_normalise_tick_volume_delta_uses_ltq_on_first_tick() -> None:
    manager = MarketDataManager(broker=None, websocket=None)
    symbol = 'NFO:NIFTY26MAYFUT'

    first = manager._normalise_tick_volume_delta(
        symbol,
        {'symbol': symbol, 'volume_traded_today': 1000, 'last_traded_quantity': 12},
    )
    second = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 1040})

    assert first['volume'] == 12.0
    assert second['volume'] == 40.0



def test_normalise_tick_volume_delta_clamps_option_spike(monkeypatch) -> None:
    monkeypatch.setenv('OPTION_MAX_REASONABLE_TICK_VOLUME_DELTA', '1000000')
    manager = MarketDataManager(broker=None, websocket=None)
    symbol = 'NFO:NIFTY26MAY23500CE'

    _ = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 1000})
    spiked = manager._normalise_tick_volume_delta(symbol, {'symbol': symbol, 'volume_traded_today': 2005000})

    assert spiked['volume_cumulative'] == 2005000.0
    assert spiked['volume'] == 0.0
    assert spiked['volume_delta'] == 0.0
