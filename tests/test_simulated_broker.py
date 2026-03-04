from nifty_scalper_bot.testing.simulated_broker import SimulatedZerodhaBroker


def test_simulated_market_fill_updates_position() -> None:
    broker = SimulatedZerodhaBroker()
    broker.set_price('NIFTY', 201.5)
    order = broker.place_order(
        symbol='NIFTY',
        side='BUY',
        quantity=2,
        order_type='MARKET',
    )
    assert 'order_id' in order
    positions = broker.get_positions()
    assert positions[0]['symbol'] == 'NIFTY'
    assert positions[0]['qty'] == 2
