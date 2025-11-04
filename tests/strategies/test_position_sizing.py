from nifty_scalper_bot.strategies.position_sizing import calculate_position_size


def test_calculate_position_size_uses_live_delta() -> None:
    market_data = {
        "greeks": {
            "NIFTY25OCT20000CE": {"delta": 0.35},
        }
    }
    quantity = calculate_position_size(
        account_balance=100_000.0,
        risk_pct=1.0,
        or_high=22050.0,
        or_low=22000.0,
        option_price=120.0,
        lot_size=50,
        tradingsymbol="NIFTY25OCT20000CE",
        market_data=market_data,
    )
    # Range is 50; delta 0.35 => stop distance 17.5; per lot risk 875.
    # Risk budget 1000 -> 1 lot (50 quantity).
    assert quantity == 50


def test_calculate_position_size_handles_missing_greeks() -> None:
    market_data = {"greeks": {}}
    quantity = calculate_position_size(
        account_balance=50_000.0,
        risk_pct=1.0,
        or_high=22010.0,
        or_low=21990.0,
        option_price=110.0,
        lot_size=50,
        tradingsymbol="NIFTY25OCT20000CE",
        market_data=market_data,
    )
    assert quantity == 0
