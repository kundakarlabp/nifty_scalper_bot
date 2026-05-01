from pathlib import Path


def test_market_data_manager_does_not_reference_missing_tick_instrument_token() -> None:
    src = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text()
    assert 'tick.instrument_token' not in src
