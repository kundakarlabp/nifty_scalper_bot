from pathlib import Path


def test_premium_squeeze_uses_normalized_selected_symbols_and_throttled_skip() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'normalized_selected_ce' in source
    assert 'normalized_selected_pe' in source
    assert 'premium_outside_window:' in source
    assert 'outside_selected_strike_window' in source
