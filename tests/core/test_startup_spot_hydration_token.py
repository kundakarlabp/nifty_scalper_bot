from pathlib import Path


def test_startup_spot_hydration_has_well_known_nifty_token_fallback() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'well_known_spot_tokens' in source
    assert '"NSE:NIFTY": 256265' in source
    assert '"NIFTY": 256265' in source
    assert 'HYDRATION_SYMBOLS_FINAL' in source
