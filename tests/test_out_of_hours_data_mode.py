from pathlib import Path


def test_out_of_hours_mode_log_present() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'OUT_OF_HOURS_DATA_MODE' in text or 'BASKET_BUILD_DEFERRED' in text
