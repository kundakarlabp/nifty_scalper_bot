from pathlib import Path

from nifty_scalper_bot.core.app import nearest_available_strike


def test_no_hardcoded_24000_in_app() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert '24000.0' not in text
    assert 'NIFTY_FALLBACK_LTP' not in text


def test_nearest_strike_basic() -> None:
    assert nearest_available_strike(23972, [23900, 23950, 24000]) == 23950


def test_nearest_strike_precise() -> None:
    assert nearest_available_strike(23997.55, [23900, 23950, 24000, 24050]) == 24000


def test_stale_spot_blocks_message_present() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'ATM_SELECTION_BLOCKED reason=spot_unavailable_or_stale' in text
