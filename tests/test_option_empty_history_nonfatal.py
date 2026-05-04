from pathlib import Path


def test_option_empty_history_log_present() -> None:
    text = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'OPTION_HISTORY_COLD' in text
