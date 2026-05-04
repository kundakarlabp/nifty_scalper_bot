from pathlib import Path


def test_app_initializes_bracket_manager_with_supported_keyword() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'market_data=' in text
    assert 'market_data_manager=' not in text[text.find('BracketManager('): text.find('BracketManager(') + 400]
