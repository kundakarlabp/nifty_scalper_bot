from pathlib import Path


def test_live_candle_support_exists() -> None:
    text = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text(encoding='utf-8')
    assert '_ingest_normalized_tick' in text
