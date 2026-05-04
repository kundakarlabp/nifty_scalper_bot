from __future__ import annotations

from pathlib import Path


def test_bar_subscriber_diagnostics_include_callback_name() -> None:
    text = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text(encoding='utf-8')
    assert 'callback_name = getattr(cb, "__qualname__", repr(cb))' in text
    assert 'BAR_SUBSCRIBER_FAILED symbol=%s callback=%s error=%s' in text
    assert '"callback": callback_name' in text
