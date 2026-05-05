from __future__ import annotations

from pathlib import Path


def test_token_integrity_mandatory_symbols() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'LIVE_TOKEN_INTEGRITY_BLOCKED' in text
    assert 'mandatory_symbols' in text
    assert 'mandatory_tokens_missing' in text

