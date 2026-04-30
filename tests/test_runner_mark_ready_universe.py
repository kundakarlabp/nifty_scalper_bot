from __future__ import annotations

from pathlib import Path


def test_app_uses_full_readiness_universe_for_mark_ready() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text()
    assert 'runner.mark_ready(readiness_symbols)' in source
    assert 'runner.mark_ready(ready_symbols)' not in source
