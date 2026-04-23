from __future__ import annotations

from pathlib import Path


def test_runner_does_not_skip_on_two_second_candle_age_gate() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(
        encoding='utf-8'
    )

    assert 'candle_age_seconds > 2' not in source
