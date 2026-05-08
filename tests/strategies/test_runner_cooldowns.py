from __future__ import annotations

from pathlib import Path


def test_on_tick_error_phase_updates_for_premium_squeeze() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'phase = "phase8_premium_squeeze"' in source


def test_active_option_selection_uses_atm_strike_key() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'atm=basket.get("atm_strike")' in source
