from __future__ import annotations

from pathlib import Path


def test_on_tick_error_phase_updates_for_premium_squeeze() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'phase = "phase8_premium_squeeze"' in source
    assert 'phase = "phase8_vwap_crossover"' in source
    assert 'phase = "phase9_strategy_manager"' in source
    assert 'phase = "phase10_signal_execution"' in source
    assert 'RUNNER_ON_TICK_ERROR symbol=%s phase=%s error_type=%s error=%s' in source
    assert 'exc_info=True' in source


def test_cooldown_rejection_diagnostics_logged() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'COOLDOWN_REJECTED reason=reason_cooldown' in source
    assert 'COOLDOWN_REJECTED reason=underlying_cooldown' in source
