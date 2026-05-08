from __future__ import annotations

from pathlib import Path


def test_on_tick_error_phase_updates_for_premium_squeeze() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'phase = "phase8_premium_squeeze"' in source


def test_active_option_selection_uses_atm_strike_key() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'atm=basket.get("atm_strike")' in source


def test_cooldown_first_trade_no_epoch_age() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'first_trade_for_key' in source
    assert 'age_seconds=None' not in source or 'age_seconds=%s' in source


def test_rejected_signal_cooldown_prevents_spam() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'SIGNAL_REJECT_COOLDOWN_ACTIVE' in source
    assert '_signal_reject_cooldown_ts' in source


def test_candidate_enrichment_raises_rr_and_option_score() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'metadata["rr_score"] = max(' in source
    assert 'metadata["option_score"] = max(' in source
