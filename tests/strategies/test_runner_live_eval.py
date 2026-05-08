from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_set_runtime_readiness_propagates_selected_option_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'selected_ce: str | None = None' in source
    assert 'self.set_active_option_context(' in source


def test_set_active_option_context_normalizes_and_stores_values() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'RUNNER_ACTIVE_OPTION_CONTEXT' in source
    assert 'normalize_symbol(str(selected_ce))' in source


def test_extract_strike_from_symbol_parses_options() -> None:
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24250CE') == 24250
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAY24250PE') == 24250
    assert StrategyRunner._extract_strike_from_symbol('NFO:NIFTY26MAYFUT') is None
    assert StrategyRunner._extract_strike_from_symbol('NSE:NIFTY') is None


def test_premium_squeeze_uses_active_atm_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'reason=outside_selected_strike_window symbol=%s selected_ce=%s selected_pe=%s atm_strike=%s' in source


def test_premium_squeeze_missing_context_logs_missing_context() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'reason=missing_active_option_context' in source


def test_signal_evaluation_failure_logs_error_type() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'SIGNAL_EVALUATION_FAILURE symbol=%s phase=%s error_type=%s error=%s trace_id=%s' in source
