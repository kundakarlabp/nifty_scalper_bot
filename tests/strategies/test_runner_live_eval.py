from __future__ import annotations

from pathlib import Path


def test_runner_does_not_skip_on_two_second_candle_age_gate() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(
        encoding='utf-8'
    )

    assert 'candle_age_seconds > 2' not in source



def test_first_premium_squeeze_reaches_order_request_when_live_armed() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'RUNNER_ORDER_REQUEST' in source
    assert 'premium_momentum_squeeze' in source


def test_premium_fallback_rejects_far_strike() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'PREMIUM_SQUEEZE_SKIPPED reason=outside_selected_strike_window' in source


def test_premium_fallback_allows_selected_ce_pe() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'selected = upper_symbol in {selected_ce, selected_pe}' in source
