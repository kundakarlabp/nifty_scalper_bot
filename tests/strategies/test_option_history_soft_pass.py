from pathlib import Path


def test_option_history_soft_pass_present() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'OPTION_STRICT_HISTORY_FOR_ALL_STRATEGIES", False' in source
    assert 'OPTION_HISTORY_SOFT_PASS symbol=%s bars=%d required=%d reason=fresh_tick_scalping_mode' in source
