from pathlib import Path


def test_runner_defends_missing_set_indicators() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert 'hasattr(self._indicator_engine, "set_indicators")' in source
    assert 'INDICATOR_CONTEXT_SETTER_MISSING' in source
    assert 'MARKET_DEPTH_SOFT_FAIL_SIGNAL_CONTINUES' in source
