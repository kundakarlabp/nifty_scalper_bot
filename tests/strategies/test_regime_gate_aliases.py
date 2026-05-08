from pathlib import Path


def test_regime_gate_aliases_include_volatile_mapping() -> None:
    source = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
    assert '"VOLATILE": "HIGH_VOLATILITY"' in source
    assert '"TRENDING": "TREND"' in source
    assert 'REGIME_GATE_DECISION' in source
