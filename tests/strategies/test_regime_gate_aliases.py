"""Runner regime gate must resolve legacy aliases through the canonical ontology.

This previously asserted on literal alias strings in ``runner.py`` source text,
which passed while the gate compared against a vocabulary the regime engine
never emitted. It now exercises the gate itself.
"""

from pathlib import Path

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime


def test_legacy_regime_aliases_resolve_to_canonical_states() -> None:
    assert normalize_regime("VOLATILE") is MarketRegime.VOLATILE
    assert normalize_regime("HIGH_VOLATILITY") is MarketRegime.VOLATILE
    assert normalize_regime("TRENDING") is MarketRegime.TREND
    assert normalize_regime("RANGING") is MarketRegime.RANGE



def test_runner_no_longer_declares_a_private_regime_vocabulary() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert '"VOLATILE": "HIGH_VOLATILITY"' not in source
    assert "RUNNER_VWAP_ALLOWED_REGIMES" not in source
    assert "RUNNER_ORB_ALLOWED_REGIMES" not in source
