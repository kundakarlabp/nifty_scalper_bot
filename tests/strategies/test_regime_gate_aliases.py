"""Runner regime gate must resolve legacy aliases through the canonical ontology.

This previously asserted on literal alias strings in ``runner.py`` source text,
which passed while the gate compared against a vocabulary the regime engine
never emitted. It now exercises the gate itself.
"""

from pathlib import Path

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_legacy_regime_aliases_resolve_to_canonical_states() -> None:
    assert normalize_regime("VOLATILE") is MarketRegime.VOLATILE
    assert normalize_regime("HIGH_VOLATILITY") is MarketRegime.VOLATILE
    assert normalize_regime("TRENDING") is MarketRegime.TREND
    assert normalize_regime("RANGING") is MarketRegime.RANGE


def test_runner_gate_accepts_legacy_alias_configuration(monkeypatch) -> None:
    """Operators with legacy env values must keep working after canonicalisation."""
    monkeypatch.setenv("RUNNER_ENABLE_REGIME_GATE", "true")
    monkeypatch.setenv("RUNNER_VWAP_ALLOWED_REGIMES", "TREND,HIGH_VOLATILITY")
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = __import__("logging").getLogger("test")

    assert StrategyRunner._strategy_allowed_for_regime(
        runner, "vwap_pro", MarketRegime.VOLATILE
    )
    assert not StrategyRunner._strategy_allowed_for_regime(
        runner, "vwap_pro", MarketRegime.RANGE
    )


def test_runner_gate_blocks_unresolved_and_inactive_regimes(monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_ENABLE_REGIME_GATE", "true")
    monkeypatch.delenv("RUNNER_ORB_ALLOWED_REGIMES", raising=False)
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = __import__("logging").getLogger("test")

    for regime in (MarketRegime.UNKNOWN, MarketRegime.LOW_ACTIVITY):
        assert not StrategyRunner._strategy_allowed_for_regime(runner, "orb_pro", regime)


def test_runner_no_longer_declares_a_private_regime_vocabulary() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert '"VOLATILE": "HIGH_VOLATILITY"' not in source
    assert "REGIME_GATE_DECISION" in source
