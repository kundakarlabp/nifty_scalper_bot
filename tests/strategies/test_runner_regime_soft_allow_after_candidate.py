from pathlib import Path


def test_runner_has_no_strategy_specific_regime_admission_or_execution_soft_allow() -> None:
    """Manager weighting and candidate economics are the canonical owners."""
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")

    assert "def _strategy_allowed_for_regime" not in source
    assert "def _strategy_regime_decision" not in source
    assert "REGIME_GATE_REJECTED" not in source
    assert "VWAP_HIGH_VOL_MAX_SPREAD_PCT" not in source
    assert "VWAP_HIGH_VOL_MIN_RR" not in source
    assert "runner_regime_policy" not in source
