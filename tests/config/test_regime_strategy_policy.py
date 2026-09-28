from __future__ import annotations

import json

from nifty_scalper_bot.config.regime_ontology import MarketRegime
from nifty_scalper_bot.config.regime_strategy_policy import (
    REGIME_STRATEGY_WEIGHTS,
    regime_strategy_compatibility,
    regime_strategy_weight,
    runner_regime_policy,
)


def test_regime_strategy_policy_preserves_current_weights() -> None:
    assert regime_strategy_weight(MarketRegime.TREND, "SMC") == 1.2
    assert regime_strategy_weight("RANGE", "ORBPro") == 0.7
    assert regime_strategy_weight("VOLATILE", "OrderFlow") == 0.75


def test_regime_strategy_policy_defaults_to_neutral_for_unknown_pair() -> None:
    assert regime_strategy_weight(MarketRegime.TREND, "UnknownStrategy") == 1.0
    assert regime_strategy_weight(None, "SMC") == 1.0


def test_regime_strategy_policy_remains_profile_json_serializable() -> None:
    encoded = json.dumps({"strategy_weights": REGIME_STRATEGY_WEIGHTS}, sort_keys=True)
    assert '"TREND"' in encoded
    assert '"SMC"' in encoded


def test_regime_strategy_compatibility_is_structural_and_non_numeric() -> None:
    assert regime_strategy_compatibility("TREND", "VWAPPro") == "preferred"
    assert regime_strategy_compatibility("RANGE", "VWAPPro") == "compatible"
    assert regime_strategy_compatibility("EVENT", "VWAPPro") == "caution"
    assert regime_strategy_compatibility("UNKNOWN", "VWAPPro") == "unknown"


def test_runner_regime_policy_is_canonical_and_explicit_only() -> None:
    assert runner_regime_policy("VWAPPro") == (
        "RUNNER_VWAP_ALLOWED_REGIMES",
        ("TREND",),
    )
    assert runner_regime_policy("ORBPro") == (
        "RUNNER_ORB_ALLOWED_REGIMES",
        ("TREND", "VOLATILE"),
    )
    assert runner_regime_policy("SMC") is None
    assert runner_regime_policy("RSI Mean Reversion") is None
