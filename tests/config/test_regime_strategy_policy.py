from __future__ import annotations

import json

from nifty_scalper_bot.config.regime_ontology import MarketRegime
import nifty_scalper_bot.config.regime_strategy_policy as regime_policy


def test_regime_strategy_policy_preserves_current_weights() -> None:
    assert regime_policy.regime_strategy_weight(MarketRegime.TREND, "SMC") == 1.2
    assert regime_policy.regime_strategy_weight("RANGE", "ORBPro") == 0.7
    assert regime_policy.regime_strategy_weight("VOLATILE", "OrderFlow") == 0.75
    assert regime_policy.regime_strategy_weight("TREND", "smc_lite") == 1.2
    assert regime_policy.regime_strategy_weight("RANGE", "orb_pro") == 0.7
    assert regime_policy.regime_strategy_weight("VOLATILE", "order_flow") == 0.75


def test_regime_strategy_policy_defaults_to_neutral_for_unknown_pair() -> None:
    assert regime_policy.regime_strategy_weight(MarketRegime.TREND, "UnknownStrategy") == 1.0
    assert regime_policy.regime_strategy_weight(None, "SMC") == 1.0


def test_regime_strategy_policy_remains_profile_json_serializable() -> None:
    encoded = json.dumps({"strategy_weights": regime_policy.REGIME_STRATEGY_WEIGHTS}, sort_keys=True)
    assert '"TREND"' in encoded
    assert '"smc_lite"' in encoded


def test_regime_strategy_compatibility_is_structural_and_non_numeric() -> None:
    assert regime_policy.regime_strategy_compatibility("TREND", "VWAPPro") == "preferred"
    assert regime_policy.regime_strategy_compatibility("RANGE", "VWAPPro") == "compatible"
    assert regime_policy.regime_strategy_compatibility("EVENT", "VWAPPro") == "caution"
    assert regime_policy.regime_strategy_compatibility("UNKNOWN", "VWAPPro") == "unknown"


