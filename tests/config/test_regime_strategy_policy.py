from __future__ import annotations

import json

from nifty_scalper_bot.config.regime_ontology import MarketRegime
from nifty_scalper_bot.config.regime_strategy_policy import (
    REGIME_STRATEGY_WEIGHTS,
    regime_strategy_weight,
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
