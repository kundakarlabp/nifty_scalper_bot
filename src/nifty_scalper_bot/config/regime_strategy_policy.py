"""Canonical regime-to-strategy weighting policy.

This module owns only the structural mapping between the canonical market-regime
ontology and strategy setup families.  It deliberately does not classify the
market, infer direction, generate signals, or decide execution eligibility.

Keep numerical changes evidence-gated: changing a weight is a strategy research
change and must pass chronological out-of-sample / walk-forward validation.
"""

from __future__ import annotations

from nifty_scalper_bot.config.regime_ontology import MarketRegime
from nifty_scalper_bot.config.strategy_taxonomy import normalize_strategy_name

REGIME_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    MarketRegime.TREND.value: {
        "SMC": 1.2,
        "VWAPPro": 1.2,
        "ORBPro": 1.15,
        "BBSqueeze": 1.1,
        "OrderFlow": 1.15,
        "RSIDivergence": 0.8,
    },
    MarketRegime.RANGE.value: {
        "RSIDivergence": 1.15,
        "OIMaxPain": 1.05,
        "ORBPro": 0.7,
        "VWAPPro": 0.8,
        "SMC": 0.85,
    },
    MarketRegime.VOLATILE.value: {
        "SMC": 0.7,
        "VWAPPro": 0.7,
        "ORBPro": 0.7,
        "BBSqueeze": 0.75,
        "OrderFlow": 0.75,
        "RSIDivergence": 0.7,
    },
    MarketRegime.EVENT.value: {
        "SMC": 0.6,
        "VWAPPro": 0.6,
        "ORBPro": 0.6,
        "BBSqueeze": 0.6,
        "OrderFlow": 0.6,
        "RSIDivergence": 0.6,
    },
    MarketRegime.LOW_ACTIVITY.value: {
        "BBSqueeze": 0.6,
        "VWAPPro": 0.7,
        "ORBPro": 0.6,
        "SMC": 0.7,
    },
}


# Structural routing metadata only. These labels do not authorize or block a trade;
# activation requires post-cost chronological validation on this bot's own data.
# Keeping this policy descriptive prevents literature-derived priors from becoming
# unvalidated live alpha.
_REGIME_PREFERRED_FAMILIES: dict[str, frozenset[str]] = {
    MarketRegime.TREND.value: frozenset({"smc_lite", "vwap_pro", "orb_pro", "bb_squeeze"}),
    MarketRegime.RANGE.value: frozenset({"rsi_divergence", "oi_max_pain"}),
    MarketRegime.VOLATILE.value: frozenset({"smc_lite", "vwap_pro", "orb_pro"}),
    MarketRegime.EVENT.value: frozenset(),
    MarketRegime.LOW_ACTIVITY.value: frozenset({"bb_squeeze"}),
}


def regime_strategy_compatibility(
    regime: MarketRegime | str | None, strategy: str
) -> str:
    """Return descriptive regime/setup compatibility without gating execution."""
    regime_key = (
        regime.value
        if isinstance(regime, MarketRegime)
        else str(regime or "").strip().upper()
    )
    strategy_key = normalize_strategy_name(strategy)
    preferred = _REGIME_PREFERRED_FAMILIES.get(regime_key)
    if preferred is None:
        return "unknown"
    if strategy_key in preferred:
        return "preferred"
    if regime_key in {MarketRegime.EVENT.value, MarketRegime.LOW_ACTIVITY.value}:
        return "caution"
    return "compatible"


def regime_strategy_weight(regime: MarketRegime | str | None, strategy: str) -> float:
    """Return the configured regime multiplier, defaulting to neutral weight 1.0."""
    key = (
        regime.value
        if isinstance(regime, MarketRegime)
        else str(regime or "").strip().upper()
    )
    return float(REGIME_STRATEGY_WEIGHTS.get(key, {}).get(str(strategy), 1.0))


__all__ = ["REGIME_STRATEGY_WEIGHTS", "regime_strategy_compatibility", "regime_strategy_weight"]
