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
        "smc_lite": 1.2,
        "vwap_pro": 1.2,
        "orb_pro": 1.15,
        "bb_squeeze": 1.1,
        "order_flow": 1.15,
        "rsi_divergence": 0.8,
    },
    MarketRegime.RANGE.value: {
        "rsi_divergence": 1.15,
        "oi_max_pain": 1.05,
        "orb_pro": 0.7,
        "vwap_pro": 0.8,
        "smc_lite": 0.85,
    },
    MarketRegime.VOLATILE.value: {
        "smc_lite": 0.7,
        "vwap_pro": 0.7,
        "orb_pro": 0.7,
        "bb_squeeze": 0.75,
        "order_flow": 0.75,
        "rsi_divergence": 0.7,
    },
    MarketRegime.EVENT.value: {
        "smc_lite": 0.6,
        "vwap_pro": 0.6,
        "orb_pro": 0.6,
        "bb_squeeze": 0.6,
        "order_flow": 0.6,
        "rsi_divergence": 0.6,
    },
    MarketRegime.LOW_ACTIVITY.value: {
        "bb_squeeze": 0.6,
        "vwap_pro": 0.7,
        "orb_pro": 0.6,
        "smc_lite": 0.7,
    },
}


# Structural routing metadata only. These labels do not authorize or block a trade;
# activation requires post-cost chronological validation on this bot's own data.
# Keeping this policy descriptive prevents literature-derived priors from becoming
# unvalidated live alpha.
_REGIME_PREFERRED_FAMILIES: dict[str, frozenset[str]] = {
    MarketRegime.TREND.value: frozenset(
        {"smc_lite", "vwap_pro", "orb_pro", "bb_squeeze"}
    ),
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


# Explicit Runner admission policies that pre-date observe-only compatibility.
# Keeping identity/default ownership here prevents runner.py from defining a
# second regime-policy vocabulary. Strategies absent from this table are not
# hard-filtered by the Runner regime gate.
RUNNER_REGIME_POLICIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vwap_pro": ("RUNNER_VWAP_ALLOWED_REGIMES", (MarketRegime.TREND.value,)),
    "orb_pro": (
        "RUNNER_ORB_ALLOWED_REGIMES",
        (MarketRegime.TREND.value, MarketRegime.VOLATILE.value),
    ),
    "premium_squeeze": (
        "RUNNER_PREMIUM_SQUEEZE_ALLOWED_REGIMES",
        (MarketRegime.TREND.value, MarketRegime.VOLATILE.value),
    ),
}


def runner_regime_policy(strategy: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the explicit Runner regime policy for a canonical strategy."""
    return RUNNER_REGIME_POLICIES.get(normalize_strategy_name(strategy))


def regime_strategy_weight(regime: MarketRegime | str | None, strategy: str) -> float:
    """Return the configured regime multiplier, defaulting to neutral weight 1.0."""
    key = (
        regime.value
        if isinstance(regime, MarketRegime)
        else str(regime or "").strip().upper()
    )
    return float(
        REGIME_STRATEGY_WEIGHTS.get(key, {}).get(normalize_strategy_name(strategy), 1.0)
    )


__all__ = [
    "REGIME_STRATEGY_WEIGHTS",
    "RUNNER_REGIME_POLICIES",
    "regime_strategy_compatibility",
    "regime_strategy_weight",
    "runner_regime_policy",
]
