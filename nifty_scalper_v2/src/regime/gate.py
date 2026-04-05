"""Regime-based strategy enable/disable gate."""

from __future__ import annotations

from .detector import RegimeType

# Which regimes allow which strategies
STRATEGY_REGIME_MAP: dict[str, list[RegimeType]] = {
    "vwap_pro":       [RegimeType.TREND, RegimeType.RANGE],
    "smc_liquidity":  [RegimeType.TREND, RegimeType.RANGE],
    "rsi_divergence": [RegimeType.RANGE, RegimeType.CALM],
    "gamma_scalping": [RegimeType.TREND, RegimeType.VOLATILE],
    "oi_max_pain":    [RegimeType.RANGE, RegimeType.CALM],
    "orb_pro":        [RegimeType.TREND, RegimeType.RANGE],
    "bb_squeeze":     [RegimeType.RANGE, RegimeType.TREND],
    "cpr_breakout":   [RegimeType.TREND, RegimeType.RANGE],
    "order_flow":     [RegimeType.TREND, RegimeType.VOLATILE],
    "straddle_theta": [RegimeType.RANGE, RegimeType.CALM],
    "trend_momentum": [RegimeType.TREND],
    "tuesday_gamma":  [RegimeType.TREND, RegimeType.RANGE, RegimeType.VOLATILE],
}


def allows(strategy_name: str, regime: RegimeType) -> bool:
    """Return True if the strategy is allowed in the given regime."""
    allowed_regimes = STRATEGY_REGIME_MAP.get(strategy_name.lower(), list(RegimeType))
    return regime in allowed_regimes
