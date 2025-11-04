"""Compatibility facade for elite strategy configuration objects."""

from __future__ import annotations

from .config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    EliteStrategiesSettings,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    RSIDivergenceStrategyConfig,
    SMCStrategyConfig,
    StraddleThetaStrategyConfig,
    VWAPProStrategyConfig,
)

# Plain list, safe to import from config.settings without creating cycles
ELITE_STRATEGY_MODULES = [
    "smc_liquidity",
    "vwap_pro",
    "oi_max_pain",
    "gamma_scalping",
    "cpr_breakout",
    "order_flow",
    "bb_squeeze",
    "rsi_divergence",
    "orb_pro",
    "straddle_theta",
]

__all__ = [
    "ELITE_STRATEGY_MODULES",
    "BBSqueezeStrategyConfig",
    "CPRBreakoutStrategyConfig",
    "EliteStrategiesSettings",
    "GammaScalpingStrategyConfig",
    "OIMaxPainStrategyConfig",
    "ORBProStrategyConfig",
    "OrderFlowStrategyConfig",
    "RSIDivergenceStrategyConfig",
    "SMCStrategyConfig",
    "StraddleThetaStrategyConfig",
    "VWAPProStrategyConfig",
]
