"""
Compatibility facade for elite strategy configuration objects.
Standardises imports and prevents circular dependencies across the bot.
"""

from __future__ import annotations

# ✅ Standardized re-exports from the source of truth
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
    TuesdayGammaBuyerStrategyConfig,
    VWAPProStrategyConfig,
)

# ✅ EXPLICIT MODULE REGISTRY
# Used by the StrategyRunner and DataHub to identify valid strategy files.
# These names must match the filenames in the elite_strategies folder exactly.
ELITE_STRATEGY_MODULES = [
    "smc_liquidity",
    "vwap_pro",
    "oi_max_pain",
    "gamma_scalping",
    "elite_tuesday_gamma_buyer",
    "cpr_breakout",
    "order_flow",
    "bb_squeeze",
    "rsi_divergence",
    "orb_pro",
    "straddle_theta",
]

# ✅ PUBLIC API DEFINITION
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
    "TuesdayGammaBuyerStrategyConfig",
    "VWAPProStrategyConfig",
]
