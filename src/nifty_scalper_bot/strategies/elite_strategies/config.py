"""Purpose: Provide stable compatibility exports for elite configuration.

Responsibilities: Re-export the canonical catalog's config models, including
the opt-in RangeVWAPReclaimStrategyConfig; keep existing imports compatible.
Safety: No runtime admission settings are changed by this facade.
"""

from __future__ import annotations

from .builder import strategy_module_names
from .config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    EliteStrategiesSettings,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    RangeVWAPReclaimStrategyConfig,
    RSIDivergenceStrategyConfig,
    SMCStrategyConfig,
    StraddleThetaStrategyConfig,
    TuesdayGammaBuyerStrategyConfig,
    VWAPProStrategyConfig,
)

# Compatibility export; values are derived from the canonical builder catalog.
ELITE_STRATEGY_MODULES = strategy_module_names()

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
    "RangeVWAPReclaimStrategyConfig",
    "SMCStrategyConfig",
    "StraddleThetaStrategyConfig",
    "TuesdayGammaBuyerStrategyConfig",
    "VWAPProStrategyConfig",
]
