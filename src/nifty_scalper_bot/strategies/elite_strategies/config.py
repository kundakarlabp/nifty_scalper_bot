"""Compatibility facade for elite strategy configuration objects."""

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
    "SMCStrategyConfig",
    "StraddleThetaStrategyConfig",
    "TuesdayGammaBuyerStrategyConfig",
    "VWAPProStrategyConfig",
]
