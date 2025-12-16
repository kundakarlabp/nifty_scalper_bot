"""
Factory for building elite strategies dynamically.
World-Class implementation: Safe Builders + Static Tag Resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, cast

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    EliteStrategiesSettings,
    EliteStrategyConfig,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    RSIDivergenceStrategyConfig,
    SMCStrategyConfig,
    StraddleThetaStrategyConfig,
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import (
    CPRBreakoutStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import (
    GammaScalpingStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.oi_max_pain import OIMaxPainStrategy
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import (
    RSIDivergenceStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.straddle_theta import (
    StraddleThetaStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class _StrategyPlan:
    factory: Callable[..., EliteStrategy]
    config: EliteStrategyConfig
    tags: List[str]


def build_elite_strategies(
    settings: EliteStrategiesSettings, indicator_engine: Any
) -> Sequence[EliteStrategy]:
    """
    Construct active elite strategy instances.
    Safely injects configuration and indicator engine.
    """
    strategies: list[EliteStrategy] = []
    plans: list[_StrategyPlan] = []

    # Map settings to plans
    if settings.smc.enabled:
        plans.append(
            _StrategyPlan(SMCStrategy, settings.smc, ["reversal", "liquidity"])
        )
    if settings.vwap.enabled:
        plans.append(
            _StrategyPlan(VWAPProStrategy, settings.vwap, ["trend", "intraday"])
        )
    if settings.oi_max_pain.enabled:
        plans.append(
            _StrategyPlan(OIMaxPainStrategy, settings.oi_max_pain, ["reversion", "oi"])
        )
    if settings.gamma_scalping.enabled:
        plans.append(
            _StrategyPlan(
                GammaScalpingStrategy, settings.gamma_scalping, ["volatility", "gamma"]
            )
        )
    if settings.cpr.enabled:
        plans.append(
            _StrategyPlan(CPRBreakoutStrategy, settings.cpr, ["breakout", "levels"])
        )
    if settings.order_flow.enabled:
        plans.append(
            _StrategyPlan(OrderFlowStrategy, settings.order_flow, ["scalp", "depth"])
        )
    if settings.bb_squeeze.enabled:
        plans.append(
            _StrategyPlan(BBSqueezeStrategy, settings.bb_squeeze, ["volatility", "breakout"])
        )
    if settings.rsi_div.enabled:
        plans.append(
            _StrategyPlan(RSIDivergenceStrategy, settings.rsi_div, ["reversal", "momentum"])
        )
    if settings.orb.enabled:
        plans.append(
            _StrategyPlan(ORBProStrategy, settings.orb, ["breakout", "opening"])
        )
    if settings.straddle.enabled:
        plans.append(
            _StrategyPlan(
                StraddleThetaStrategy, settings.straddle, ["theta", "delta_neutral"]
            )
        )

    # Build instances
    for plan in plans:
        try:
            # CRITICAL: We pass both config and engine here.
            # This matches the new __init__ signatures we fixed.
            strategy = plan.factory(config=plan.config, indicator_engine=indicator_engine)
            strategies.append(strategy)
            LOGGER.info("Built elite strategy: %s", strategy.name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failed to build strategy %s: %s",
                plan.factory.__name__,
                exc,
                exc_info=exc,
                extra={"event": "strategy_build_failed"},
            )

    return strategies


def elite_strategy_tags(settings: EliteStrategiesSettings) -> dict[str, List[str]]:
    """
    Return map of strategy names to their capability tags.
    Updated to use STATIC mapping to avoid instantiation crashes.
    """
    tags: dict[str, List[str]] = {}
    
    # We map the Factory Class -> The Hardcoded Name used in __init__
    # This avoids creating the object (which requires engine) just to read the name.
    
    if settings.smc.enabled:
        tags["SMC Liquidity"] = ["reversal", "liquidity"]
        
    if settings.vwap.enabled:
        tags["VWAP Pro"] = ["trend", "intraday"]
        
    if settings.oi_max_pain.enabled:
        tags["OI Max Pain"] = ["reversion", "oi"]
        
    if settings.gamma_scalping.enabled:
        tags["Gamma Scalping"] = ["volatility", "gamma"]
        
    if settings.cpr.enabled:
        tags["CPR Breakout"] = ["breakout", "levels"]
        
    if settings.order_flow.enabled:
        tags["Order Flow Imbalance"] = ["scalp", "depth"]
        
    if settings.bb_squeeze.enabled:
        tags["BB Squeeze"] = ["volatility", "breakout"]
        
    if settings.rsi_div.enabled:
        tags["RSI Divergence"] = ["reversal", "momentum"]
        
    if settings.orb.enabled:
        tags["ORB Pro"] = ["breakout", "opening"]
        
    if settings.straddle.enabled:
        tags["ATM Straddle Theta"] = ["theta", "delta_neutral"]

    return tags


__all__ = [
    "EliteSignal",
    "EliteStrategy",
    "EliteStrategyConfig",
    "SMCStrategyConfig",
    "VWAPProStrategyConfig",
    "OIMaxPainStrategyConfig",
    "GammaScalpingStrategyConfig",
    "CPRBreakoutStrategyConfig",
    "OrderFlowStrategyConfig",
    "BBSqueezeStrategyConfig",
    "RSIDivergenceStrategyConfig",
    "ORBProStrategyConfig",
    "StraddleThetaStrategyConfig",
    "EliteStrategiesSettings",
    "SMCStrategy",
    "VWAPProStrategy",
    "OIMaxPainStrategy",
    "GammaScalpingStrategy",
    "CPRBreakoutStrategy",
    "OrderFlowStrategy",
    "BBSqueezeStrategy",
    "RSIDivergenceStrategy",
    "ORBProStrategy",
    "StraddleThetaStrategy",
    "build_elite_strategies",
    "elite_strategy_tags",
]
