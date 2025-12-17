"""
Factory for building elite strategies dynamically.
World-Class implementation: Fault-Tolerant, Reflection-Based, and Safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, cast

# --- Strategy Imports ---
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
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

# --- Config Imports ---
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
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class _StrategyPlan:
    factory: Callable[..., EliteStrategy]
    config: EliteStrategyConfig
    tags: List[str]


# ------------------------------------------------------------------------------
# CENTRAL DEFINITION MAP
# Maps 'config_attribute_name' -> (StrategyClass, [Tags])
# This single source of truth prevents code duplication and sync errors.
# ------------------------------------------------------------------------------
STRATEGY_MAP = {
    "smc": (SMCStrategy, ["reversal", "liquidity"]),
    "vwap": (VWAPProStrategy, ["trend", "intraday"]),
    "oi_max_pain": (OIMaxPainStrategy, ["reversion", "oi"]),
    "gamma_scalping": (GammaScalpingStrategy, ["volatility", "gamma"]),
    "cpr": (CPRBreakoutStrategy, ["breakout", "levels"]),
    "order_flow": (OrderFlowStrategy, ["scalp", "depth"]),
    "bb_squeeze": (BBSqueezeStrategy, ["volatility", "breakout"]),
    "rsi_div": (RSIDivergenceStrategy, ["reversal", "momentum"]),
    "orb": (ORBProStrategy, ["breakout", "opening"]),
    "straddle": (StraddleThetaStrategy, ["theta", "delta_neutral"]),
}


def build_elite_strategies(
    settings: EliteStrategiesSettings, indicator_engine: Any
) -> Sequence[EliteStrategy]:
    """
    Construct active elite strategy instances.
    Safely injects configuration and indicator engine using reflection.
    """
    strategies: list[EliteStrategy] = []
    plans: list[_StrategyPlan] = []

    # 1. Plan Construction Loop (Fault Tolerant)
    for attr_name, (factory_cls, tags) in STRATEGY_MAP.items():
        # A. Safety Check: Does config actually have this field?
        if not hasattr(settings, attr_name):
            LOGGER.debug(
                f"Skipping {attr_name}: Attribute missing in EliteStrategiesSettings."
            )
            continue

        # B. Retrieve Config Object
        strategy_config = getattr(settings, attr_name)
        
        # C. Check Enabled Status (Handle None/Missing safely)
        is_enabled = False
        if strategy_config and hasattr(strategy_config, "enabled"):
            is_enabled = bool(strategy_config.enabled)
            
        if is_enabled:
            plans.append(_StrategyPlan(factory_cls, strategy_config, tags))

    # 2. Instantiation Loop (Crash Proof)
    for plan in plans:
        try:
            # D. Inject Dependencies (Standardized Constructor)
            # This is where we ensure 'indicator_engine' is passed
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
    Uses static mapping to avoid instantiation overhead/crashes.
    """
    tags: dict[str, List[str]] = {}
    
    for attr_name, (factory_cls, tag_list) in STRATEGY_MAP.items():
        # 1. Check existence
        if not hasattr(settings, attr_name):
            continue
            
        # 2. Check enabled
        cfg = getattr(settings, attr_name)
        if cfg and getattr(cfg, "enabled", False):
            # We use a readable name key based on the class name or attribute
            # E.g. "SMC Liquidity" or "smc"
            # Using factory class name for consistency if needed, or a pretty name
            # Here we map keys to match the previous implementation's style
            pretty_name = _get_pretty_name(attr_name)
            tags[pretty_name] = tag_list

    return tags


def _get_pretty_name(attr_name: str) -> str:
    """Helper to format attribute names into readable strategy titles."""
    lookup = {
        "smc": "SMC Liquidity",
        "vwap": "VWAP Pro",
        "oi_max_pain": "OI Max Pain",
        "gamma_scalping": "Gamma Scalping",
        "cpr": "CPR Breakout",
        "order_flow": "Order Flow Imbalance",
        "bb_squeeze": "BB Squeeze",
        "rsi_div": "RSI Divergence",
        "orb": "ORB Pro",
        "straddle": "ATM Straddle Theta"
    }
    return lookup.get(attr_name, attr_name.replace("_", " ").title())


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
