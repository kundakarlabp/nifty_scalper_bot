"""
Factory for building elite strategies dynamically.
Production-Grade: Explicit Registry Mapping for Stability and Fault-Tolerance.
"""

from __future__ import annotations

from typing import Any, List, Dict, Type

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import EliteStrategiesSettings

# --- Strategy Class Imports ---
from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import CPRBreakoutStrategy
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import GammaScalpingStrategy
from nifty_scalper_bot.strategies.elite_strategies.oi_max_pain import OIMaxPainStrategy
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import RSIDivergenceStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.straddle_theta import StraddleThetaStrategy
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

def build_elite_strategies(
    settings: EliteStrategiesSettings,
    indicator_engine: Any
) -> List[EliteStrategy]:
    """
    Instantiate enabled strategies using a robust explicit registry.
    
    Args:
        settings: The aggregate EliteStrategiesSettings object.
        indicator_engine: The global engine for data/indicators.
        
    Returns:
        A list of fully initialized strategy instances.
    """
    strategies: List[EliteStrategy] = []

    # ✅ PRODUCTION REGISTRY: Maps Config Field -> Strategy Class
    # This is the "Source of Truth" for loading.
    registry: Dict[str, Type[EliteStrategy]] = {
        "smc": SMCStrategy,
        "vwap": VWAPProStrategy,
        "oi_max_pain": OIMaxPainStrategy,
        "gamma_scalping": GammaScalpingStrategy,
        "cpr": CPRBreakoutStrategy,
        "order_flow": OrderFlowStrategy,
        "bb_squeeze": BBSqueezeStrategy,
        "rsi_div": RSIDivergenceStrategy,
        "orb": ORBProStrategy,
        "straddle": StraddleThetaStrategy,
    }

    LOGGER.info("🏗️  Building Elite Strategy Engine...")

    for field_name, strategy_cls in registry.items():
        try:
            # 1. Verify the config field exists in the settings object
            if not hasattr(settings, field_name):
                LOGGER.warning(f"⚠️  Builder: No config found for '{field_name}'. Skipping.")
                continue

            # 2. Extract specific strategy configuration
            strat_config = getattr(settings, field_name)

            # 3. Check if strategy is enabled in .env / config.yaml
            if not strat_config or not strat_config.enabled:
                LOGGER.debug(f"ℹ️  Strategy '{field_name}' is disabled.")
                continue

            # 4. Instantiate with Dependency Injection
            # Every strategy gets the engine it needs to function.
            strategy_instance = strategy_cls(
                config=strat_config,
                indicator_engine=indicator_engine
            )
            
            strategies.append(strategy_instance)
            LOGGER.info(f"✅ Strategy Loaded: {strategy_instance.name}")

        except Exception as e:
            # Shielding: One strategy failing to load should not kill the bot
            LOGGER.error(f"❌ Critical Failure loading '{field_name}': {e}", exc_info=True)

    passed = len(strategies)
    total = len(registry)
    LOGGER.info(f"📊 Strategy Build Complete: {passed}/{total} active.")

    return strategies


def get_strategy_tags(settings: EliteStrategiesSettings) -> Dict[str, List[str]]:
    """
    Helper for Telegram/UI to show which strategies are active and their focus.
    """
    tags = {}
    
    # Mapping for UI display names
    display_names = {
        "smc": "SMC Liquidity",
        "vwap": "VWAP Pro Pullback",
        "oi_max_pain": "OI Mean Reversion",
        "gamma_scalping": "Gamma Acceleration",
        "cpr": "CPR Trend Breakout",
        "order_flow": "Order Flow Imbalance",
        "bb_squeeze": "BB Volatility Squeeze",
        "rsi_div": "RSI Divergence",
        "orb": "Opening Range Breakout",
        "straddle": "Theta Decay"
    }

    for key, label in display_names.items():
        config = getattr(settings, key, None)
        if config and config.enabled:
            tags[label] = ["Elite", "Active"]
            
    return tags


__all__ = ["build_elite_strategies", "get_strategy_tags"]
