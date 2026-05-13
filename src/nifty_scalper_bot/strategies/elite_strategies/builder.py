"""
Factory for building elite strategies dynamically.
Production-Grade: Explicit Registry Mapping for Stability and Fault-Tolerance.
"""

from __future__ import annotations

import os
from typing import Any, List, Dict, Type

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import EliteStrategiesSettings

# --- Strategy Class Imports ---
from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import CPRBreakoutStrategy
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import GammaScalpingStrategy
from nifty_scalper_bot.strategies.elite_tuesday_gamma_buyer import EliteTuesdayGammaBuyer
from nifty_scalper_bot.strategies.elite_strategies.oi_max_pain import OIMaxPainStrategy
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import RSIDivergenceStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.straddle_theta import StraddleThetaStrategy
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_PRODUCTION_DIRECTIONAL = {'smc', 'vwap', 'order_flow', 'bb_squeeze', 'orb'}
_CONTEXT_ONLY = {'oi_max_pain', 'order_flow'}
_DISABLED_UNTIL_FEATURE_COMPLETE = {'cpr', 'rsi_div'}
_EXPIRY_ONLY = {'gamma_scalping', 'tuesday_gamma_buyer'}
_THETA_ONLY = {'straddle'}

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
    strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).strip().lower()
    allow_expiry_gamma = str(
        os.getenv('ALLOW_EXPIRY_GAMMA_STRATEGIES', 'false')
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    active_names: list[str] = []
    context_names: list[str] = []
    disabled_names: list[str] = []

    # ✅ PRODUCTION REGISTRY: Maps Config Field -> Strategy Class
    # This is the "Source of Truth" for loading.
    registry: Dict[str, Type[EliteStrategy]] = {
        "smc": SMCStrategy,
        "vwap": VWAPProStrategy,
        "oi_max_pain": OIMaxPainStrategy,
        "gamma_scalping": GammaScalpingStrategy,
        "tuesday_gamma_buyer": EliteTuesdayGammaBuyer,
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
                disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                continue

            if field_name == 'oi_max_pain' and str(os.getenv('ENABLE_OI_CONTEXT_PROVIDER', 'false')).strip().lower() not in {'1','true','yes','on'}:
                disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                continue

            if strategy_mode == 'directional_scalp':
                if field_name in _EXPIRY_ONLY or field_name in _THETA_ONLY:
                    disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                    continue
                if field_name not in _PRODUCTION_DIRECTIONAL and field_name not in _CONTEXT_ONLY:
                    disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                    continue
                if field_name in _DISABLED_UNTIL_FEATURE_COMPLETE:
                    if field_name == 'cpr' and str(os.getenv('ENABLE_CPR_EXPERIMENTAL', 'false')).strip().lower() not in {'1','true','yes','on'}:
                        disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                        continue
                    if field_name == 'rsi_div' and str(os.getenv('ENABLE_RSI_DIVERGENCE_EXPERIMENTAL', 'false')).strip().lower() not in {'1','true','yes','on'}:
                        disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                        continue
                if field_name in _CONTEXT_ONLY:
                    context_names.append(strategy_cls.__name__.replace('Strategy', ''))
            if field_name in _EXPIRY_ONLY and not (
                strategy_mode == 'expiry_gamma' and allow_expiry_gamma
            ):
                disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                continue
            if field_name in _THETA_ONLY and strategy_mode != 'theta':
                disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                continue

            # 4. Instantiate with Dependency Injection
            # Every strategy gets the engine it needs to function.
            strategy_instance = strategy_cls(
                config=strat_config,
                indicator_engine=indicator_engine
            )
            
            strategies.append(strategy_instance)
            LOGGER.info(f"✅ Strategy Loaded: {strategy_instance.name}")
            active_names.append(strategy_instance.name)

        except Exception as e:
            # Shielding: One strategy failing to load should not kill the bot
            LOGGER.error(f"❌ Critical Failure loading '{field_name}': {e}", exc_info=True)

    passed = len(strategies)
    total = len(registry)
    LOGGER.info(f"📊 Strategy Build Complete: {passed}/{total} active.")
    LOGGER.info(
        "STRATEGY_REGISTRY_LOADED active=%s context=%s disabled=%s mode=%s",
        active_names,
        context_names or ['OIMaxPain'],
        sorted(set(disabled_names)),
        strategy_mode,
    )

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
        "tuesday_gamma_buyer": "Tuesday Gamma Buyer",
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
