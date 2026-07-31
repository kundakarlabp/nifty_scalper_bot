"""
Factory for building elite strategies dynamically.
Production-Grade: Explicit Registry Mapping for Stability and Fault-Tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence, Type

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy

# --- Strategy Class Imports ---
from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
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
from nifty_scalper_bot.strategies.elite_tuesday_gamma_buyer import (
    EliteTuesdayGammaBuyer,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_PRIMARY_DIRECTIONAL = {'smc', 'vwap', 'orb'}
_CONTEXT_ONLY = {'oi_max_pain', 'order_flow', 'bb_squeeze'}
_UNWIRED_DIRECTIONAL_CONTEXT = {'bb_squeeze'}
_DISABLED_UNTIL_FEATURE_COMPLETE = {'cpr', 'rsi_div'}
_EXPIRY_ONLY = {'gamma_scalping', 'tuesday_gamma_buyer'}
_THETA_ONLY = {'straddle'}
_TRUE_VALUES = {'1', 'true', 'yes', 'on'}


def _env_true(name: str, default: str = 'false') -> bool:
    return str(os.getenv(name, default) or default).strip().lower() in _TRUE_VALUES


def _production_strategy_roles(
    active_names: Sequence[str],
    *,
    strategy_mode: str,
) -> tuple[list[str], list[str]]:
    """Return the effective trigger and context strategy sets."""
    context_names: list[str] = []
    if strategy_mode == 'directional_scalp':
        context_class_names = {
            OIMaxPainStrategy.__name__.replace('Strategy', ''),
            OrderFlowStrategy.__name__.replace('Strategy', ''),
            BBSqueezeStrategy.__name__.replace('Strategy', ''),
        }
        context_names = [name for name in active_names if name in context_class_names]

    if _env_true('ORDERFLOW_ALLOW_TRIGGER_ROLE'):
        context_names = [name for name in context_names if name != 'OrderFlow']
    elif 'OrderFlow' in active_names and 'OrderFlow' not in context_names:
        context_names.append('OrderFlow')

    trigger_names = [name for name in active_names if name not in context_names]
    return trigger_names, context_names


def build_production_strategy_profile(
    *,
    settings: Any,
    strategies: Sequence[EliteStrategy],
    mode_profile: Mapping[str, Any],
    global_min_confidence: float,
) -> dict[str, Any]:
    """Build a deterministic, observational snapshot of material live settings."""
    strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).strip().lower()
    active_names = [str(strategy.name) for strategy in strategies]
    trigger_names, context_names = _production_strategy_roles(
        active_names,
        strategy_mode=strategy_mode,
    )
    confidence_thresholds: dict[str, float] = {}
    for strategy in strategies:
        config = getattr(strategy, 'config', None)
        if config is None:
            continue
        raw_threshold = (
            config.get('min_confidence')
            if isinstance(config, Mapping)
            else getattr(config, 'min_confidence', None)
        )
        if raw_threshold is not None:
            confidence_thresholds[str(strategy.name)] = float(raw_threshold)

    from nifty_scalper_bot.config import settings as app_settings
    from nifty_scalper_bot.core.strategy_manager import REGIME_STRATEGY_WEIGHTS

    profile: dict[str, Any] = {
        'schema_version': 1,
        'execution_mode': str(mode_profile.get('mode') or settings.execution_mode),
        'strategies': {
            'mode': strategy_mode,
            'active': active_names,
            'trigger_capable': trigger_names,
            'context_only': context_names,
        },
        'score_thresholds': {
            'global_min_confidence': float(global_min_confidence),
            'per_strategy_min_confidence': confidence_thresholds,
            'mode_gate': dict(mode_profile),
        },
        'quote_policy': {
            'order_max_age_ms': int(settings.orders.max_quote_age_ms),
            'liquidity_max_spread_pct': float(settings.liquidity.max_spread_pct),
            'live_entry_max_spread_pct': float(
                os.getenv('LIVE_MAX_SPREAD_PCT', '0.75') or '0.75'
            ),
            'order_max_spread_pct': float(
                os.getenv(
                    'ORDER_MAX_SPREAD_PCT',
                    os.getenv('SPREAD_MAX_PCT', '10.0'),
                )
                or '10.0'
            ),
        },
        'risk': {
            'per_trade_risk_pct': float(settings.risk.per_trade_risk_pct),
            'per_trade_cap_pct': float(settings.risk.per_trade_cap_pct),
            'min_lots': int(settings.risk.min_lots_per_trade),
            'max_lots': int(settings.risk.max_lots_per_trade),
            'max_open_positions': int(settings.risk.max_open_positions),
        },
        'exit_policy': {
            **asdict(settings.orders.lifecycle),
            'atr_stop_multiple': float(settings.risk.atr_stop_multiple),
        },
        'regime': {
            'adaptive_enabled': bool(app_settings.USE_REGIME_ADAPTIVE),
            'sizing_multipliers': {
                'trend': float(app_settings.REGIME_TREND_SIZING_MULT),
                'range': float(app_settings.REGIME_RANGE_SIZING_MULT),
                'volatile': float(app_settings.REGIME_VOLATILE_SIZING_MULT),
                'event': float(app_settings.REGIME_EVENT_SIZING_MULT),
            },
            'strategy_weights': REGIME_STRATEGY_WEIGHTS,
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    profile['version'] = f"production-v1-{digest}"
    return profile


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
    allow_expiry_gamma = _env_true('ALLOW_EXPIRY_GAMMA_STRATEGIES')
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

            if field_name == 'oi_max_pain' and not _env_true('ENABLE_OI_CONTEXT_PROVIDER'):
                disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                continue

            if strategy_mode == 'directional_scalp':
                if field_name in _EXPIRY_ONLY or field_name in _THETA_ONLY:
                    disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                    continue
                if field_name in _UNWIRED_DIRECTIONAL_CONTEXT:
                    disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                    continue
                if field_name not in _PRIMARY_DIRECTIONAL and field_name not in _CONTEXT_ONLY:
                    disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                    continue
                if field_name in _DISABLED_UNTIL_FEATURE_COMPLETE:
                    if field_name == 'cpr' and not _env_true('ENABLE_CPR_EXPERIMENTAL'):
                        disabled_names.append(strategy_cls.__name__.replace('Strategy', ''))
                        continue
                    if field_name == 'rsi_div' and not _env_true('ENABLE_RSI_DIVERGENCE_EXPERIMENTAL'):
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
    trigger_capable, context_names = _production_strategy_roles(
        active_names,
        strategy_mode=strategy_mode,
    )
    LOGGER.info(
        "STRATEGY_PRODUCTION_SET active=%s trigger_capable=%s context_only=%s disabled_experimental=%s",
        active_names,
        trigger_capable,
        context_names or ['OIMaxPain'],
        sorted(set(disabled_names)),
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


__all__ = [
    "build_elite_strategies",
    "build_production_strategy_profile",
    "get_strategy_tags",
]
