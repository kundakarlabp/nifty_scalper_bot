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

_PRIMARY_DIRECTIONAL = {"smc", "vwap", "orb"}
_CONTEXT_ONLY = {"oi_max_pain", "order_flow", "bb_squeeze", "cpr", "rsi_div"}
_CONTEXT_CLASS_NAMES = {
    OIMaxPainStrategy.__name__.replace("Strategy", ""),
    OrderFlowStrategy.__name__.replace("Strategy", ""),
    BBSqueezeStrategy.__name__.replace("Strategy", ""),
    CPRBreakoutStrategy.__name__.replace("Strategy", ""),
    RSIDivergenceStrategy.__name__.replace("Strategy", ""),
}
_EXPERIMENTAL_CONTEXT_FLAGS = {
    "bb_squeeze": "ENABLE_BB_SQUEEZE_CONTEXT",
    "cpr": "ENABLE_CPR_EXPERIMENTAL",
    "rsi_div": "ENABLE_RSI_DIVERGENCE_EXPERIMENTAL",
}
_EXPIRY_ONLY = {"gamma_scalping", "tuesday_gamma_buyer"}
_THETA_ONLY = {"straddle"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_true(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default) or default).strip().lower() in _TRUE_VALUES


def _strategy_runtime_role(
    field_name: str,
    *,
    strategy_mode: str,
    allow_expiry_gamma: bool,
) -> str | None:
    """Return canonical runtime role for a strategy field, or None when inactive."""
    if field_name == "oi_max_pain" and not _env_true("ENABLE_OI_CONTEXT_PROVIDER"):
        return None

    experimental_flag = _EXPERIMENTAL_CONTEXT_FLAGS.get(field_name)
    if experimental_flag and not _env_true(experimental_flag):
        return None

    if field_name in _EXPIRY_ONLY:
        return "trigger" if strategy_mode == "expiry_gamma" and allow_expiry_gamma else None
    if field_name in _THETA_ONLY:
        return "trigger" if strategy_mode == "theta" else None

    if strategy_mode == "directional_scalp":
        if field_name in _PRIMARY_DIRECTIONAL:
            return "trigger"
        if field_name in _CONTEXT_ONLY:
            return "context"
        return None

    return "context" if field_name in _CONTEXT_ONLY else "trigger"


def _production_strategy_roles(
    active_names: Sequence[str],
    *,
    strategy_mode: str,
) -> tuple[list[str], list[str]]:
    """Return the effective trigger and context strategy sets."""
    del strategy_mode
    context_names = [name for name in active_names if name in _CONTEXT_CLASS_NAMES]
    trigger_names = [name for name in active_names if name not in _CONTEXT_CLASS_NAMES]
    return trigger_names, context_names


def build_production_strategy_profile(
    *,
    settings: Any,
    strategies: Sequence[EliteStrategy],
    mode_profile: Mapping[str, Any],
    global_min_confidence: float,
) -> dict[str, Any]:
    """Build a deterministic, observational snapshot of material live settings."""
    strategy_mode = str(os.getenv("STRATEGY_MODE", "directional_scalp")).strip().lower()
    active_names = [str(strategy.name) for strategy in strategies]
    trigger_names, context_names = _production_strategy_roles(
        active_names,
        strategy_mode=strategy_mode,
    )
    confidence_thresholds: dict[str, float] = {}
    for strategy in strategies:
        config = getattr(strategy, "config", None)
        if config is None:
            continue
        raw_threshold = (
            config.get("min_confidence")
            if isinstance(config, Mapping)
            else getattr(config, "min_confidence", None)
        )
        if raw_threshold is not None:
            confidence_thresholds[str(strategy.name)] = float(raw_threshold)

    from nifty_scalper_bot.config import settings as app_settings
    from nifty_scalper_bot.core.strategy_manager import REGIME_STRATEGY_WEIGHTS

    profile: dict[str, Any] = {
        "schema_version": 1,
        "execution_mode": str(mode_profile.get("mode") or settings.execution_mode),
        "strategies": {
            "mode": strategy_mode,
            "active": active_names,
            "trigger_capable": trigger_names,
            "context_only": context_names,
        },
        "score_thresholds": {
            "global_min_confidence": float(global_min_confidence),
            "per_strategy_min_confidence": confidence_thresholds,
            "mode_gate": dict(mode_profile),
        },
        "quote_policy": {
            "order_max_age_ms": int(settings.orders.max_quote_age_ms),
            "liquidity_max_spread_pct": float(settings.liquidity.max_spread_pct),
            "live_entry_max_spread_pct": float(
                os.getenv("LIVE_MAX_SPREAD_PCT", "0.75") or "0.75"
            ),
            "order_max_spread_pct": float(
                os.getenv(
                    "ORDER_MAX_SPREAD_PCT",
                    os.getenv("SPREAD_MAX_PCT", "10.0"),
                )
                or "10.0"
            ),
        },
        "risk": {
            "per_trade_risk_pct": float(settings.risk.per_trade_risk_pct),
            "per_trade_cap_pct": float(settings.risk.per_trade_cap_pct),
            "min_lots": int(settings.risk.min_lots_per_trade),
            "max_lots": int(settings.risk.max_lots_per_trade),
            "max_open_positions": int(settings.risk.max_open_positions),
        },
        "exit_policy": {
            **asdict(settings.orders.lifecycle),
            "atr_stop_multiple": float(settings.risk.atr_stop_multiple),
        },
        "regime": {
            "adaptive_enabled": bool(app_settings.USE_REGIME_ADAPTIVE),
            "sizing_multipliers": {
                "trend": float(app_settings.REGIME_TREND_SIZING_MULT),
                "range": float(app_settings.REGIME_RANGE_SIZING_MULT),
                "volatile": float(app_settings.REGIME_VOLATILE_SIZING_MULT),
                "event": float(app_settings.REGIME_EVENT_SIZING_MULT),
            },
            "strategy_weights": REGIME_STRATEGY_WEIGHTS,
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    profile["version"] = f"production-v1-{digest}"
    return profile


def build_elite_strategies(
    settings: EliteStrategiesSettings,
    indicator_engine: Any,
) -> List[EliteStrategy]:
    """Instantiate enabled strategies using the canonical runtime-role policy."""
    strategies: List[EliteStrategy] = []
    strategy_mode = str(os.getenv("STRATEGY_MODE", "directional_scalp")).strip().lower()
    allow_expiry_gamma = _env_true("ALLOW_EXPIRY_GAMMA_STRATEGIES")
    active_names: list[str] = []
    disabled_names: list[str] = []

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
            if not hasattr(settings, field_name):
                LOGGER.warning("⚠️  Builder: No config found for '%s'. Skipping.", field_name)
                continue

            strat_config = getattr(settings, field_name)
            class_name = strategy_cls.__name__.replace("Strategy", "")
            if not strat_config or not strat_config.enabled:
                disabled_names.append(class_name)
                continue

            role = _strategy_runtime_role(
                field_name,
                strategy_mode=strategy_mode,
                allow_expiry_gamma=allow_expiry_gamma,
            )
            if role is None:
                disabled_names.append(class_name)
                continue

            strategy_instance = strategy_cls(
                config=strat_config,
                indicator_engine=indicator_engine,
            )
            strategies.append(strategy_instance)
            active_names.append(strategy_instance.name)
            LOGGER.info(
                "✅ Strategy Loaded: %s role=%s",
                strategy_instance.name,
                role,
            )
        except Exception as exc:
            LOGGER.error(
                "❌ Critical Failure loading '%s': %s",
                field_name,
                exc,
                exc_info=True,
            )

    LOGGER.info("📊 Strategy Build Complete: %s/%s active.", len(strategies), len(registry))
    trigger_capable, context_names = _production_strategy_roles(
        active_names,
        strategy_mode=strategy_mode,
    )
    LOGGER.info(
        "STRATEGY_PRODUCTION_SET active=%s trigger_capable=%s context_only=%s disabled=%s",
        active_names,
        trigger_capable,
        context_names,
        sorted(set(disabled_names)),
    )
    return strategies


def get_strategy_tags(settings: EliteStrategiesSettings) -> Dict[str, List[str]]:
    """Return UI tags only for strategies that can actually load in this mode."""
    tags: Dict[str, List[str]] = {}
    strategy_mode = str(os.getenv("STRATEGY_MODE", "directional_scalp")).strip().lower()
    allow_expiry_gamma = _env_true("ALLOW_EXPIRY_GAMMA_STRATEGIES")
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
        "straddle": "Theta Decay",
    }

    for field_name, label in display_names.items():
        config = getattr(settings, field_name, None)
        if not config or not config.enabled:
            continue
        role = _strategy_runtime_role(
            field_name,
            strategy_mode=strategy_mode,
            allow_expiry_gamma=allow_expiry_gamma,
        )
        if role is not None:
            tags[label] = ["Elite", "Active"]
    return tags


__all__ = [
    "build_elite_strategies",
    "build_production_strategy_profile",
    "get_strategy_tags",
]
