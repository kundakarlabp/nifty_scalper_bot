"""
Elite strategy registry.
"""

from importlib import import_module
from typing import Any, Callable

from nifty_scalper_bot.strategies.elite_strategies import config_models

ELITE_MODULES = [
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

CLASS_MAP: dict[str, tuple[str, str, Callable[[], Any] | None]] = {
    "SMC Liquidity": (
        "smc_liquidity",
        "SMCStrategy",
        config_models.SMCStrategyConfig,
    ),
    "VWAP Pro": (
        "vwap_pro",
        "VWAPProStrategy",
        config_models.VWAPProStrategyConfig,
    ),
    "OI Max Pain": (
        "oi_max_pain",
        "OIMaxPainStrategy",
        config_models.OIMaxPainStrategyConfig,
    ),
    "Gamma Scalping": (
        "gamma_scalping",
        "GammaScalpingStrategy",
        config_models.GammaScalpingStrategyConfig,
    ),
    "Tuesday Gamma Buyer": (
        "elite_tuesday_gamma_buyer",
        "EliteTuesdayGammaBuyer",
        config_models.TuesdayGammaBuyerStrategyConfig,
    ),
    "CPR Breakout": (
        "cpr_breakout",
        "CPRBreakoutStrategy",
        config_models.CPRBreakoutStrategyConfig,
    ),
    "Order Flow": (
        "order_flow",
        "OrderFlowStrategy",
        config_models.OrderFlowStrategyConfig,
    ),
    "BB Squeeze": (
        "bb_squeeze",
        "BBSqueezeStrategy",
        config_models.BBSqueezeStrategyConfig,
    ),
    "RSI Divergence": (
        "rsi_divergence",
        "RSIDivergenceStrategy",
        config_models.RSIDivergenceStrategyConfig,
    ),
    "ORB Pro": ("orb_pro", "ORBProStrategy", config_models.ORBProStrategyConfig),
    "ATM Straddle Theta": (
        "straddle_theta",
        "StraddleThetaStrategy",
        config_models.StraddleThetaStrategyConfig,
    ),
}

_MODULE_CLASS_MAP = {
    module: (class_name, config_factory)
    for module, class_name, config_factory in CLASS_MAP.values()
}


def _build_instance(
    module: str,
    class_name: str,
    config_factory: Callable[[], Any] | None,
) -> Any:
    """Instantiate strategy class with optional config factory."""

    mod = import_module(f"nifty_scalper_bot.strategies.elite_strategies.{module}")
    if hasattr(mod, "build"):
        builder = getattr(mod, "build")
        return builder()
    if hasattr(mod, class_name):
        strategy_cls = getattr(mod, class_name)
        if config_factory is not None:
            return strategy_cls(config_factory())
        return strategy_cls()
    raise AttributeError(f"{module} must define build() or class {class_name}")


def load_strategy_by_key(key: str):
    if key not in CLASS_MAP:
        raise KeyError(f"Unknown elite strategy key: {key}")
    module, class_name, config_factory = CLASS_MAP[key]
    return _build_instance(module, class_name, config_factory)


def load_strategy_module(module: str):
    if module not in ELITE_MODULES:
        raise KeyError(f"Unknown elite strategy module: {module}")
    class_name, config_factory = _MODULE_CLASS_MAP[module]
    return _build_instance(module, class_name, config_factory)
