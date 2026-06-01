"""Compatibility helpers for legacy strategy integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.strategies.elite_strategies.config import ELITE_STRATEGY_MODULES
from nifty_scalper_bot.strategies.registry import load_strategy_module
from nifty_scalper_bot.utils.logging import get_logger

from .signal_generator import Signal, Strategy

LOGGER = get_logger(__name__)
ACTIVE_STRATEGIES = ['SMC']


def _load_enabled_strategy_modules() -> list[str]:
    """Load enabled elite strategy modules from YAML.

    Args:
        None.

    Returns:
        list[str]: Ordered module identifiers from configuration.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _load_enabled_strategy_modules",
        extra={"event": "load_enabled_strategy_modules"},
    )
    config_path = Path(__file__).resolve().parents[1] / "config" / "strategies.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.warning(
            "strategies.yaml missing; using default elite modules",
            extra={"event": "strategy_config_missing"},
        )
        return list(ELITE_STRATEGY_MODULES)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure reading strategies.yaml: %s",
            exc,
            exc_info=exc,
            extra={"event": "strategy_config_error"},
        )
        return list(ELITE_STRATEGY_MODULES)
    modules = payload.get("enabled_strategies", [])
    if not isinstance(modules, list):
        LOGGER.error(
            "Invalid enabled_strategies payload; falling back to defaults",
            extra={"event": "strategy_config_invalid"},
        )
        return list(ELITE_STRATEGY_MODULES)
    normalized: list[str] = []
    for module in modules:
        candidate = str(module).strip()
        if candidate and candidate in ELITE_STRATEGY_MODULES:
            normalized.append(candidate)
    if not normalized:
        LOGGER.warning(
            "No elite modules enabled; using defaults",
            extra={"event": "strategy_config_empty"},
        )
        default_modules = list(ELITE_STRATEGY_MODULES)
        LOGGER.info(
            "STRATEGY_REGISTRY_LOADED strategies=%s source=default",
            default_modules,
            extra={
                "event": "STRATEGY_REGISTRY_LOADED",
                "strategies": default_modules,
                "source": "default",
            },
        )
        return default_modules
    LOGGER.info(
        "STRATEGY_REGISTRY_LOADED strategies=%s source=config",
        normalized,
        extra={
            "event": "STRATEGY_REGISTRY_LOADED",
            "strategies": normalized,
            "source": "config",
        },
    )
    return normalized


def basic_signal(
    symbol: str,
    ltp: float,
    indicators: dict[str, float] | None = None,
) -> Signal | None:
    """Generate a minimal signal using the elite RSI divergence strategy.

    Args:
        symbol: Trading symbol being evaluated.
        ltp: Latest traded price for *symbol*.
        indicators: Optional indicator snapshot.

    Returns:
        Signal | None: Generated signal when conditions permit.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered basic_signal",
        extra={"event": "basic_signal", "symbol": symbol},
    )
    indicators = indicators or {}
    try:
        strategy = load_strategy_module("rsi_divergence")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in basic_signal: %s",
            exc,
            exc_info=exc,
            extra={"event": "basic_signal_error", "symbol": symbol},
        )
        return None
    return strategy.generate_signal(symbol, indicators, ltp, position=None)


def build_default_strategy_manager(
    indicator_engine: Any, position_manager: Any
) -> StrategyManager:
    """Create a strategy manager initialised with elite strategies.

    Args:
        indicator_engine: Indicator engine supplying market context.
        position_manager: Position manager used for exposure checks.

    Returns:
        StrategyManager: Manager initialised with configured strategies.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered build_default_strategy_manager",
        extra={"event": "build_default_strategy_manager"},
    )
    modules = _load_enabled_strategy_modules()
    strategies: list[Strategy] = []
    for module in modules:
        try:
            strategy_instance = load_strategy_module(module)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure loading elite strategy %s: %s",
                module,
                exc,
                exc_info=exc,
                extra={"event": "elite_strategy_load_error", "strategy_module": module},
            )
            continue
        strategies.append(strategy_instance)
    if not strategies:
        LOGGER.warning(
            "No elite strategies available; manager will be empty",
            extra={"event": "elite_strategies_unavailable"},
        )
    return StrategyManager(strategies, indicator_engine, position_manager)


__all__ = [
    "Signal",
    "Strategy",
    "StrategyManager",
    "basic_signal",
    "build_default_strategy_manager",
]
