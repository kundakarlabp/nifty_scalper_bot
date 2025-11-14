"""Tests for elite strategies package integration."""

from __future__ import annotations

import math

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    build_elite_strategies,
    elite_strategy_tags,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy


class _DummyIndicatorEngine:
    """Minimal indicator engine stub for strategy manager tests."""

    def get_indicators(self, symbol: str, names):
        """Return placeholder indicators for requested *names*.

        Args:
            symbol: Trading symbol (unused).
            names: Iterable of indicator names requested.

        Returns:
            dict[str, float | None]: Mapping of requested indicator names to ``None``.

        Raises:
            None.
        """

        return {name: None for name in names}


class _DummyPositionManager:
    """Minimal position manager stub returning no active position."""

    def get_position(self, symbol: str):
        """Return ``None`` for *symbol* to represent no open position.

        Args:
            symbol: Trading symbol inspected (unused).

        Returns:
            None: Always indicates absence of a position.

        Raises:
            None.
        """

        return None


def test_build_elite_strategies_enforces_enabled_flag() -> None:
    """Ensure the factory returns the complete elite bundle by default.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the active elite strategies do not match the bundle.
    """

    settings = EliteStrategiesSettings()
    strategies = build_elite_strategies(settings)
    assert strategies, "Expected elite strategies to be instantiated"
    names = {strategy.name for strategy in strategies}
    expected = {
        "SMC Liquidity",
        "VWAP Mean Reversion Pro",
        "OI Max Pain",
        "Gamma Scalping",
        "CPR Breakout",
        "Order Flow Imbalance",
        "BB Squeeze",
        "RSI Divergence",
        "ORB Pro",
        "ATM Straddle Theta",
    }
    assert names == expected


def test_elite_strategy_tags_align_with_instances() -> None:
    """Ensure elite tag helper covers each active strategy without duplicates."""

    settings = EliteStrategiesSettings()
    strategies = build_elite_strategies(settings)
    tags = elite_strategy_tags(settings, strategies)
    names = {strategy.name for strategy in strategies}
    assert set(tags) == names
    assert all(tag_tuple and tag_tuple[0] == "elite" for tag_tuple in tags.values())


def test_elite_strategy_stats_flow() -> None:
    """Validate elite strategy stats surface through strategy manager.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If stats output mismatches expected structure.
    """

    config = SMCStrategyConfig(
        enabled=True,
        min_confidence=40.0,
        equal_level_tolerance_pct=1.0,
        sweep_distance_points=5.0,
        volume_spike_mult=1.0,
        order_block_volume_mult=1.0,
    )
    strategy = SMCStrategy(config)
    indicator_engine = _DummyIndicatorEngine()
    position_manager = _DummyPositionManager()
    manager = StrategyManager(
        strategies=[strategy],
        indicator_engine=indicator_engine,
        position_manager=position_manager,
    )
    indicators = {
        "atr": 10.0,
        "volume_spike_ratio": 3.0,
        "bollinger_lower": 80.0,
        "bollinger_upper": 100.0,
        "vwap": 90.0,
    }
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators=indicators,
        current_price=120.0,
        position=None,
    )
    assert signal is not None
    stats = manager.get_elite_strategy_stats()
    assert stats, "Elite stats should include generated strategy"
    entry = stats[0]
    assert entry["strategy"] == "SMC Liquidity"
    assert entry["signals"] == 1
    last_signal = entry["last_signal"]
    assert isinstance(last_signal, dict)
    assert math.isclose(last_signal["entry_price"], 120.0, rel_tol=1e-6)


def test_registry_loads_all_modules() -> None:
    """Ensure registry can instantiate every elite module."""

    from nifty_scalper_bot.strategies.elite_strategies.config import (
        ELITE_STRATEGY_MODULES,
    )
    from nifty_scalper_bot.strategies.registry import load_strategy_module

    for module in ELITE_STRATEGY_MODULES:
        strategy = load_strategy_module(module)
        assert strategy is not None
        assert hasattr(strategy, "name")
