"""Elite institutional-grade trading strategies package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, cast

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


@dataclass(frozen=True, slots=True)
class EliteStrategyPlan:
    """Describe how to construct and categorise an elite strategy."""

    config: EliteStrategyConfig
    factory: type[EliteStrategy]
    tags: tuple[str, ...]

    def build(self) -> EliteStrategy | None:
        """Instantiate the strategy when enabled else return ``None``."""

        if not self.config.enabled:
            return None
        factory = cast(Callable[[EliteStrategyConfig], EliteStrategy], self.factory)
        return factory(self.config)


def _build_plan(settings: EliteStrategiesSettings) -> Sequence[EliteStrategyPlan]:
    """Return the deterministic elite strategy build plan for *settings*."""

    return (
        EliteStrategyPlan(settings.smc, SMCStrategy, ("elite", "liquidity")),
        EliteStrategyPlan(
            settings.vwap,
            VWAPProStrategy,
            ("elite", "mean_reversion"),
        ),
        EliteStrategyPlan(
            settings.oi_max_pain,
            OIMaxPainStrategy,
            ("elite", "structure"),
        ),
        EliteStrategyPlan(
            settings.gamma,
            GammaScalpingStrategy,
            ("elite", "volatility"),
        ),
        EliteStrategyPlan(
            settings.cpr,
            CPRBreakoutStrategy,
            ("elite", "momentum"),
        ),
        EliteStrategyPlan(
            settings.order_flow,
            OrderFlowStrategy,
            ("elite", "orderflow"),
        ),
        EliteStrategyPlan(
            settings.bb_squeeze,
            BBSqueezeStrategy,
            ("elite", "volatility"),
        ),
        EliteStrategyPlan(
            settings.rsi_div,
            RSIDivergenceStrategy,
            ("elite", "mean_reversion"),
        ),
        EliteStrategyPlan(settings.orb, ORBProStrategy, ("elite", "opening")),
        EliteStrategyPlan(
            settings.straddle,
            StraddleThetaStrategy,
            ("elite", "income"),
        ),
    )


def build_elite_strategies(settings: EliteStrategiesSettings) -> List[EliteStrategy]:
    """Build elite strategies enabled by configuration.

    Args:
        settings: Aggregated elite strategy settings.

    Returns:
        List[EliteStrategy]: Instantiated strategies when enabled.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered build_elite_strategies",
        extra={"event": "elite_build", "enabled": settings.enabled},
    )
    if not settings.enabled:
        return []
    strategies: list[EliteStrategy] = []
    seen_names: set[str] = set()
    for plan in _build_plan(settings):
        try:
            strategy = plan.build()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure building elite strategy %s: %s",
                plan.factory.__name__,
                exc,
                exc_info=exc,
                extra={"event": "elite_build_error", "strategy": plan.factory.__name__},
            )
            continue
        if strategy is None:
            continue
        if strategy.name in seen_names:
            LOGGER.warning(
                "Duplicate elite strategy detected: %s",
                strategy.name,
                extra={"event": "elite_duplicate", "strategy": strategy.name},
            )
            continue
        seen_names.add(strategy.name)
        strategies.append(strategy)
    return strategies


def elite_strategy_tags(
    settings: EliteStrategiesSettings,
    strategies: Sequence[EliteStrategy] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return correlation tag mapping for enabled elite strategies."""

    tags: dict[str, tuple[str, ...]] = {}
    if not settings.enabled:
        return tags
    indexed: dict[type[EliteStrategy], EliteStrategy] = {}
    if strategies:
        indexed = {type(strategy): strategy for strategy in strategies}
    for plan in _build_plan(settings):
        if not plan.config.enabled:
            continue
        strategy = indexed.get(cast(type[EliteStrategy], plan.factory))
        if strategy is None:
            try:
                factory = cast(
                    Callable[[EliteStrategyConfig], EliteStrategy],
                    plan.factory,
                )
                strategy = factory(plan.config)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure resolving elite strategy tags for %s: %s",
                    plan.factory.__name__,
                    exc,
                    exc_info=exc,
                    extra={
                        "event": "elite_tag_error",
                        "strategy": plan.factory.__name__,
                    },
                )
                continue
        tags[strategy.name] = plan.tags
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
