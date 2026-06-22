from importlib import import_module

from .elite_strategies import *  # noqa: F401,F403

try:
    from .elite_strategies import __all__ as _elite_all  # type: ignore
    __all__ = list(_elite_all)
except Exception:
    __all__ = []

_trade_selector_module = import_module(f"{__name__}.trade_selector")
from .hardened_trade_selector import HardenedTradeCandidateSelector  # noqa: E402
_trade_selector_module.TradeCandidateSelector = HardenedTradeCandidateSelector

_runner_module = import_module(f"{__name__}.runner")
_runner_module.TradeCandidateSelector = HardenedTradeCandidateSelector
from .hardened_strategy_runner import HardenedStrategyRunner  # noqa: E402
_runner_module.StrategyRunner = HardenedStrategyRunner

__all__ += ["HardenedTradeCandidateSelector", "HardenedStrategyRunner"]
