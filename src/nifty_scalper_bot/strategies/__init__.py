"""Strategy package exports.

Identity and runtime-context behavior are owned by their native classes. Importing
this package must not mutate Signal, EliteStrategy, or IndicatorEngine methods.
"""

from nifty_scalper_bot.utils.logging import get_logger

from .elite_strategies import *  # noqa: F401,F403

LOGGER = get_logger(__name__)

try:
    from .elite_strategies import __all__ as _elite_all  # type: ignore

    __all__ = list(_elite_all)
except ImportError:  # pragma: no cover - optional convenience export only
    LOGGER.warning(
        "ELITE_STRATEGY_EXPORTS_UNAVAILABLE",
        extra={"event": "ELITE_STRATEGY_EXPORTS_UNAVAILABLE"},
    )
    __all__ = []
