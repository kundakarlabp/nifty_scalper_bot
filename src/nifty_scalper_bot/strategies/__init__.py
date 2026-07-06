# Elite-only export surface

from .elite_strategies import *  # noqa: F401,F403

# Keep the runtime context contract active for direct imports such as
# ``nifty_scalper_bot.strategies.indicators``.  This preserves live direction
# fields required by OrderFlowStrategy without admitting arbitrary context keys.
from .runtime_context_contract import install_indicator_runtime_context_contract

install_indicator_runtime_context_contract()

# explicit __all__ comes from elite_strategies.__init__
try:
    from .elite_strategies import __all__ as _elite_all  # type: ignore

    __all__ = list(_elite_all)
except Exception:
    __all__ = []
