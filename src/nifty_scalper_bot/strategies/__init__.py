"""Strategy package exports and identity/runtime contract installation.

The identity and runtime-context contracts were previously installed inside
bare ``try/except: pass`` blocks. A failure there left ``deterministic_id``
unpatched and the indicator runtime contract uninstalled, silently degrading
duplicate suppression, the structural stop-rearm gate and the live-safety
identity checks, with no log line anywhere. These contracts are load-bearing,
so a failure must fail loudly rather than be swallowed.
"""

from nifty_scalper_bot.utils.logging import get_logger

from .elite_strategies import *  # noqa: F401,F403
from .runtime_context_contract import install_indicator_runtime_context_contract
from .signal_identity_patch import apply_patches as _apply_signal_identity_patches

LOGGER = get_logger(__name__)

_apply_signal_identity_patches()
install_indicator_runtime_context_contract()

try:
    from .elite_strategies import __all__ as _elite_all  # type: ignore

    __all__ = list(_elite_all)
except ImportError:  # pragma: no cover - optional convenience export only
    LOGGER.warning(
        "ELITE_STRATEGY_EXPORTS_UNAVAILABLE",
        extra={"event": "ELITE_STRATEGY_EXPORTS_UNAVAILABLE"},
    )
    __all__ = []
