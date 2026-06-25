"""Stable public order API with one explicit runtime manager.

The historical implementation is preserved in ``order_manager_legacy``.  All
public models, enums and helpers remain available from this module, while
``OrderManager`` resolves directly to ``RuntimeOrderManager``.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import order_manager_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

LegacyOrderManager = _legacy.OrderManager

from nifty_scalper_bot.execution.runtime_order_manager import (  # noqa: E402
    RuntimeOrderManager,
)

OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_legacy) if not name.startswith("_")],
        "LegacyOrderManager",
        "OrderManager",
        "RuntimeOrderManager",
    }
)
