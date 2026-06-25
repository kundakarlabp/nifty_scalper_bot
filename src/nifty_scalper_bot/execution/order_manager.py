"""Stable public order API with one explicit runtime manager.

The complete order engine lives in ``order_manager_core``. Public models, enums
and helpers remain available from this module, while ``OrderManager`` resolves
directly to ``RuntimeOrderManager``.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import order_manager_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution.runtime_order_manager import (  # noqa: E402
    RuntimeOrderManager,
)

OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "OrderManager",
        "RuntimeOrderManager",
    }
)
