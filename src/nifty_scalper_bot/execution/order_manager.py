"""File purpose:
    Provide the stable public order-execution API used by the strategy runner.

Key responsibilities:
    - Re-export public order models and helpers from ``order_manager_core``.
    - Expose ``RuntimeOrderManager`` as the single production ``OrderManager``.
    - Leave bracket provenance and quote preflight to the canonical runtime module.

Operational constraints:
    - This facade must not add a second execution path or duplicate order state.
    - Runtime recovery and entry gating remain owned by ``RuntimeOrderManager``.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import order_manager_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution import (  # noqa: E402,I001
    runtime_order_manager as _runtime,
)

RuntimeOrderManager = _runtime.RuntimeOrderManager
# Preserve the existing helper import without a wrapper or runtime mutation.
_enrich_trade_plan_exit_provenance = _runtime._enrich_trade_plan_exit_provenance

OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "OrderManager",
        "RuntimeOrderManager",
    }
)
