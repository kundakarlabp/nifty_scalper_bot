"""File purpose:
    Provide the stable public order-execution API used by the strategy runner.

Key responsibilities:
    - Re-export public order models and helpers from ``order_manager_core``.
    - Expose ``RuntimeOrderManager`` as the single production ``OrderManager``.

Operational constraints:
    - This facade must not add a second execution path or duplicate order state.
    - Runtime recovery and entry gating remain owned by ``RuntimeOrderManager``.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import order_manager_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution import runtime_order_manager as _runtime  # noqa: E402


_original_enrich_trade_plan_exit_provenance = _runtime._enrich_trade_plan_exit_provenance


def _enrich_trade_plan_exit_provenance(plan):
    """Carry the TradePlan bracket anchor contract into durable provenance."""
    plan = _original_enrich_trade_plan_exit_provenance(plan)
    try:
        provenance = dict(getattr(plan, "trade_provenance", {}) or {})
    except (TypeError, ValueError):
        provenance = {}
    provenance.setdefault(
        "bracket_anchor_mode",
        str(getattr(plan, "bracket_anchor_mode", "distance") or "distance"),
    )
    setattr(plan, "trade_provenance", provenance)
    return plan


# RuntimeOrderManager methods resolve this module-global helper at call time.
# Patch that one helper rather than introducing a second submission path.
_runtime._enrich_trade_plan_exit_provenance = _enrich_trade_plan_exit_provenance
RuntimeOrderManager = _runtime.RuntimeOrderManager
OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "OrderManager",
        "RuntimeOrderManager",
    }
)
