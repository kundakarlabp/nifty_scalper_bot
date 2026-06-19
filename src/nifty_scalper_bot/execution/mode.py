"""Canonical runtime execution-mode resolution.

Runtime role:
- Owns the single source of truth for effective LIVE/PAPER/SHADOW mode.
- Consumes environment, settings, runtime context, and OrderManager state.
- Must not place orders or weaken execution safety gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "y", "on", "live"}


@dataclass(slots=True)
class RuntimeExecutionModeSnapshot:
    effective_mode: str
    live_enabled: bool
    shadow_enabled: bool
    paper_enabled: bool
    order_manager_live: bool
    live_orders_armed: bool
    source: str = "runtime"
    configured_mode: str = "SHADOW"
    mismatch_detected: bool = False
    mismatch_details: dict[str, Any] = field(default_factory=dict)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _get_env(env: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if env is None:
        return os.getenv(key, default)
    return env.get(key, default)


def resolve_runtime_execution_mode(ctx: Any = None, *, env: Mapping[str, Any] | None = None) -> RuntimeExecutionModeSnapshot:
    """Return fail-closed canonical execution mode for dashboards and order gates."""
    settings = getattr(ctx, "settings", None) if ctx is not None else None
    configured = str(
        getattr(settings, "execution_mode", None)
        or _get_env(env, "EXECUTION_MODE", "SHADOW")
        or "SHADOW"
    ).strip().upper()
    if configured == "SIMULATION":
        configured = "PAPER"
    if configured not in {"LIVE", "PAPER", "SHADOW"}:
        configured = "SHADOW"

    live_enabled = bool(
        configured == "LIVE"
        or _truthy(_get_env(env, "ENABLE_LIVE"))
        or _truthy(_get_env(env, "ENABLE_LIVE_TRADING"))
        or _truthy(getattr(ctx, "live_enabled", False) if ctx is not None else False)
    )
    if ctx is not None and settings is not None:
        shadow_enabled = bool(getattr(ctx, "shadow_mode_enabled", False))
        paper_enabled = bool(getattr(ctx, "paper_enabled", False))
    elif ctx is not None and hasattr(ctx, "shadow_mode_enabled"):
        shadow_enabled = bool(getattr(ctx, "shadow_mode_enabled", False))
        paper_enabled = bool(getattr(ctx, "paper_enabled", False))
    else:
        shadow_enabled = bool(_truthy(_get_env(env, "SHADOW_MODE")))
        paper_enabled = bool(
            _truthy(_get_env(env, "PAPER__ENABLED"))
            or _truthy(_get_env(env, "PAPER_MODE"))
        )
    order_manager = getattr(ctx, "order_manager", None) if ctx is not None else None
    order_manager_live = False
    if order_manager is not None:
        is_live = getattr(order_manager, "is_live_mode", None)
        if callable(is_live):
            try:
                order_manager_live = bool(is_live())
            except Exception:
                order_manager_live = False
        else:
            om_mode = str(getattr(order_manager, "execution_mode", "")).upper()
            order_manager_live = (om_mode == "LIVE") if om_mode else False

    if configured == "LIVE" and live_enabled and not shadow_enabled and not paper_enabled:
        effective = "LIVE"
    elif paper_enabled or configured == "PAPER":
        effective = "PAPER"
    else:
        effective = "SHADOW"

    if order_manager is not None and not callable(getattr(order_manager, "is_live_mode", None)) and not str(getattr(order_manager, "execution_mode", "")).strip():
        order_manager_live = effective == "LIVE"
    mismatch_details: dict[str, Any] = {}
    mismatch = False
    if order_manager is not None and (effective == "LIVE") != bool(order_manager_live):
        mismatch = True
        mismatch_details = {
            "configured_mode": configured,
            "effective_mode": effective,
            "order_manager_live": order_manager_live,
            "live_enabled": live_enabled,
            "shadow_enabled": shadow_enabled,
            "paper_enabled": paper_enabled,
        }
    live_orders_armed = bool(
        effective == "LIVE"
        and order_manager_live
        and bool(getattr(ctx, "live_orders_armed", False) if ctx is not None else False)
        and not mismatch
    )
    snap = RuntimeExecutionModeSnapshot(
        effective_mode=effective,
        configured_mode=configured,
        live_enabled=live_enabled,
        shadow_enabled=shadow_enabled,
        paper_enabled=paper_enabled,
        order_manager_live=order_manager_live,
        live_orders_armed=live_orders_armed,
        mismatch_detected=mismatch,
        mismatch_details=mismatch_details,
    )
    if mismatch:
        LOGGER.error("EXECUTION_MODE_MISMATCH %s", mismatch_details, extra={"event": "EXECUTION_MODE_MISMATCH", **mismatch_details})
        if ctx is not None:
            try:
                setattr(ctx, "live_orders_armed", False)
                setattr(ctx, "execution_armed", False)
                setattr(ctx, "live_block_reason", "execution_mode_mismatch")
            except Exception:
                pass
    return snap


__all__ = ["RuntimeExecutionModeSnapshot", "resolve_runtime_execution_mode"]
