"""Wake live-readiness exactly at the NSE session-open boundary."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from functools import wraps
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market_hours import MarketState

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


def _sync_trading_switch_with_readiness(ctx: Any) -> bool:
    """Keep the final entry switch consistent with canonical LIVE arming."""

    if not bool(getattr(ctx, "live_orders_armed", False)):
        return False
    switch = trading_switch()
    arm_for_runtime = getattr(switch, "arm_for_runtime", None)
    switch_ready = (
        bool(arm_for_runtime())
        if callable(arm_for_runtime)
        else bool(getattr(switch, "can_trade", lambda: False)())
    )
    if switch_ready:
        return True

    ctx.live_orders_armed = False
    with suppress(Exception):
        ctx.execution_armed = False
    ctx.live_block_reason = "execution_not_armed:trading_switch_off"
    with suppress(Exception):
        ctx.execution_block_reason = ctx.live_block_reason
    LOGGER.warning(
        "TRADING_SWITCH_READINESS_BLOCK live_orders_armed=False reason=trading_switch_off",
        extra={
            "event": "TRADING_SWITCH_READINESS_BLOCK",
            "reason": "trading_switch_off",
        },
    )
    return False


async def _market_open_boundary_worker(app_module: Any, ctx: Any) -> None:
    """Perform one idempotent readiness refresh at each trading-day open."""
    last_rearmed_date = None
    while True:
        state = app_module.get_market_state()
        now_ist = datetime.now(_IST)
        if state == MarketState.OPEN:
            if last_rearmed_date != now_ist.date():
                await app_module._ensure_strategy_runner_started(
                    ctx, reason="market_open_boundary"
                )
                await app_module._recompute_and_push_runtime_readiness(
                    ctx, reason="market_open_boundary"
                )
                last_rearmed_date = now_ist.date()
                LOGGER.info(
                    "MARKET_OPEN_BOUNDARY_REARM_COMPLETE date=%s",
                    now_ist.date().isoformat(),
                    extra={
                        "event": "MARKET_OPEN_BOUNDARY_REARM_COMPLETE",
                        "date": now_ist.date().isoformat(),
                    },
                )
            await asyncio.sleep(60.0)
            continue

        next_open = app_module._next_nse_open_after(now_ist)
        delay = max(0.0, (next_open - now_ist).total_seconds())
        await asyncio.sleep(min(30.0, delay) if delay > 0 else 0.0)


def apply_app_patch(app_module: Any) -> None:
    """Run the exact-boundary worker beside the existing periodic rearm loop."""
    if getattr(app_module, "_session_boundary_rearm_installed", False):
        return
    original = getattr(app_module, "_live_readiness_rearm_loop", None)
    original_recompute = getattr(
        app_module, "_recompute_and_push_runtime_readiness", None
    )
    if not callable(original):
        raise RuntimeError("live readiness rearm loop unavailable")
    if not callable(original_recompute):
        raise RuntimeError("runtime readiness recompute unavailable")

    @wraps(original_recompute)
    async def recompute_and_sync(ctx: Any, *args: Any, **kwargs: Any) -> None:
        await original_recompute(ctx, *args, **kwargs)
        _sync_trading_switch_with_readiness(ctx)

    @wraps(original)
    async def rearm_loop(ctx: Any) -> None:
        boundary_task = asyncio.create_task(
            _market_open_boundary_worker(app_module, ctx),
            name="market_open_boundary_rearm",
        )
        try:
            await original(ctx)
        finally:
            boundary_task.cancel()
            with suppress(asyncio.CancelledError):
                await boundary_task

    app_module._session_boundary_rearm_original = original
    app_module._session_boundary_readiness_recompute_original = original_recompute
    app_module._recompute_and_push_runtime_readiness = recompute_and_sync
    app_module._live_readiness_rearm_loop = rearm_loop
    app_module._session_boundary_rearm_installed = True


__all__ = ["apply_app_patch"]
