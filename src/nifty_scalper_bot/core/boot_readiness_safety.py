"""Session readiness and startup-orchestration adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

_LOGGER = logging.getLogger("nifty_scalper_bot.core.app")
_T = TypeVar("_T")


def adapt_compute_live_readiness(
    original: Callable[..., tuple[bool, list[str]]],
) -> Callable[..., tuple[bool, list[str]]]:
    """Return an adapter that keeps option quote checks quiet outside session."""

    @wraps(original)
    def wrapped(**kwargs: Any) -> tuple[bool, list[str]]:
        if bool(kwargs.get("live_mode")) and not bool(kwargs.get("market_open")):
            min_bars = int(kwargs.get("option_exec_min_bars") or 1)
            adjusted = dict(kwargs)
            adjusted["ce_quote_ready"] = True
            adjusted["pe_quote_ready"] = True
            adjusted["ce_bars"] = max(int(adjusted.get("ce_bars") or 0), min_bars)
            adjusted["pe_bars"] = max(int(adjusted.get("pe_bars") or 0), min_bars)
            return original(**adjusted)
        return original(**kwargs)

    return wrapped


def adapt_replay_latest_mdm_ticks_to_bus(
    original: Callable[..., Awaitable[int]],
) -> Callable[..., Awaitable[int]]:
    """Replay cached ticks through the active authoritative ingress."""

    @wraps(original)
    async def wrapped(ctx: Any, *, reason: str) -> int:
        if bool(getattr(ctx, "data_observation_ready", False)):
            return int(await original(ctx, reason=reason))

        bus = getattr(ctx, "message_bus", None)
        if bus is not None and bool(getattr(bus, "running", False)):
            return int(await original(ctx, reason=reason))

        mdm = getattr(ctx, "market_data_manager", None)
        hub = getattr(ctx, "data_hub", None)
        ingest = getattr(hub, "ingest_tick_sync", None)
        latest_ticks = getattr(mdm, "_latest_ticks", {}) if mdm is not None else {}
        if callable(ingest) and isinstance(latest_ticks, Mapping):
            replayed = 0
            for symbol, tick in list(latest_ticks.items()):
                if not isinstance(tick, Mapping):
                    continue
                payload = dict(tick)
                payload["symbol"] = str(symbol)
                payload["source"] = "mdm_replay"
                ingest(payload)
                replayed += 1
            _LOGGER.info(
                "MDM_CACHED_TICKS_REPLAYED count=%d reason=%s path=direct_datahub",
                replayed,
                reason,
                extra={
                    "event": "MDM_CACHED_TICKS_REPLAYED",
                    "count": replayed,
                    "reason": reason,
                    "path": "direct_datahub",
                },
            )
            return replayed

        _LOGGER.info(
            "MDM_CACHED_TICKS_REPLAY_SKIPPED "
            "reason=message_bus_not_running requested_reason=%s",
            reason,
            extra={
                "event": "MDM_CACHED_TICKS_REPLAY_SKIPPED",
                "reason": "message_bus_not_running",
                "requested_reason": reason,
            },
        )
        return 0

    return wrapped


class _RunnerWiringView:
    """Expose runner callbacks while withholding activation authority."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        if name == "add_symbol":
            raise AttributeError(name)
        return getattr(self._runner, name)


class _DeferredDataHubView:
    """Keep startup subscriptions deferred until the canonical readiness flush."""

    def __init__(self, hub: Any) -> None:
        self._hub = hub

    def subscribe_ticks(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["force_live"] = False
        return self._hub.subscribe_ticks(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hub, name)


class _WiringContextView:
    """Read-through context with startup-safe runner/DataHub views."""

    def __init__(
        self,
        ctx: Any,
        *,
        defer_datahub: bool,
        withhold_runner_activation: bool,
    ) -> None:
        self._ctx = ctx
        runner = getattr(ctx, "strategy_runner", None)
        hub = getattr(ctx, "data_hub", None)
        self.strategy_runner = (
            _RunnerWiringView(runner)
            if withhold_runner_activation and runner is not None
            else runner
        )
        self.data_hub = (
            _DeferredDataHubView(hub) if defer_datahub and hub is not None else hub
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


def adapt_register_and_subscribe_live_symbol(
    original: Callable[..., bool],
) -> Callable[..., bool]:
    """Wire market data without bypassing the canonical runner-readiness gate."""

    @wraps(original)
    def wrapped(
        ctx: Any,
        symbol: str | None,
        token: int | None,
        reason: str,
        role: str = "tradable_option",
    ) -> bool:
        startup_wiring = reason == "basket_commit_live_startup"
        view = _WiringContextView(
            ctx,
            defer_datahub=startup_wiring,
            withhold_runner_activation=startup_wiring,
        )
        return bool(original(view, symbol, token, reason, role))

    return wrapped


def adapt_wire_and_start_message_bus(
    original: Callable[..., bool],
) -> Callable[..., bool]:
    """Detach an inactive, subscriber-less bus from direct MDM tick ingress."""

    @wraps(original)
    def wrapped(ctx: Any) -> bool:
        started = bool(original(ctx))
        bus = getattr(ctx, "message_bus", None)
        mdm = getattr(ctx, "market_data_manager", None)
        subscribers = getattr(bus, "subscribers", {}) if bus is not None else {}
        has_subscribers = bool(
            isinstance(subscribers, dict)
            and any(bool(items) for items in subscribers.values())
        )
        if not started and not has_subscribers and mdm is not None:
            if getattr(mdm, "bus", None) is bus:
                mdm.bus = None
        return started

    return wrapped


def adapt_sync_history_from_mdm(original: Callable[..., _T]) -> Callable[..., _T]:
    """Correct unambiguous spot/futures history-role mismatches at the SSOT."""

    @wraps(original)
    def wrapped(self: Any, symbol: str, *args: Any, **kwargs: Any) -> _T:
        normalized = str(symbol or "").strip().upper().replace(" ", "")
        role = str(kwargs.get("role") or "").strip().lower()
        if normalized.endswith("FUT") and role != "futures_context":
            kwargs["role"] = "futures_context"
        elif (
            normalized in {"NIFTY", "NIFTY50", "NSE:NIFTY", "NSE:NIFTY50"}
            and role != "spot_context"
        ):
            kwargs["role"] = "spot_context"
        return original(self, symbol, *args, **kwargs)

    return wrapped


def adapt_mdm_pipeline_overload(original: Callable[..., Any]) -> Callable[..., Any]:
    """Do not report overload recovery while a tick batch is still in flight."""

    @wraps(original)
    def wrapped(self: Any) -> Any:
        if bool(getattr(self, "_pipeline_overloaded", False)) and int(
            getattr(self, "_tick_active_drains", 0) or 0
        ) > 0:
            return None
        return original(self)

    return wrapped


def _patch_function(
    target: Any,
    name: str,
    adapter: Callable[[Any], Any],
    marker: str,
) -> None:
    current = getattr(target, name, None)
    if not callable(current) or bool(getattr(current, marker, False)):
        return
    wrapped = adapter(current)
    setattr(wrapped, marker, True)
    setattr(target, name, wrapped)


def apply_app_patch(app_module: Any) -> None:
    """Install startup/readiness adapters on a loaded app module."""

    _patch_function(
        app_module,
        "compute_live_readiness",
        adapt_compute_live_readiness,
        "_session_readiness_adapted",
    )
    _patch_function(
        app_module,
        "_replay_latest_mdm_ticks_to_bus",
        adapt_replay_latest_mdm_ticks_to_bus,
        "_inactive_bus_replay_guarded",
    )
    _patch_function(
        app_module,
        "_register_and_subscribe_live_symbol",
        adapt_register_and_subscribe_live_symbol,
        "_runner_activation_gated",
    )
    _patch_function(
        app_module,
        "_wire_and_start_message_bus",
        adapt_wire_and_start_message_bus,
        "_direct_mdm_bus_detach_adapted",
    )

    runner_cls = getattr(app_module, "StrategyRunner", None)
    if runner_cls is None:
        try:
            from nifty_scalper_bot.strategies.runner import StrategyRunner

            runner_cls = StrategyRunner
        except Exception:  # noqa: BLE001 - optional import compatibility
            runner_cls = None
    if runner_cls is not None:
        _patch_function(
            runner_cls,
            "sync_history_from_mdm",
            adapt_sync_history_from_mdm,
            "_history_role_corrected",
        )

    mdm_cls = getattr(app_module, "MarketDataManager", None)
    if mdm_cls is None:
        try:
            from nifty_scalper_bot.data.market_data_manager import MarketDataManager

            mdm_cls = MarketDataManager
        except Exception:  # noqa: BLE001 - optional import compatibility
            mdm_cls = None
    if mdm_cls is not None:
        _patch_function(
            mdm_cls,
            "_update_pipeline_overload_locked",
            adapt_mdm_pipeline_overload,
            "_active_drain_overload_recovery_guarded",
        )


__all__ = [
    "adapt_compute_live_readiness",
    "adapt_mdm_pipeline_overload",
    "adapt_register_and_subscribe_live_symbol",
    "adapt_replay_latest_mdm_ticks_to_bus",
    "adapt_sync_history_from_mdm",
    "adapt_wire_and_start_message_bus",
    "apply_app_patch",
]
