"""Session readiness and startup-orchestration adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.utils.logging import log_throttled

_LOGGER = logging.getLogger("nifty_scalper_bot.core.app")
_T = TypeVar("_T")
_OPTION_DIRECTION_CONTEXT_KEYS = {
    "direction_bias",
    "underlying_direction_bias",
    "underlying_direction_confidence",
    "context_age_seconds",
    "context_fresh",
    "direction_context_source",
    "direction_context_reasons",
}


def adapt_compute_live_readiness(
    original: Callable[..., tuple[bool, list[str]]],
) -> Callable[..., tuple[bool, list[str]]]:
    """Keep session details quiet and every live refusal diagnostically explicit."""

    @wraps(original)
    def wrapped(**kwargs: Any) -> tuple[bool, list[str]]:
        adjusted = dict(kwargs)
        if bool(adjusted.get("live_mode")) and not bool(adjusted.get("market_open")):
            min_bars = int(adjusted.get("option_exec_min_bars") or 1)
            adjusted["ce_quote_ready"] = True
            adjusted["pe_quote_ready"] = True
            adjusted["ce_bars"] = max(int(adjusted.get("ce_bars") or 0), min_bars)
            adjusted["pe_bars"] = max(int(adjusted.get("pe_bars") or 0), min_bars)

        armed, reasons = original(**adjusted)
        normalized_reasons = list(reasons or [])
        if bool(adjusted.get("live_mode")) and bool(armed):
            switch = trading_switch()
            arm_for_runtime = getattr(switch, "arm_for_runtime", None)
            switch_ready = (
                bool(arm_for_runtime())
                if callable(arm_for_runtime)
                else bool(switch.can_trade())
            )
            if not switch_ready:
                armed = False
                if "trading_switch_off" not in normalized_reasons:
                    normalized_reasons.append("trading_switch_off")
        if bool(adjusted.get("live_mode")) and not armed and not normalized_reasons:
            minimum = int(adjusted.get("option_exec_min_bars") or 1)
            if not bool(adjusted.get("hard_ready")):
                normalized_reasons.append("startup_pipeline_incomplete")
            elif not bool(adjusted.get("market_open")):
                normalized_reasons.append("market_closed")
            elif not bool(adjusted.get("runner_running")):
                normalized_reasons.append("runner_not_running")
            elif not adjusted.get("selected_ce") or not adjusted.get("selected_pe"):
                normalized_reasons.append("selected_options_missing")
            elif int(adjusted.get("ce_bars") or 0) < minimum:
                normalized_reasons.append("ce_exec_bars_missing")
            elif int(adjusted.get("pe_bars") or 0) < minimum:
                normalized_reasons.append("pe_exec_bars_missing")
            elif not bool(adjusted.get("ce_quote_ready", True)):
                normalized_reasons.append("selected_ce_quote_missing")
            elif not bool(adjusted.get("pe_quote_ready", True)):
                normalized_reasons.append("selected_pe_quote_missing")
            elif not (
                bool(adjusted.get("quote_available"))
                or bool(adjusted.get("ws_quote_proof"))
            ):
                normalized_reasons.append("market_data_proof_unavailable")
            else:
                normalized_reasons.append("readiness_inconsistent")
        return bool(armed), normalized_reasons

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


def adapt_indicator_get_history(original: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
    """Short-circuit missing histories before duplicate INFO instrumentation runs."""

    @wraps(original)
    def wrapped(self: Any, symbol: str, *args: Any, **kwargs: Any) -> list[Any]:
        histories = getattr(self, "_histories", None)
        lock = getattr(self, "_lock", None)
        if not isinstance(histories, dict):
            return original(self, symbol, *args, **kwargs)
        if lock is not None:
            with lock:
                missing = symbol not in histories
        else:
            missing = symbol not in histories
        if not missing:
            return original(self, symbol, *args, **kwargs)

        market_open = True
        try:
            from nifty_scalper_bot.utils.market_hours import is_market_open_now

            market_open = bool(is_market_open_now())
        except Exception:  # noqa: BLE001 - diagnostics must not affect data access
            pass
        logger = getattr(
            self,
            "_logger",
            logging.getLogger("nifty_scalper_bot.strategies.indicators"),
        )
        log_throttled(
            logger,
            (
                f"indicator_history_missing:{symbol}"
                if market_open
                else f"indicator_history_missing_offmarket:{symbol}"
            ),
            (
                "Condition met: indicator_history_missing"
                if market_open
                else "Condition met: indicator_history_missing (market_closed)"
            ),
            interval_sec=60.0 if market_open else 900.0,
            level=logging.INFO if market_open else logging.DEBUG,
            extra={
                "event": "indicator_engine_history_missing",
                "symbol": symbol,
                "market_session_state": "open" if market_open else "closed",
            },
        )
        return []

    return wrapped


def adapt_option_indicator_direction_context(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Keep option indicators free of inherited direction; context snapshots own it."""

    @wraps(original)
    def wrapped(self: Any, symbol: str, *args: Any, **kwargs: Any) -> Any:
        result = original(self, symbol, *args, **kwargs)
        normalized = str(symbol or "").strip().upper().replace(" ", "")
        if not normalized.endswith(("CE", "PE")) or not isinstance(result, Mapping):
            return result
        cleaned = dict(result)
        for key in _OPTION_DIRECTION_CONTEXT_KEYS:
            cleaned.pop(key, None)
        return cleaned

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

    indicator_cls = getattr(app_module, "IndicatorEngine", None)
    if indicator_cls is None:
        try:
            from nifty_scalper_bot.strategies.indicators import IndicatorEngine

            indicator_cls = IndicatorEngine
        except Exception:  # noqa: BLE001 - optional import compatibility
            indicator_cls = None
    if indicator_cls is not None:
        _patch_function(
            indicator_cls,
            "get_history",
            adapt_indicator_get_history,
            "_missing_history_single_log_adapted",
        )
        _patch_function(
            indicator_cls,
            "get_indicators",
            adapt_option_indicator_direction_context,
            "_option_direction_context_authority_adapted",
        )


__all__ = [
    "adapt_compute_live_readiness",
    "adapt_indicator_get_history",
    "adapt_mdm_pipeline_overload",
    "adapt_option_indicator_direction_context",
    "adapt_register_and_subscribe_live_symbol",
    "adapt_replay_latest_mdm_ticks_to_bus",
    "adapt_sync_history_from_mdm",
    "adapt_wire_and_start_message_bus",
    "apply_app_patch",
]
