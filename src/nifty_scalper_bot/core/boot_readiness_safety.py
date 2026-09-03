"""Session readiness and startup-orchestration adapters."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar
from zoneinfo import ZoneInfo

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.utils.logging import log_throttled

_LOGGER = logging.getLogger("nifty_scalper_bot.core.app")
_T = TypeVar("_T")
_INDIA_TZ = ZoneInfo("Asia/Kolkata")
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


def _bar_timestamp(row: Mapping[str, Any]) -> datetime | None:
    """Return a UTC bar timestamp for safe same-session history reconciliation."""
    value = row.get("timestamp")
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        try:
            result = datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).replace(microsecond=0)


def _indicator_bar_rows(runner: Any, symbol: str) -> list[dict[str, Any]]:
    """Read completed IndicatorEngine bars without reaching into broker hydration."""
    engine = getattr(runner, "_indicator_engine", None)
    getter = getattr(engine, "get_history", None)
    if not callable(getter):
        return []
    try:
        rows = getter(symbol, field="bars")
    except Exception:  # noqa: BLE001 - preservation is defensive, not readiness authority
        return []
    result: list[dict[str, Any]] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        if row.get("is_provisional") is True or row.get("is_complete") is False:
            continue
        ts = _bar_timestamp(row)
        if ts is None:
            continue
        copied = dict(row)
        copied["timestamp"] = ts
        result.append(copied)
    result.sort(key=lambda item: item["timestamp"])
    return result


def _same_session_merge(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge only the latest IST trading session, preferring post-sync MDM rows."""
    if not before or not after:
        return after
    latest_after = _bar_timestamp(after[-1])
    if latest_after is None:
        return after
    session_date = latest_after.astimezone(_INDIA_TZ).date()
    merged: dict[datetime, dict[str, Any]] = {}
    for row in before:
        ts = _bar_timestamp(row)
        if ts is not None and ts.astimezone(_INDIA_TZ).date() == session_date:
            merged[ts] = dict(row)
    for row in after:
        ts = _bar_timestamp(row)
        if ts is not None and ts.astimezone(_INDIA_TZ).date() == session_date:
            copied = dict(row)
            copied["timestamp"] = ts
            merged[ts] = copied
    return [merged[key] for key in sorted(merged)]


def adapt_sync_history_from_mdm(original: Callable[..., _T]) -> Callable[..., _T]:
    """Correct history roles and prevent destructive underlying-session reseeds.

    The canonical MDM fetch policy intentionally caps hot-path spot/futures fetches.
    A later reseed must not turn that fetch cap into an in-memory retention cap by
    replacing a longer already-observed session with only the newest slice. ORB
    needs the 09:15 opening range while current bars remain fresh. We therefore
    preserve already-known bars from the same IST session for spot/futures only;
    incoming MDM rows remain authoritative for duplicate timestamps.
    """

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
        role = str(kwargs.get("role") or role).strip().lower()

        preserve_context = role in {"spot_context", "futures_context"}
        canonicalizer = getattr(self, "_normalize_symbol", None)
        canonical_symbol = (
            str(canonicalizer(symbol)) if callable(canonicalizer) else str(symbol)
        )
        before = (
            _indicator_bar_rows(self, canonical_symbol) if preserve_context else []
        )

        result = original(self, symbol, *args, **kwargs)
        if not preserve_context or not before or not bool(getattr(result, "success", True)):
            return result

        after = _indicator_bar_rows(self, canonical_symbol)
        merged = _same_session_merge(before, after)
        if len(merged) <= len(after):
            return result

        try:
            raw_cap = os.getenv("RUNNER_SYMBOL_HISTORY_MAX_BARS", "500") or "500"
            cap = max(1, int(float(raw_cap)))
        except (TypeError, ValueError):
            cap = 500
        merged = merged[-cap:]
        required = max(1, int(kwargs.get("required_bars") or 1))
        reason = str(kwargs.get("reason") or "runtime_sync")
        try:
            restored_count = int(
                self.reseed_history_from_bars(
                    canonical_symbol,
                    merged,
                    source=f"{reason}:preserve_context_session",
                    min_bars=min(required, len(merged)),
                )
                or 0
            )
            indicator_after = len(_indicator_bar_rows(self, canonical_symbol))
            if hasattr(result, "runner_bars"):
                result.runner_bars = max(int(getattr(result, "runner_bars", 0) or 0), restored_count)
            if hasattr(result, "indicator_bars"):
                result.indicator_bars = max(
                    int(getattr(result, "indicator_bars", 0) or 0), indicator_after
                )
            log_throttled(
                _LOGGER,
                f"context_session_history_preserved:{role}:{canonical_symbol}",
                "CONTEXT_SESSION_HISTORY_PRESERVED "
                f"symbol={canonical_symbol} role={role} before={len(before)} "
                f"post_sync={len(after)} restored={indicator_after}",
                interval_sec=300.0,
                level=logging.INFO,
                extra={
                    "event": "CONTEXT_SESSION_HISTORY_PRESERVED",
                    "symbol": canonical_symbol,
                    "role": role,
                    "before_bars": len(before),
                    "post_sync_bars": len(after),
                    "restored_bars": indicator_after,
                },
            )
        except Exception as exc:  # noqa: BLE001 - keep canonical sync result authoritative
            _LOGGER.warning(
                "CONTEXT_SESSION_HISTORY_PRESERVE_FAILED symbol=%s role=%s error=%s",
                canonical_symbol,
                role,
                exc,
                extra={
                    "event": "CONTEXT_SESSION_HISTORY_PRESERVE_FAILED",
                    "symbol": canonical_symbol,
                    "role": role,
                    "error_type": type(exc).__name__,
                },
            )
        return result

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
