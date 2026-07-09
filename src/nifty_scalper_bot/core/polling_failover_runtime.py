"""Runtime adapter for polling fallback supervision.

The polling path is a recovery path.  It must not activate only because a broad
``options_fresh=False`` flag was produced while the selected option quote age is
well below the configured stale threshold.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
from typing import Any, Mapping

from nifty_scalper_bot.core.polling_failover import decide_polling_fallback

_PATCH_ATTR = "_polling_failover_runtime_patch_installed"
_APP_MODULE_NAME = "nifty_scalper_bot.core.app"
_APP_MODULE_REF: Any | None = None
LOGGER = logging.getLogger("nifty_scalper_bot.core.app")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _safe_callable(value: Any, *, name: str, default: Any = None) -> tuple[bool, Any]:
    if callable(value):
        try:
            return True, value()
        except Exception as exc:  # noqa: BLE001 - supervisor must stay alive
            LOGGER.warning(
                "POLLING_SUPERVISOR_CALL_FAILED name=%s error_type=%s error=%s",
                name,
                type(exc).__name__,
                exc,
                extra={
                    "event": "POLLING_SUPERVISOR_CALL_FAILED",
                    "dependency": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False, default
    if value is not None and not isinstance(value, (bool, int, float, str)):
        LOGGER.warning(
            "POLLING_SUPERVISOR_NONCALLABLE name=%s value_type=%s",
            name,
            type(value).__name__,
            extra={
                "event": "POLLING_SUPERVISOR_NONCALLABLE",
                "dependency": name,
                "value_type": type(value).__name__,
            },
        )
    return False, value if isinstance(value, bool) else default


def _bool(payload: Mapping[str, Any], key: str, default: bool = True) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_feed_health(mdm: Any) -> Mapping[str, Any]:
    if mdm is None:
        return {}
    getter = getattr(mdm, "trading_feed_health", None)
    if not callable(getter):
        return {}
    try:
        value = getter()
    except Exception as exc:  # noqa: BLE001 - fallback decision must not crash WS path
        LOGGER.warning(
            "POLLING_FALLBACK_HEALTH_FAILED error_type=%s error=%s",
            type(exc).__name__,
            exc,
            extra={"event": "POLLING_FALLBACK_HEALTH_FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        return {}
    return value if isinstance(value, Mapping) else {}


def _safe_data_age_ms(mdm: Any) -> float | None:
    if mdm is None:
        return None
    getter = getattr(mdm, "data_age_ms", None)
    if not callable(getter):
        return None
    try:
        value = getter()
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001 - fallback decision must not crash WS path
        LOGGER.warning(
            "POLLING_FALLBACK_AGE_FAILED error_type=%s error=%s",
            type(exc).__name__,
            exc,
            extra={"event": "POLLING_FALLBACK_AGE_FAILED", "error_type": type(exc).__name__, "error": str(exc)},
        )
        return None


async def _stop_fallback_safely(fallback: Any, *, reason: str) -> None:
    try:
        running_fn = getattr(fallback, "is_running", None)
        running = bool(running_fn()) if callable(running_fn) else bool(getattr(fallback, "_running", False))
        if not running:
            # Already stopped: repeating set_websocket_mode(True) every healthy
            # supervisor iteration is redundant async churn / noisy telemetry.
            return
        mode_fn = getattr(fallback, "set_websocket_mode", None)
        if callable(mode_fn):
            await _maybe_await(mode_fn(True))
        stop_fn = getattr(fallback, "stop", None)
        if callable(stop_fn):
            await _maybe_await(stop_fn())
    except Exception as exc:  # noqa: BLE001 - supervisor must remain non-fatal
        LOGGER.warning(
            "POLLING_FALLBACK_STOP_FAILED reason=%s error_type=%s error=%s",
            reason,
            type(exc).__name__,
            exc,
            extra={"event": "POLLING_FALLBACK_STOP_FAILED", "reason": reason, "error_type": type(exc).__name__, "error": str(exc)},
        )


async def _start_fallback_safely(fallback: Any, *, decision_reason: str | None) -> None:
    try:
        mode_fn = getattr(fallback, "set_websocket_mode", None)
        if callable(mode_fn):
            await _maybe_await(mode_fn(False))
        running_fn = getattr(fallback, "is_running", None)
        running = bool(running_fn()) if callable(running_fn) else bool(getattr(fallback, "_running", False))
        if not running:
            start_fn = getattr(fallback, "start", None)
            if callable(start_fn):
                await _maybe_await(start_fn())
    except Exception as exc:  # noqa: BLE001 - never destabilize WS path
        LOGGER.warning(
            "POLLING_FALLBACK_START_FAILED reason=%s error_type=%s error=%s",
            decision_reason,
            type(exc).__name__,
            exc,
            extra={"event": "POLLING_FALLBACK_START_FAILED", "reason": decision_reason, "error_type": type(exc).__name__, "error": str(exc)},
        )


def _polling_fallback_degraded(
    *,
    ws_ok: bool,
    lagging: bool,
    futures_fresh: bool,
    options_fresh: bool,
    quote_stale_ms: float = 120000.0,
    feed_health: Mapping[str, Any] | None = None,
    data_age_ms: float | None = None,
) -> bool:
    """Runtime boolean wrapper around the age-gated fallback decision."""

    return decide_polling_fallback(
        ws_ok=ws_ok,
        lagging=lagging,
        futures_fresh=futures_fresh,
        options_fresh=options_fresh,
        quote_stale_ms=quote_stale_ms,
        feed_health=feed_health,
        data_age_ms=data_age_ms,
    ).activate


def _resolve_market_open_callable(ctx: Any, app_module: Any | None = None) -> Any:
    ctx_hook = getattr(ctx, "is_market_open_now", None)
    if ctx_hook is not None:
        return ctx_hook
    module = app_module or _APP_MODULE_REF or sys.modules.get(_APP_MODULE_NAME)
    return getattr(module, "is_market_open_now", None)


async def _polling_failover_supervisor_iteration(
    ctx: Any,
    fallback: Any,
    *,
    quote_stale_ms: float,
    degraded_since: float | None,
    recovered_since: float | None,
    activate_after: float,
    recover_cooldown: float = 10.0,
    _app_module: Any | None = None,
) -> tuple[float | None, float | None]:
    """One non-fatal polling failover supervisor iteration.

    Returns the updated ``(degraded_since, recovered_since)`` state.
    """

    _called, market_open = _safe_callable(
        _resolve_market_open_callable(ctx, _app_module),
        name="is_market_open_now",
        default=False,
    )
    if not bool(market_open):
        await _stop_fallback_safely(fallback, reason="market_closed")
        return None, time.monotonic()

    ws_mgr = getattr(ctx, "websocket_manager", None)
    is_connected = getattr(ws_mgr, "is_connected", None)
    _called_ws, ws_state = _safe_callable(
        is_connected,
        name="websocket_manager.is_connected",
        default=False,
    )
    ws_ok = bool(ws_state)
    mdm = getattr(ctx, "market_data_manager", None)
    feed_health = _safe_feed_health(mdm)
    data_age_ms = _safe_data_age_ms(mdm)
    lagging = bool(
        feed_health.get("lagging")
        or feed_health.get("event_loop_lagging")
        or getattr(ctx, "event_loop_lagging", False)
    )
    futures_fresh = _bool(feed_health, "futures_fresh", True)
    options_fresh = _bool(feed_health, "options_fresh", True)
    decision = decide_polling_fallback(
        ws_ok=ws_ok,
        lagging=lagging,
        futures_fresh=futures_fresh,
        options_fresh=options_fresh,
        quote_stale_ms=quote_stale_ms,
        feed_health=feed_health,
        data_age_ms=data_age_ms,
    )
    LOGGER.info(
        "POLLING_FALLBACK_DECISION activate=%s reason=%s ws_ok=%s lagging=%s futures_fresh=%s options_fresh=%s max_age_ms=%s threshold_ms=%s",
        decision.activate,
        decision.reason,
        decision.ws_ok,
        decision.lagging,
        decision.futures_fresh,
        decision.options_fresh,
        decision.max_age_ms,
        decision.threshold_ms,
        extra=decision.as_log_extra(),
    )
    now = time.monotonic()
    if not decision.activate:
        # Anti-flap hysteresis (mirrors core.app:_polling_failover_supervisor_
        # iteration): only stop the fallback after recover_cooldown seconds of
        # continuous recovery, so a briefly healthy feed cannot bounce the
        # REST poller on/off.
        recovered_since = recovered_since or now
        if now - recovered_since >= max(0.0, float(recover_cooldown or 0.0)):
            await _stop_fallback_safely(fallback, reason="feed_recovered")
        return None, recovered_since
    degraded_since = degraded_since or now
    if now - degraded_since >= max(0.0, float(activate_after or 0.0)):
        await _start_fallback_safely(fallback, decision_reason=decision.reason)
    return degraded_since, None


def apply_app_patch(app_module: Any) -> bool:
    """Install runtime polling fallback helpers on ``core.app``."""

    global _APP_MODULE_REF
    _APP_MODULE_REF = app_module
    if bool(getattr(app_module, _PATCH_ATTR, False)):
        return False

    async def _installed_polling_failover_supervisor_iteration(
        ctx: Any,
        fallback: Any,
        *,
        quote_stale_ms: float,
        degraded_since: float | None,
        recovered_since: float | None,
        activate_after: float,
        recover_cooldown: float = 10.0,
    ) -> tuple[float | None, float | None]:
        return await _polling_failover_supervisor_iteration(
            ctx,
            fallback,
            quote_stale_ms=quote_stale_ms,
            degraded_since=degraded_since,
            recovered_since=recovered_since,
            activate_after=activate_after,
            recover_cooldown=recover_cooldown,
            _app_module=app_module,
        )

    setattr(app_module, "_polling_fallback_degraded", _polling_fallback_degraded)
    setattr(
        app_module,
        "_polling_failover_supervisor_iteration",
        _installed_polling_failover_supervisor_iteration,
    )
    setattr(app_module, _PATCH_ATTR, True)
    return True


__all__ = [
    "apply_app_patch",
    "_polling_fallback_degraded",
    "_polling_failover_supervisor_iteration",
]
